"""Standalone inference: restore every image under --input_dir, save results
(same filename, same format) under --output_dir.

Supports .npy (float32 grayscale arrays, the format used by this dataset) and
common raster formats (.png/.jpg/.jpeg/.tif/.tiff/.bmp, saved as 8-bit
grayscale). Runs FP16 by default; pass --fp32 to disable.

Output values ARE clipped to [0, 1] (and rescaled to [0, 255] uint8 for raster
formats) before saving. The provided NoisyLR data is NOT clipped, but KLA's
evaluation does not clip on their end either, so this script performs the
clip explicitly here rather than relying on downstream behavior.

Usage:
    python inference.py --input_dir NoisyLR --output_dir results/restored \
        --checkpoint weights/nafnet_sr_best.pth
"""
import argparse
import os
import time

import numpy as np
import torch
import yaml
from PIL import Image

from src.device import resolve_device
from src.models.nafnet_sr import NAFNetSR

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
NPY_EXT = ".npy"


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["config"]
    m, d = cfg["model"], cfg["data"]
    model = NAFNetSR(in_channels=d["in_channels"], width=m["width"], enc_blocks=m["enc_blocks"],
                      middle_blocks=m["middle_blocks"], dec_blocks=m["dec_blocks"],
                      scale_factor=d["scale_factor"])
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model


def load_input(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == NPY_EXT:
        arr = np.load(path).astype(np.float32)
        return arr, "npy"
    elif ext in IMAGE_EXTS:
        img = Image.open(path).convert("L")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr, "image"
    else:
        raise ValueError(f"Unsupported input file type: {path}")


def save_output(arr, out_path, fmt):
    arr = np.clip(arr, 0.0, 1.0)
    if fmt == "npy":
        np.save(out_path, arr.astype(np.float32))
    else:
        img = Image.fromarray((arr * 255.0).round().astype(np.uint8), mode="L")
        img.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--checkpoint", default="weights/nafnet_sr_best.pth")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp32", action="store_true", help="disable FP16 (default is FP16)")
    args = parser.parse_args()

    t_start = time.time()

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found at {args.checkpoint}. Run train.py first, "
            f"or point --checkpoint at a valid weights/*.pth file.")

    device = resolve_device(args.device)
    use_fp16 = (not args.fp32) and device.type in ("cuda", "mps")
    print(f"[inference] device={device}  fp16={use_fp16}  batch_size={args.batch_size}")

    t_load0 = time.time()
    model = load_model(args.checkpoint, device)
    if use_fp16:
        model = model.half()
    t_load = time.time() - t_load0
    print(f"[inference] model loaded in {t_load:.2f}s")

    os.makedirs(args.output_dir, exist_ok=True)
    filenames = sorted(f for f in os.listdir(args.input_dir)
                        if os.path.splitext(f)[1].lower() in IMAGE_EXTS | {NPY_EXT})
    if not filenames:
        print(f"[inference] no supported files found in {args.input_dir}")
        return
    print(f"[inference] found {len(filenames)} input files")

    total_io_time = 0.0
    total_preprocess_time = 0.0
    total_forward_time = 0.0
    total_postprocess_time = 0.0
    n_processed = 0

    batch_size = args.batch_size
    for batch_start in range(0, len(filenames), batch_size):
        t_batch0 = time.time()
        batch_files = filenames[batch_start:batch_start + batch_size]

        t0 = time.time()
        arrays, fmts = [], []
        for fname in batch_files:
            arr, fmt = load_input(os.path.join(args.input_dir, fname))
            arrays.append(arr)
            fmts.append(fmt)
        total_io_time += time.time() - t0

        # group by shape so we can batch the forward pass; mismatched shapes
        # in the same chunk fall back to a per-image forward pass.
        shapes = {a.shape for a in arrays}
        t0 = time.time()
        if len(shapes) == 1:
            batch_np = np.stack(arrays, axis=0)[:, None, :, :]  # (B, 1, H, W)
            batch_t = torch.from_numpy(batch_np)
            batch_t = batch_t.half() if use_fp16 else batch_t.float()
            batch_t = batch_t.to(device)
            total_preprocess_time += time.time() - t0

            t1 = time.time()
            with torch.no_grad():
                out = model(batch_t)
            if device.type in ("cuda", "mps"):
                torch.mps.synchronize() if device.type == "mps" else torch.cuda.synchronize()
            total_forward_time += time.time() - t1

            t2 = time.time()
            out_np = out.float().cpu().numpy()[:, 0, :, :]
            total_postprocess_time += time.time() - t2
        else:
            out_list = []
            for arr in arrays:
                t1 = time.time()
                single = torch.from_numpy(arr[None, None, :, :])
                single = single.half() if use_fp16 else single.float()
                single = single.to(device)
                total_preprocess_time += time.time() - t1

                t2 = time.time()
                with torch.no_grad():
                    o = model(single)
                total_forward_time += time.time() - t2

                t3 = time.time()
                out_list.append(o.float().cpu().numpy()[0, 0, :, :])
                total_postprocess_time += time.time() - t3
            out_np = out_list

        t0 = time.time()
        for fname, out_img, fmt in zip(batch_files, out_np, fmts):
            out_path = os.path.join(args.output_dir, fname)
            save_output(out_img, out_path, fmt)
        total_io_time += time.time() - t0

        n_processed += len(batch_files)
        print(f"[batch {batch_start // batch_size:04d}] {len(batch_files)} images "
              f"in {time.time() - t_batch0:.3f}s")

    t_total = time.time() - t_start
    print("\n=== Inference summary ===")
    print(f"Images processed:     {n_processed}")
    print(f"Model load time:      {t_load:.3f}s")
    print(f"Disk I/O time:        {total_io_time:.3f}s")
    print(f"Preprocess time:      {total_preprocess_time:.3f}s")
    print(f"Forward pass time:    {total_forward_time:.3f}s")
    print(f"Postprocess time:     {total_postprocess_time:.3f}s")
    print(f"Total end-to-end:     {t_total:.3f}s ({n_processed / t_total:.2f} images/sec)")


if __name__ == "__main__":
    main()
