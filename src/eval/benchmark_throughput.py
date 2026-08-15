"""Benchmark the full inference pipeline (disk read -> preprocess -> GPU
transfer -> forward -> transfer back -> postprocess -> save) at several batch
sizes, in both FP32 and FP16, and report images/sec, ms/image, and peak
device memory.

Usage:
    python -m src.eval.benchmark_throughput --checkpoint weights/nafnet_sr_best.pth \
        --input_dir NoisyLR --n_images 64
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import yaml

from src.device import resolve_device
from src.models.nafnet_sr import NAFNetSR


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


def peak_memory_mb(device):
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1e6
    return None  # not exposed for mps/cpu


def reset_memory_stats(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def run_benchmark(model, device, file_paths, batch_size, fp16, out_dir):
    dtype = torch.float16 if fp16 else torch.float32
    model_cast = model.half() if fp16 else model.float()

    reset_memory_stats(device)
    n = len(file_paths)
    t_start = time.time()

    for i in range(0, n, batch_size):
        batch_paths = file_paths[i:i + batch_size]

        arrays = [np.load(p).astype(np.float32) for p in batch_paths]
        batch_np = np.stack(arrays, axis=0)[:, None, :, :]
        batch_t = torch.from_numpy(batch_np).to(dtype).to(device)

        with torch.no_grad():
            out = model_cast(batch_t)
        sync(device)

        out_np = out.float().cpu().numpy()
        if out_dir:
            for path, img in zip(batch_paths, out_np[:, 0]):
                np.save(os.path.join(out_dir, os.path.basename(path)), np.clip(img, 0, 1))

    sync(device)
    total_time = time.time() - t_start
    images_per_sec = n / total_time
    ms_per_image = 1000.0 * total_time / n
    mem_mb = peak_memory_mb(device)

    model.float()  # restore fp32 weights for next configuration
    return {
        "batch_size": batch_size, "fp16": fp16, "n_images": n,
        "total_time_s": total_time, "images_per_sec": images_per_sec,
        "ms_per_image": ms_per_image, "peak_memory_mb": mem_mb,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint", default="weights/nafnet_sr_best.pth")
    parser.add_argument("--input_dir", default=None, help="defaults to configs/base.yaml data.test_noisy_lr_dir")
    parser.add_argument("--n_images", type=int, default=64)
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    input_dir = args.input_dir or cfg["data"]["test_noisy_lr_dir"]
    device = resolve_device(args.device)
    print(f"[benchmark] device={device}")

    model = load_model(args.checkpoint, device)

    all_files = sorted(os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".npy"))
    file_paths = all_files[:args.n_images] if args.n_images else all_files
    print(f"[benchmark] benchmarking on {len(file_paths)} images")

    results = []
    for bs in args.batch_sizes:
        if bs > len(file_paths):
            continue
        for fp16 in ([False, True] if device.type in ("cuda", "mps") else [False]):
            r = run_benchmark(model, device, file_paths, bs, fp16, out_dir=None)
            print(f"  bs={bs:>3} fp16={fp16!s:>5}  "
                  f"{r['images_per_sec']:.2f} img/s  {r['ms_per_image']:.1f} ms/img  "
                  f"mem={r['peak_memory_mb']}")
            results.append(r)

    os.makedirs(cfg["paths"]["results_dir"], exist_ok=True)
    json_path = os.path.join(cfg["paths"]["results_dir"], "throughput_benchmark.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[benchmark] saved -> {json_path}")

    md_path = os.path.join(cfg["paths"]["results_dir"], "throughput_benchmark.md")
    with open(md_path, "w") as f:
        f.write(f"# Throughput Benchmark\n\nDevice: `{device}`\n\n")
        f.write("| batch_size | precision | images/sec | ms/image | peak mem (MB) |\n")
        f.write("|---|---|---|---|---|\n")
        for r in results:
            prec = "FP16" if r["fp16"] else "FP32"
            mem = f"{r['peak_memory_mb']:.1f}" if r["peak_memory_mb"] is not None else "n/a"
            f.write(f"| {r['batch_size']} | {prec} | {r['images_per_sec']:.2f} | "
                    f"{r['ms_per_image']:.1f} | {mem} |\n")
    print(f"[benchmark] saved -> {md_path}")


if __name__ == "__main__":
    main()
