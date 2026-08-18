"""Restore all .npy images in <input-dir>, write results to <output-dir>.

Usage:
    python run.py <input-dir> <output-dir>

Reads every .npy file in the input directory, restores it with NAFNet-SR
(NAFNet-style encoder/bottleneck/decoder + sub-pixel upsample head, trained
to remove speckle/Gaussian noise and undo 2x downsampling), and writes one
output .npy per input, same filename, grayscale (H, W) array with values in
[0, 1] and no NaN/Inf. Runs on GPU if available, falls back to CPU otherwise.
No internet access, API keys, or manual configuration required.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.device import resolve_device
from src.models.nafnet_sr import NAFNetSR

CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "nafnet_sr_best.pth")


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


def main():
    if len(sys.argv) != 3:
        print("Usage: python run.py <input-dir> <output-dir>")
        sys.exit(1)
    input_dir, output_dir = sys.argv[1], sys.argv[2]

    device = resolve_device("cuda")
    print(f"[run] device={device}")

    model = load_model(CHECKPOINT_PATH, device)

    os.makedirs(output_dir, exist_ok=True)

    filenames = sorted(f for f in os.listdir(input_dir) if f.lower().endswith(".npy"))
    if not filenames:
        print(f"[run] no .npy files found in {input_dir}")
        return
    print(f"[run] found {len(filenames)} input files")

    for fname in filenames:
        arr = np.load(os.path.join(input_dir, fname)).astype(np.float32)
        if arr.ndim == 3:
            arr = arr[..., 0]

        x = torch.from_numpy(arr[None, None, :, :]).float().to(device)
        with torch.no_grad():
            out = model(x)
        out_np = out.float().cpu().numpy()[0, 0, :, :]

        out_np = np.nan_to_num(out_np, nan=0.0, posinf=1.0, neginf=0.0)
        out_np = np.clip(out_np, 0.0, 1.0).astype(np.float32)

        np.save(os.path.join(output_dir, fname), out_np)

    print(f"[run] wrote {len(filenames)} restored images to {output_dir}")


if __name__ == "__main__":
    main()
