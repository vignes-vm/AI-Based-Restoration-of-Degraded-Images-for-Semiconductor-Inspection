"""Inspect a handful of GT / NoisyLR pairs from the KLA semiconductor dataset.

Dataset layout (confirmed by manual inspection):
    train/GT/<id>.npy        float32, shape (256, 256), range [0, 1]
    train/NoisyLR/<id>.npy   float32, shape (128, 128), range NOT clipped
                              (noise pushes values below 0 / above 1)
    NoisyLR/<id>.npy         hidden-test NoisyLR only, no paired GT

Usage:
    python src/data/inspect_dataset.py --config configs/base.yaml --n 6
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def list_ids(gt_dir, n):
    files = sorted(f for f in os.listdir(gt_dir) if f.endswith(".npy"))
    return [os.path.splitext(f)[0] for f in files[:n]]


def inspect(cfg, n, out_path):
    gt_dir = cfg["data"]["gt_dir"]
    nlr_dir = cfg["data"]["noisy_lr_dir"]
    ids = list_ids(gt_dir, n)
    if not ids:
        print(f"No .npy files found under {gt_dir}", file=sys.stderr)
        sys.exit(1)

    fig, axes = plt.subplots(len(ids), 2, figsize=(6, 3 * len(ids)))
    if len(ids) == 1:
        axes = axes[None, :]

    for row, img_id in enumerate(ids):
        gt = np.load(os.path.join(gt_dir, f"{img_id}.npy"))
        nlr = np.load(os.path.join(nlr_dir, f"{img_id}.npy"))

        print(f"[{img_id}] GT    shape={gt.shape}  dtype={gt.dtype}  "
              f"min={gt.min():.4f} max={gt.max():.4f}")
        print(f"[{img_id}] NoisyLR shape={nlr.shape}  dtype={nlr.dtype}  "
              f"min={nlr.min():.4f} max={nlr.max():.4f}")

        axes[row, 0].imshow(nlr, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title(f"{img_id} NoisyLR ({nlr.shape[0]}x{nlr.shape[1]})")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(gt, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title(f"{img_id} GT ({gt.shape[0]}x{gt.shape[1]})")
        axes[row, 1].axis("off")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"\nSaved comparison grid -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--out", default="results/samples/dataset_inspection.png")
    args = parser.parse_args()

    cfg = load_config(args.config)
    inspect(cfg, args.n, args.out)
