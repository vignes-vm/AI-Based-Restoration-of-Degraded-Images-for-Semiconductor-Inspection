"""Visually sanity-check RandomDegradationPipeline against real NoisyLR examples.

Usage:
    python src/data/preview_degradations.py --config configs/base.yaml --n 4 --seeds 3
"""
import argparse
import os
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.data.degradations import RandomDegradationPipeline


def main(cfg, n, n_seeds, out_path):
    gt_dir = cfg["data"]["gt_dir"]
    nlr_dir = cfg["data"]["noisy_lr_dir"]
    ids = sorted(f.split(".")[0] for f in os.listdir(gt_dir) if f.endswith(".npy"))[:n]

    pipeline = RandomDegradationPipeline.from_config(cfg)

    n_cols = 2 + n_seeds  # real NoisyLR, GT, then n_seeds synthetic variants
    fig, axes = plt.subplots(n, n_cols, figsize=(3 * n_cols, 3 * n))
    if n == 1:
        axes = axes[None, :]

    for row, img_id in enumerate(ids):
        gt = np.load(os.path.join(gt_dir, f"{img_id}.npy")).astype(np.float32)
        real_nlr = np.load(os.path.join(nlr_dir, f"{img_id}.npy")).astype(np.float32)

        axes[row, 0].imshow(real_nlr, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title(f"{img_id} real NoisyLR")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(gt, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title(f"{img_id} GT")
        axes[row, 1].axis("off")

        for s in range(n_seeds):
            random.seed(1000 * s + row)
            np.random.seed(1000 * s + row)
            synth = pipeline(gt)
            axes[row, 2 + s].imshow(synth, cmap="gray", vmin=0, vmax=1)
            axes[row, 2 + s].set_title(f"synthetic seed={s}")
            axes[row, 2 + s].axis("off")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"Saved degradation preview -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--out", default="results/samples/degradation_preview.png")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    main(cfg, args.n, args.seeds, args.out)
