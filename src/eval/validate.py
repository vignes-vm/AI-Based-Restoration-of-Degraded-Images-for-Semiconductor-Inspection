"""Run full validation for a restoration checkpoint: per-image + aggregate
PSNR/SSIM/LPIPS, plus a qualitative grid of best/worst/random examples.

Usage:
    python -m src.eval.validate --config configs/base.yaml \
        --checkpoint weights/baseline_best.pth --model_type baseline --run_name baseline
"""
import argparse
import json
import os
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import PairedRestorationDataset, list_paired_ids, split_ids
from src.device import resolve_device
from src.eval.metrics import compute_lpips, compute_psnr, compute_ssim


def build_model(model_type, cfg, device):
    if model_type == "baseline":
        from src.models.baseline_unet import BaselineUNet
        m = cfg["baseline"]
        model = BaselineUNet(in_channels=cfg["data"]["in_channels"],
                              base_channels=m["base_channels"],
                              scale_factor=cfg["data"]["scale_factor"])
    elif model_type == "nafnet_sr":
        from src.models.nafnet_sr import NAFNetSR
        m = cfg["model"]
        model = NAFNetSR(in_channels=cfg["data"]["in_channels"],
                          width=m["width"], enc_blocks=m["enc_blocks"],
                          middle_blocks=m["middle_blocks"], dec_blocks=m["dec_blocks"],
                          scale_factor=cfg["data"]["scale_factor"])
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    return model.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model_type", required=True, choices=["baseline", "nafnet_sr"])
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--lpips_net", default="alex")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = resolve_device(cfg["train"]["device"])
    d = cfg["data"]

    ids = list_paired_ids(d["gt_dir"], d["noisy_lr_dir"])
    _, val_ids = split_ids(ids, d["val_split"], d["val_split_seed"])
    val_ds = PairedRestorationDataset(d["gt_dir"], d["noisy_lr_dir"], val_ids,
                                       scale_factor=d["scale_factor"], patch_size=None, augment=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    model = build_model(args.model_type, cfg, device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    per_image = []
    examples = []  # (img_id, nlr, pred, gt, psnr)
    with torch.no_grad():
        for i, (nlr, gt) in enumerate(tqdm(val_loader, desc=f"validating[{args.run_name}]")):
            img_id = val_ids[i]
            nlr, gt = nlr.to(device), gt.to(device)
            pred = model(nlr).clamp(0, 1)

            pred_np = pred.squeeze().cpu().numpy()
            gt_np = gt.squeeze().cpu().numpy()

            psnr = compute_psnr(pred_np, gt_np)
            ssim = compute_ssim(pred_np, gt_np)
            lpips_val = compute_lpips(pred.cpu(), gt.cpu(), net=args.lpips_net, device="cpu")

            per_image.append({"id": img_id, "psnr": psnr, "ssim": ssim, "lpips": lpips_val})
            examples.append((img_id, nlr.squeeze().cpu().numpy(), pred_np, gt_np, psnr))

    psnrs = [r["psnr"] for r in per_image]
    ssims = [r["ssim"] for r in per_image]
    lpipss = [r["lpips"] for r in per_image]

    aggregate = {
        "psnr_mean": float(np.mean(psnrs)), "psnr_std": float(np.std(psnrs)),
        "ssim_mean": float(np.mean(ssims)), "ssim_std": float(np.std(ssims)),
        "lpips_mean": float(np.mean(lpipss)), "lpips_std": float(np.std(lpipss)),
        "n_images": len(per_image),
    }
    print(f"[{args.run_name}] PSNR={aggregate['psnr_mean']:.3f}±{aggregate['psnr_std']:.3f}  "
          f"SSIM={aggregate['ssim_mean']:.4f}±{aggregate['ssim_std']:.4f}  "
          f"LPIPS={aggregate['lpips_mean']:.4f}±{aggregate['lpips_std']:.4f}")

    os.makedirs(cfg["paths"]["results_dir"], exist_ok=True)
    out_json = os.path.join(cfg["paths"]["results_dir"], f"metrics_{args.run_name}.json")
    with open(out_json, "w") as f:
        json.dump({"aggregate": aggregate, "per_image": per_image}, f, indent=2)
    print(f"Saved metrics -> {out_json}")

    # qualitative grid: best, worst, 3 random
    examples_sorted = sorted(examples, key=lambda e: e[4])
    worst = examples_sorted[0]
    best = examples_sorted[-1]
    rest = examples_sorted[1:-1]
    random.seed(cfg["seed"])
    rand_examples = random.sample(rest, min(3, len(rest)))
    picks = [("best", best), ("worst", worst)] + [(f"random{i}", e) for i, e in enumerate(rand_examples)]

    fig, axes = plt.subplots(len(picks), 3, figsize=(9, 3 * len(picks)))
    for row, (label, (img_id, nlr, pred, gt, psnr)) in enumerate(picks):
        axes[row, 0].imshow(nlr, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title(f"{img_id} NoisyLR")
        axes[row, 1].imshow(pred, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title(f"{label} restored (PSNR={psnr:.2f})")
        axes[row, 2].imshow(gt, cmap="gray", vmin=0, vmax=1)
        axes[row, 2].set_title("GT")
        for c in range(3):
            axes[row, c].axis("off")

    out_png = os.path.join(cfg["paths"]["results_dir"], "samples", f"val_examples_{args.run_name}.png")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    print(f"Saved qualitative examples -> {out_png}")


if __name__ == "__main__":
    main()
