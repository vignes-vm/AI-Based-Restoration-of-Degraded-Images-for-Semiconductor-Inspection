"""Full reproducible training script for NAFNet-SR.

Usage:
    python train.py --config configs/base.yaml
"""
import argparse
import datetime
import itertools
import math
import os
import random
import time

import numpy as np
import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.data.degradations import RandomDegradationPipeline
from src.data.dataset import (
    PairedRestorationDataset,
    SyntheticDegradationDataset,
    list_paired_ids,
    split_ids,
)
from src.device import resolve_device
from src.eval.metrics import compute_psnr, compute_ssim
from src.losses.losses import CombinedRestorationLoss
from src.models.nafnet_sr import NAFNetSR


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_lr_lambda(warmup_epochs, total_epochs, min_lr_ratio):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine
    return lr_lambda


@torch.no_grad()
def quick_validate(model, val_loader, device, max_batches=20):
    model.eval()
    psnrs, ssims = [], []
    for i, (nlr, gt) in enumerate(val_loader):
        if i >= max_batches:
            break
        nlr, gt = nlr.to(device), gt.to(device)
        pred = model(nlr).clamp(0, 1)
        for b in range(pred.shape[0]):
            p = pred[b, 0].cpu().numpy()
            g = gt[b, 0].cpu().numpy()
            psnrs.append(compute_psnr(p, g))
            ssims.append(compute_ssim(p, g))
    model.train()
    return float(np.mean(psnrs)), float(np.mean(ssims))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="override configs/base.yaml train.epochs")
    parser.add_argument("--run_name", default="nafnet_sr")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = cfg["seed"]
    set_seed(seed)
    print(f"[seed] using fixed random seed = {seed}")

    d, t, m = cfg["data"], cfg["train"], cfg["model"]
    epochs = args.epochs if args.epochs is not None else t["epochs"]
    device = resolve_device(t["device"])
    print(f"[device] using {device}")

    # --- data: real pairs + synthetic-degradation pairs mixed by ratio ---
    ids = list_paired_ids(d["gt_dir"], d["noisy_lr_dir"])
    train_ids, val_ids = split_ids(ids, d["val_split"], d["val_split_seed"])
    print(f"[data] train={len(train_ids)} val={len(val_ids)}  synthetic_ratio={d['synthetic_ratio']}")

    real_ds = PairedRestorationDataset(d["gt_dir"], d["noisy_lr_dir"], train_ids,
                                        scale_factor=d["scale_factor"], patch_size=d["patch_size"], augment=True)
    pipeline = RandomDegradationPipeline.from_config(cfg)
    synth_ds = SyntheticDegradationDataset(d["gt_dir"], train_ids, pipeline,
                                            scale_factor=d["scale_factor"], patch_size=d["patch_size"], augment=True)

    real_loader = DataLoader(real_ds, batch_size=t["batch_size"], shuffle=True,
                              num_workers=d["num_workers"], drop_last=True)
    synth_loader = DataLoader(synth_ds, batch_size=t["batch_size"], shuffle=True,
                               num_workers=d["num_workers"], drop_last=True)

    val_ds = PairedRestorationDataset(d["gt_dir"], d["noisy_lr_dir"], val_ids,
                                       scale_factor=d["scale_factor"], patch_size=None, augment=False)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=d["num_workers"])

    steps_per_epoch = len(real_loader)

    # --- model / loss / optim ---
    model = NAFNetSR(in_channels=d["in_channels"], width=m["width"], enc_blocks=m["enc_blocks"],
                      middle_blocks=m["middle_blocks"], dec_blocks=m["dec_blocks"],
                      scale_factor=d["scale_factor"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] NAFNetSR params: {n_params / 1e6:.2f}M")

    criterion = CombinedRestorationLoss.from_config(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    min_lr_ratio = t["min_lr"] / t["lr"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, build_lr_lambda(t["warmup_epochs"], epochs, min_lr_ratio))

    amp_enabled = t["amp"] and device.type in ("cuda", "mps", "cpu")
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    scaler = torch.amp.GradScaler(enabled=(amp_enabled and device.type == "cuda"))

    os.makedirs(cfg["paths"]["weights_dir"], exist_ok=True)
    writer = SummaryWriter(os.path.join(cfg["paths"]["log_dir"], args.run_name))

    # snapshot exact config used for this run, for reproducibility
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_cfg_path = os.path.join("configs", f"run_{timestamp}.yaml")
    with open(run_cfg_path, "w") as f:
        yaml.safe_dump({**cfg, "run_name": args.run_name, "resolved_epochs": epochs,
                         "resolved_device": str(device)}, f)
    print(f"[config] snapshot saved -> {run_cfg_path}")

    best_score = -float("inf")
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        epoch_components = {"charbonnier": 0.0, "ms_ssim": 0.0, "lpips": 0.0, "total": 0.0}

        real_iter = iter(real_loader)
        synth_iter = itertools.cycle(synth_loader)

        for step in range(steps_per_epoch):
            use_synth = random.random() < d["synthetic_ratio"]
            nlr, gt = next(synth_iter) if use_synth else next(real_iter)
            nlr, gt = nlr.to(device), gt.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                pred = model(nlr)
                loss, components = criterion(pred, gt)

            if not torch.isfinite(loss):
                # A single NaN/Inf loss (e.g. from an unstable LPIPS/MS-SSIM
                # step) would otherwise poison every weight via backward().
                # Skip this step entirely instead of corrupting the model.
                print(f"[warn] non-finite loss at step {global_step} "
                      f"(components={components}) — skipping optimizer step")
                global_step += 1
                continue

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            for k in epoch_components:
                epoch_components[k] += components[k]

            if global_step % t["log_every_steps"] == 0:
                for k, v in components.items():
                    writer.add_scalar(f"train_step/{k}", v, global_step)
                writer.add_scalar("train_step/lr", optimizer.param_groups[0]["lr"], global_step)
            global_step += 1

        scheduler.step()

        for k in epoch_components:
            epoch_components[k] /= steps_per_epoch
            writer.add_scalar(f"train_epoch/{k}", epoch_components[k], epoch)

        val_psnr, val_ssim = quick_validate(model, val_loader, device)
        writer.add_scalar("val/psnr", val_psnr, epoch)
        writer.add_scalar("val/ssim", val_ssim, epoch)

        dt = time.time() - t0
        print(f"[epoch {epoch:03d}/{epochs}] loss={epoch_components['total']:.4f} "
              f"(char={epoch_components['charbonnier']:.4f} ssim_l={epoch_components['ms_ssim']:.4f} "
              f"lpips={epoch_components['lpips']:.4f})  val_psnr={val_psnr:.2f}  val_ssim={val_ssim:.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}  ({dt:.1f}s)")

        if epoch % t["checkpoint_every"] == 0 or epoch == epochs:
            ckpt_path = os.path.join(cfg["paths"]["weights_dir"], f"nafnet_sr_epoch{epoch}.pth")
            torch.save({"model": model.state_dict(), "epoch": epoch, "config": cfg}, ckpt_path)
            print(f"[checkpoint] saved -> {ckpt_path}")

        score = val_psnr + val_ssim * 10  # combine on comparable scales
        if math.isnan(score):
            print(f"[warn] epoch {epoch}: NaN validation score — not eligible as best checkpoint")
        elif score > best_score:
            best_score = score
            best_path = os.path.join(cfg["paths"]["weights_dir"], "nafnet_sr_best.pth")
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_psnr": val_psnr,
                        "val_ssim": val_ssim, "config": cfg}, best_path)
            print(f"[best] new best (psnr={val_psnr:.2f}, ssim={val_ssim:.4f}) -> {best_path}")

    writer.close()
    print("[done]")


if __name__ == "__main__":
    main()
