"""Train the lightweight U-Net baseline (comparison reference, not the final model).

Usage:
    python train_baseline.py --config configs/base.yaml
"""
import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.data.dataset import PairedRestorationDataset, list_paired_ids, split_ids
from src.device import resolve_device
from src.models.baseline_unet import BaselineUNet


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="override configs/base.yaml train.baseline.epochs")
    parser.add_argument("--run_name", default="baseline")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = cfg["seed"]
    set_seed(seed)
    print(f"[seed] using fixed random seed = {seed}")

    d, b = cfg["data"], cfg["baseline"]
    epochs = args.epochs if args.epochs is not None else b["epochs"]
    device = resolve_device(cfg["train"]["device"])
    print(f"[device] using {device}")

    ids = list_paired_ids(d["gt_dir"], d["noisy_lr_dir"])
    train_ids, val_ids = split_ids(ids, d["val_split"], d["val_split_seed"])
    print(f"[data] train={len(train_ids)} val={len(val_ids)}")

    train_ds = PairedRestorationDataset(d["gt_dir"], d["noisy_lr_dir"], train_ids,
                                         scale_factor=d["scale_factor"], patch_size=d["patch_size"], augment=True)
    val_ds = PairedRestorationDataset(d["gt_dir"], d["noisy_lr_dir"], val_ids,
                                       scale_factor=d["scale_factor"], patch_size=d["patch_size"], augment=False)

    train_loader = DataLoader(train_ds, batch_size=b["batch_size"], shuffle=True,
                               num_workers=d["num_workers"], drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=b["batch_size"], shuffle=False,
                             num_workers=d["num_workers"])

    model = BaselineUNet(in_channels=d["in_channels"], base_channels=b["base_channels"],
                          scale_factor=d["scale_factor"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=b["lr"])
    criterion = nn.L1Loss()

    os.makedirs(cfg["paths"]["weights_dir"], exist_ok=True)
    log_dir = os.path.join(cfg["paths"]["log_dir"], args.run_name)
    writer = SummaryWriter(log_dir)

    best_val_loss = float("inf")
    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        train_loss_sum = 0.0
        for nlr, gt in train_loader:
            nlr, gt = nlr.to(device), gt.to(device)
            opt.zero_grad()
            pred = model(nlr)
            loss = criterion(pred, gt)
            loss.backward()
            opt.step()

            train_loss_sum += loss.item() * nlr.size(0)
            writer.add_scalar("train/l1_loss_step", loss.item(), global_step)
            global_step += 1

        train_loss = train_loss_sum / len(train_ds)

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for nlr, gt in val_loader:
                nlr, gt = nlr.to(device), gt.to(device)
                pred = model(nlr)
                val_loss_sum += criterion(pred, gt).item() * nlr.size(0)
        val_loss = val_loss_sum / len(val_ds)

        writer.add_scalar("train/l1_loss_epoch", train_loss, epoch)
        writer.add_scalar("val/l1_loss_epoch", val_loss, epoch)
        dt = time.time() - t0
        print(f"[epoch {epoch:03d}/{epochs}] train_l1={train_loss:.4f}  val_l1={val_loss:.4f}  ({dt:.1f}s)")

        if epoch % 10 == 0 or epoch == epochs:
            ckpt_path = os.path.join(cfg["paths"]["weights_dir"], f"baseline_epoch{epoch}.pth")
            torch.save({"model": model.state_dict(), "epoch": epoch, "config": cfg}, ckpt_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(cfg["paths"]["weights_dir"], "baseline_best.pth")
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_l1": val_loss, "config": cfg}, best_path)

    writer.close()
    print(f"[done] best val_l1={best_val_loss:.4f}")


if __name__ == "__main__":
    main()
