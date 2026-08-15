"""Paired GT / NoisyLR dataset with leak-free train/val split.

GT images are float32 in [0, 1]; NoisyLR images are float32 and NOT clipped
(degradation can push values outside [0, 1]) — we preserve that on purpose so
the model learns the real input distribution instead of a clipped proxy.
"""
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset


def list_paired_ids(gt_dir, noisy_lr_dir):
    gt_ids = {os.path.splitext(f)[0] for f in os.listdir(gt_dir) if f.endswith(".npy")}
    nlr_ids = {os.path.splitext(f)[0] for f in os.listdir(noisy_lr_dir) if f.endswith(".npy")}
    ids = sorted(gt_ids & nlr_ids)
    if len(ids) != len(gt_ids) or len(ids) != len(nlr_ids):
        missing_gt = nlr_ids - gt_ids
        missing_nlr = gt_ids - nlr_ids
        print(f"[dataset] WARNING: {len(missing_gt)} NoisyLR ids missing GT, "
              f"{len(missing_nlr)} GT ids missing NoisyLR — using intersection only.")
    return ids


def split_ids(ids, val_split, seed):
    """Split by image ID (not per-crop) so no image contributes to both splits."""
    ids = sorted(ids)
    rng = random.Random(seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_split)))
    val_ids = sorted(shuffled[:n_val])
    train_ids = sorted(shuffled[n_val:])
    return train_ids, val_ids


class PairedRestorationDataset(Dataset):
    """Loads (NoisyLR, GT) .npy pairs, with optional random-crop + flip/rotate augmentation.

    Args:
        gt_dir: directory of GT `<id>.npy` files, float32 [0, 1].
        noisy_lr_dir: directory of NoisyLR `<id>.npy` files, float32, unclipped.
        ids: list of image ids to include (from split_ids).
        scale_factor: GT resolution / NoisyLR resolution.
        patch_size: GT-space crop size. Must be a multiple of scale_factor. If
            None, returns full images uncropped (used for validation).
        augment: whether to apply random flip/rotation.
    """

    def __init__(self, gt_dir, noisy_lr_dir, ids, scale_factor=2, patch_size=128, augment=True):
        assert patch_size is None or patch_size % scale_factor == 0, \
            "patch_size must be a multiple of scale_factor"
        self.gt_dir = gt_dir
        self.noisy_lr_dir = noisy_lr_dir
        self.ids = ids
        self.scale_factor = scale_factor
        self.patch_size = patch_size
        self.augment = augment

    def __len__(self):
        return len(self.ids)

    def _random_crop(self, gt, nlr):
        gt_ps = self.patch_size
        lr_ps = gt_ps // self.scale_factor
        h_lr, w_lr = nlr.shape
        top_lr = random.randint(0, h_lr - lr_ps)
        left_lr = random.randint(0, w_lr - lr_ps)
        top_gt, left_gt = top_lr * self.scale_factor, left_lr * self.scale_factor

        nlr = nlr[top_lr:top_lr + lr_ps, left_lr:left_lr + lr_ps]
        gt = gt[top_gt:top_gt + gt_ps, left_gt:left_gt + gt_ps]
        return gt, nlr

    def _augment(self, gt, nlr):
        if random.random() < 0.5:
            gt, nlr = np.fliplr(gt), np.fliplr(nlr)
        if random.random() < 0.5:
            gt, nlr = np.flipud(gt), np.flipud(nlr)
        k = random.randint(0, 3)
        if k:
            gt, nlr = np.rot90(gt, k), np.rot90(nlr, k)
        return gt, nlr

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        gt = np.load(os.path.join(self.gt_dir, f"{img_id}.npy")).astype(np.float32)
        nlr = np.load(os.path.join(self.noisy_lr_dir, f"{img_id}.npy")).astype(np.float32)

        if self.patch_size is not None:
            gt, nlr = self._random_crop(gt, nlr)
        if self.augment:
            gt, nlr = self._augment(gt, nlr)

        gt_t = torch.from_numpy(np.ascontiguousarray(gt)).unsqueeze(0)
        nlr_t = torch.from_numpy(np.ascontiguousarray(nlr)).unsqueeze(0)
        return nlr_t, gt_t


class SyntheticDegradationDataset(Dataset):
    """Generates synthetic (NoisyLR, GT) pairs on the fly: random-crops a GT
    patch and degrades it with `RandomDegradationPipeline`. Used to mix
    synthetic augmentation into training alongside real provided pairs.
    """

    def __init__(self, gt_dir, ids, pipeline, scale_factor=2, patch_size=128, augment=True):
        assert patch_size % scale_factor == 0, "patch_size must be a multiple of scale_factor"
        self.gt_dir = gt_dir
        self.ids = ids
        self.pipeline = pipeline
        self.scale_factor = scale_factor
        self.patch_size = patch_size
        self.augment = augment

    def __len__(self):
        return len(self.ids)

    def _random_crop(self, gt):
        ps = self.patch_size
        h, w = gt.shape
        top = random.randint(0, h - ps)
        left = random.randint(0, w - ps)
        return gt[top:top + ps, left:left + ps]

    def _augment(self, gt):
        if random.random() < 0.5:
            gt = np.fliplr(gt)
        if random.random() < 0.5:
            gt = np.flipud(gt)
        k = random.randint(0, 3)
        if k:
            gt = np.rot90(gt, k)
        return gt

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        gt = np.load(os.path.join(self.gt_dir, f"{img_id}.npy")).astype(np.float32)
        gt = self._random_crop(gt)
        if self.augment:
            gt = self._augment(gt)
        gt = np.ascontiguousarray(gt)
        nlr = self.pipeline(gt)

        gt_t = torch.from_numpy(gt).unsqueeze(0)
        nlr_t = torch.from_numpy(np.ascontiguousarray(nlr)).unsqueeze(0)
        return nlr_t, gt_t


class UnpairedNoisyLRDataset(Dataset):
    """Loads NoisyLR-only .npy files (the hidden test set, no GT available)."""

    def __init__(self, noisy_lr_dir):
        self.noisy_lr_dir = noisy_lr_dir
        self.ids = sorted(os.path.splitext(f)[0] for f in os.listdir(noisy_lr_dir) if f.endswith(".npy"))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        nlr = np.load(os.path.join(self.noisy_lr_dir, f"{img_id}.npy")).astype(np.float32)
        nlr_t = torch.from_numpy(np.ascontiguousarray(nlr)).unsqueeze(0)
        return nlr_t, img_id


if __name__ == "__main__":
    import yaml
    from torch.utils.data import DataLoader

    with open("configs/base.yaml") as f:
        cfg = yaml.safe_load(f)

    d = cfg["data"]
    ids = list_paired_ids(d["gt_dir"], d["noisy_lr_dir"])
    train_ids, val_ids = split_ids(ids, d["val_split"], d["val_split_seed"])
    print(f"Total paired ids: {len(ids)} | train: {len(train_ids)} | val: {len(val_ids)}")
    assert set(train_ids).isdisjoint(val_ids), "train/val leakage detected!"

    train_ds = PairedRestorationDataset(
        d["gt_dir"], d["noisy_lr_dir"], train_ids,
        scale_factor=d["scale_factor"], patch_size=d["patch_size"], augment=True,
    )
    loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    nlr, gt = next(iter(loader))
    print(f"NoisyLR batch: shape={tuple(nlr.shape)} dtype={nlr.dtype} "
          f"min={nlr.min():.4f} max={nlr.max():.4f}")
    print(f"GT batch:      shape={tuple(gt.shape)} dtype={gt.dtype} "
          f"min={gt.min():.4f} max={gt.max():.4f}")
