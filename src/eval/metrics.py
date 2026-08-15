"""PSNR / SSIM / LPIPS metrics for restored-vs-GT single-channel images.

All functions expect numpy float arrays in [0, 1] of shape (H, W), except
`compute_lpips`, which expects torch tensors of shape (1, 1, H, W) in [0, 1]
(LPIPS internally replicates the single channel to 3-channel RGB).
"""
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

_LPIPS_MODEL = None


def compute_psnr(pred, gt):
    pred = np.clip(pred, 0.0, 1.0)
    gt = np.clip(gt, 0.0, 1.0)
    return float(peak_signal_noise_ratio(gt, pred, data_range=1.0))


def compute_ssim(pred, gt):
    pred = np.clip(pred, 0.0, 1.0)
    gt = np.clip(gt, 0.0, 1.0)
    return float(structural_similarity(gt, pred, data_range=1.0))


def _get_lpips_model(net="alex", device="cpu"):
    global _LPIPS_MODEL
    if _LPIPS_MODEL is None:
        import lpips
        _LPIPS_MODEL = lpips.LPIPS(net=net).to(device)
        _LPIPS_MODEL.eval()
    return _LPIPS_MODEL


@torch.no_grad()
def compute_lpips(pred, gt, net="alex", device="cpu"):
    """pred, gt: torch tensors (1, 1, H, W) in [0, 1]."""
    model = _get_lpips_model(net, device)
    pred = pred.clamp(0, 1).to(device)
    gt = gt.clamp(0, 1).to(device)
    # LPIPS expects 3-channel input in [-1, 1]
    pred3 = pred.repeat(1, 3, 1, 1) * 2 - 1
    gt3 = gt.repeat(1, 3, 1, 1) * 2 - 1
    return float(model(pred3, gt3).item())
