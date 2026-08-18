"""Charbonnier + MS-SSIM + LPIPS combined training loss."""
import lpips
import torch
import torch.nn as nn
from pytorch_msssim import ms_ssim


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps2 = eps ** 2

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps2))


class MSSSIMLoss(nn.Module):
    """1 - MS-SSIM. Inputs are expected in [0, 1].

    pytorch_msssim's ms_ssim requires image_size > (win_size - 1) * 16
    regardless of how many weight levels are used (its internal 4-downsample
    bound). With the default win_size=11 that's >160px, larger than our
    128px GT training patches. We use win_size=7 (>96px required) so training
    can run on 128px patches; full 256px validation/inference is unaffected.
    """

    def __init__(self, data_range=1.0, win_size=7):
        super().__init__()
        self.data_range = data_range
        self.win_size = win_size

    def forward(self, pred, target):
        # ms_ssim divides by local variance internally; under fp16 autocast
        # on near-flat semiconductor background patches this can overflow to
        # NaN/Inf. Force fp32 here too, same rationale as PerceptualLoss.
        with torch.autocast(device_type=pred.device.type, enabled=False):
            pred = pred.float().clamp(0, 1)
            target = target.float().clamp(0, 1)
            return 1.0 - ms_ssim(pred, target, data_range=self.data_range,
                                  size_average=True, win_size=self.win_size)


class PerceptualLoss(nn.Module):
    """Thin, differentiable wrapper around LPIPS usable as a training loss.
    Expects single-channel [0, 1] input; internally expands to 3-channel
    [-1, 1] as LPIPS requires.
    """

    def __init__(self, net="alex"):
        super().__init__()
        self.model = lpips.LPIPS(net=net)
        for p in self.model.parameters():
            p.requires_grad_(False)

    def forward(self, pred, target):
        # LPIPS's VGG/AlexNet backbone is prone to NaN/Inf under fp16/bf16
        # autocast (squared-difference + normalization ops overflow in low
        # precision). Force this submodule to run in fp32 regardless of the
        # surrounding autocast context.
        with torch.autocast(device_type=pred.device.type, enabled=False):
            pred = pred.float().clamp(0, 1).repeat(1, 3, 1, 1) * 2 - 1
            target = target.float().clamp(0, 1).repeat(1, 3, 1, 1) * 2 - 1
            return self.model(pred, target).mean()


class CombinedRestorationLoss(nn.Module):
    def __init__(self, charbonnier_weight=1.0, ms_ssim_weight=0.2, lpips_weight=0.05,
                 charbonnier_eps=1e-3, lpips_net="alex"):
        super().__init__()
        self.charbonnier = CharbonnierLoss(eps=charbonnier_eps)
        self.ms_ssim = MSSSIMLoss()
        self.perceptual = PerceptualLoss(net=lpips_net)
        self.w_char = charbonnier_weight
        self.w_ssim = ms_ssim_weight
        self.w_lpips = lpips_weight

    def forward(self, pred, target):
        l_char = self.charbonnier(pred, target)
        l_ssim = self.ms_ssim(pred, target)
        l_lpips = self.perceptual(pred, target)

        total = self.w_char * l_char + self.w_ssim * l_ssim + self.w_lpips * l_lpips
        components = {
            "charbonnier": l_char.item(),
            "ms_ssim": l_ssim.item(),
            "lpips": l_lpips.item(),
            "total": total.item(),
        }
        return total, components

    @classmethod
    def from_config(cls, cfg):
        loss_cfg = cfg["loss"]
        return cls(
            charbonnier_weight=loss_cfg["charbonnier_weight"],
            ms_ssim_weight=loss_cfg["ms_ssim_weight"],
            lpips_weight=loss_cfg["lpips_weight"],
            charbonnier_eps=loss_cfg["charbonnier_eps"],
        )
