"""Randomized synthetic degradation pipeline (BSRGAN/Real-ESRGAN style shuffle).

Operates on numpy float32 arrays in [0, 1], shape (H, W). Individual
degradation ops are NOT clipped to [0, 1] on output — matching the real
NoisyLR data, which also exceeds that range after noise is added.
"""
import random

import cv2
import numpy as np


def add_speckle_noise(img, variance_range):
    """Multiplicative speckle noise: img + img * n, n ~ N(0, var)."""
    var = random.uniform(*variance_range)
    noise = np.random.normal(0.0, var ** 0.5, img.shape).astype(np.float32)
    return img + img * noise


def add_gaussian_noise(img, sigma_range):
    """Additive Gaussian noise with sigma sampled in normalized pixel units."""
    sigma = random.uniform(*sigma_range)
    noise = np.random.normal(0.0, sigma, img.shape).astype(np.float32)
    return img + noise


_CV2_INTERP = {
    "bicubic": cv2.INTER_CUBIC,
    "bilinear": cv2.INTER_LINEAR,
    "nearest": cv2.INTER_NEAREST,
}


def downsample(img, scale_factor, kernel="bicubic"):
    h, w = img.shape
    new_h, new_w = h // scale_factor, w // scale_factor
    interp = _CV2_INTERP[kernel]
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


class RandomDegradationPipeline:
    """Applies speckle noise, gaussian noise, and downsampling in a random
    order with randomly sampled severity, mirroring the randomized-order
    degradation shuffle used in BSRGAN / Real-ESRGAN to avoid the model
    overfitting to one fixed degradation recipe.
    """

    def __init__(self, speckle_variance_range, gaussian_sigma_range,
                 downscale_kernels, scale_factor):
        self.speckle_variance_range = tuple(speckle_variance_range)
        self.gaussian_sigma_range = tuple(gaussian_sigma_range)
        self.downscale_kernels = list(downscale_kernels)
        self.scale_factor = scale_factor

    def __call__(self, gt_patch):
        """gt_patch: float32 numpy array in [0, 1], shape (H, W).
        Returns a synthetic NoisyLR patch of shape (H // scale_factor, W // scale_factor).
        """
        ops = ["speckle", "gaussian", "downsample"]
        random.shuffle(ops)

        img = gt_patch.astype(np.float32).copy()
        downsampled = False
        for op in ops:
            if op == "speckle":
                img = add_speckle_noise(img, self.speckle_variance_range)
            elif op == "gaussian":
                img = add_gaussian_noise(img, self.gaussian_sigma_range)
            elif op == "downsample":
                kernel = random.choice(self.downscale_kernels)
                img = downsample(img, self.scale_factor, kernel)
                downsampled = True

        if not downsampled:  # safety net, ops list always includes it, kept for robustness
            img = downsample(img, self.scale_factor, random.choice(self.downscale_kernels))

        return img.astype(np.float32)

    @classmethod
    def from_config(cls, cfg):
        deg = cfg["degradation"]
        return cls(
            speckle_variance_range=deg["speckle_variance_range"],
            gaussian_sigma_range=deg["gaussian_sigma_range"],
            downscale_kernels=deg["downscale_kernels"],
            scale_factor=cfg["data"]["scale_factor"],
        )
