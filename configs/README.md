# Config fields (`configs/base.yaml`)

Single source of truth for every script in this repo — no script hardcodes a
hyperparameter that's already here.

- `seed` — global random seed (Python/NumPy/torch), logged at the start of every run.
- `data.gt_dir` / `data.noisy_lr_dir` — paired training directories (`train/GT`, `train/NoisyLR`).
- `data.test_noisy_lr_dir` — hidden-test NoisyLR-only directory (`NoisyLR/`, no GT provided).
- `data.in_channels` — 1 (grayscale imagery).
- `data.gt_size` / `data.scale_factor` — GT images are 256px, NoisyLR is 128px → 2x scale factor.
- `data.val_split` / `data.val_split_seed` — fraction of image IDs (not crops) held out for validation, and the seed controlling which IDs.
- `data.patch_size` — GT-space random-crop size used during training. Must be a multiple of `scale_factor`, and >96px to satisfy `MSSSIMLoss`'s minimum window size.
- `data.synthetic_ratio` — fraction of training steps that draw a synthetically-degraded batch (`RandomDegradationPipeline`) instead of a real pair.
- `data.num_workers` — DataLoader worker count.
- `degradation.*` — ranges for the synthetic speckle/gaussian/downsample degradation pipeline (Phase 2).
- `train.*` — batch size, epoch count, LR schedule, AMP, checkpoint cadence, and `device` (honored with cuda → mps → cpu fallback, see `src/device.py`).
- `model.*` — NAFNet-SR width, per-stage block counts, and bottleneck depth.
- `loss.*` — Charbonnier / MS-SSIM / LPIPS weights for `CombinedRestorationLoss`.
- `baseline.*` — separate, smaller hyperparameters for the comparison U-Net baseline (Phase 3).
- `paths.*` — output directories for checkpoints, results, and TensorBoard logs.
