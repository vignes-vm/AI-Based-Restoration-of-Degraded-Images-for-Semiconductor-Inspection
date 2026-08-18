# Results

Validation set: 320 held-out image IDs (10% split, seed 42), never seen during
training of either model. Metrics computed by `src/eval/validate.py` against
`weights/baseline_best.pth` (epoch 55/60) and `weights/nafnet_sr_best.pth`
(epoch 176/200), selected by best validation PSNR+SSIM during training.

## Metrics comparison

| Model | Params | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|---|
| U-Net baseline (bicubic upsample + residual denoise) | 1.9M | 28.35 ± 4.87 | 0.7675 ± 0.1592 | 0.2749 ± 0.1635 |
| **NAFNet-SR** (main model) | 29.2M | 28.34 ± 4.92 | 0.7700 ± 0.1499 | **0.1383 ± 0.0704** |

PSNR and SSIM land within noise of each other between the two models — both
recover pixel-level fidelity about equally well on this degradation. The gap
that matters is **LPIPS: 0.138 vs 0.275, essentially half**. NAFNet-SR is
trained with an LPIPS term in the loss (`loss.lpips_weight` in
`configs/base.yaml`) and produces outputs that are perceptually sharper and
better match the local texture statistics of the GT, even where per-pixel
error is similar. The qualitative grids
(`results/samples/val_examples_baseline.png` vs `val_examples_nafnet_sr.png`)
show this directly: the baseline output looks visibly smoothed/denoised in a
generic way, while NAFNet-SR reconstructs finer inspection-relevant detail.

Given the target use case (visual inspection of semiconductor imagery, where
perceptual detail matters more than raw dB), NAFNet-SR is the recommended
model and the one packaged in `submission/team_name/`.

## Throughput (NAFNet-SR, `weights/nafnet_sr_best.pth`)

Measured on `NVIDIA GeForce RTX 3050 Laptop GPU` (`src/eval/benchmark_throughput.py`,
64 images from the hidden `NoisyLR/` set, full pipeline: disk read → preprocess
→ GPU transfer → forward → transfer back → postprocess).

| batch_size | precision | images/sec | ms/image | peak mem (MB) |
|---|---|---|---|---|
| 1 | FP32 | 11.88 | 84.2 | 135.7 |
| 1 | FP16 | 9.51 | 105.2 | 67.9 |
| 4 | FP32 | 67.48 | 14.8 | 193.0 |
| 4 | FP16 | 18.08 | 55.3 | 96.5 |
| 8 | FP32 | 78.02 | 12.8 | 269.2 |
| 8 | FP16 | 17.81 | 56.2 | 134.7 |
| 16 | FP32 | 84.51 | 11.8 | 421.8 |
| 16 | FP16 | 17.53 | 57.0 | 210.9 |

**FP16 is consistently slower than FP32 on this GPU**, not faster. The RTX
3050's Tensor Cores need larger matmul/conv tile sizes than this 1-channel,
128×128-input model provides to amortize the FP16↔FP32 cast overhead — at
this shape the cast cost dominates and outweighs any throughput gain. FP32,
batch size 16 is the fastest configuration measured (84.5 img/s, 11.8 ms/img)
and is what should be used at inference time on similar hardware; larger/
higher-end GPUs (A100/V100-class, per the original CUDA target) would be
expected to show the more typical FP16 speedup, but that wasn't observed on
the hardware available for this evaluation.

## Qualitative / failure-case analysis

Best- and worst-case examples for NAFNet-SR on the validation set
(`results/samples/val_examples_nafnet_sr.png` has best/worst/3-random panels):

**Best cases** (PSNR 39–41 dB, SSIM > 0.95): images with lower structural
complexity and less severe sampled degradation — the model reconstructs GT
detail nearly exactly.

**Worst cases** (PSNR 11–16 dB, SSIM 0.28–0.29): a small cluster of validation
IDs (e.g. `002637`, `002973`, `000407`) score far below the mean despite
LPIPS staying in a normal range (0.16–0.17) — i.e. the output is perceptually
plausible-looking but structurally misaligned with GT. This pattern (low
PSNR/SSIM, normal LPIPS) is consistent with either (a) unusually severe
sampled degradation in the paired NoisyLR for that ID pushing it outside the
degradation distribution the model was trained on, or (b) real (non-synthetic)
pairs where the true degradation doesn't match the assumed speckle/Gaussian/
downsample model documented in the README's Assumptions section. This is the
main remaining failure mode: a long tail of hard examples where the model
produces a clean-looking but not-quite-correct reconstruction rather than
failing obviously.

## Training configuration

Both models trained per `configs/base.yaml` (see `configs/run_20260818_185929.yaml`
for the exact resolved snapshot of this run): seed 42, 10% val split (seed 42,
by image ID), 30% synthetic-degradation-augmented batches, AMP enabled.

- **Baseline U-Net**: 60 epochs, batch size 16, lr 1e-3, best checkpoint at epoch 55.
- **NAFNet-SR**: 200 epochs, batch size 16, lr 2e-4 (5-epoch warmup, cosine decay
  to 1e-6), Charbonnier + MS-SSIM + LPIPS combined loss, best checkpoint at
  epoch 176.

Both were trained on an `NVIDIA GeForce RTX 3050 Laptop GPU`.
