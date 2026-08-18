# AI-Based Restoration of Degraded Images for Semiconductor Inspection

Restores 128x128 speckle/Gaussian-noise-degraded, 2x-downsampled grayscale
inspection images back to clean 256x256 resolution, using NAFNet-SR: a
NAFNet-style (Chen et al., "Simple Baselines for Image Restoration",
arXiv:2204.04676) encoder/bottleneck/decoder followed by a sub-pixel
(PixelShuffle) upsample head.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

No internet access, API keys, or additional downloads are required at
runtime — the trained checkpoint is bundled under `models/`.

## Run

```bash
python run.py <input-dir> <output-dir>
```

- Reads every `.npy` file in `<input-dir>`.
- Creates `<output-dir>` if it does not already exist.
- Writes one restored `.npy` file per input, with the same filename.
- Each output is a grayscale float32 array of shape `(H, W)`, upsampled 2x
  from the input resolution (e.g. 128x128 input -> 256x256 output), with
  values clipped to `[0, 1]` and NaN/Inf sanitized before saving.
- Runs on an NVIDIA GPU automatically if available (`resolve_device` in
  `src/device.py`), falling back to CPU otherwise. No manual configuration
  or user interaction is needed.

## Contents

```
team_name/
├── run.py              # entry point: python run.py <input-dir> <output-dir>
├── requirements.txt
├── README.md
├── models/
│   └── nafnet_sr_best.pth  # trained model checkpoint (place here after training)
└── src/
    ├── device.py            # GPU/CPU device resolution
    └── models/
        ├── nafnet_sr.py     # NAFNet-SR architecture
        └── nafnet_blocks.py # NAFBlock building blocks
```

## Note

`models/nafnet_sr_best.pth` must be produced by running `train.py` from the
main repository (see top-level README) and copied into this folder before
final submission.
