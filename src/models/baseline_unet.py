"""Lightweight U-Net baseline: bicubic-upsample NoisyLR to GT resolution, then
denoise/restore with a small U-Net. Deliberately simple — this exists purely
as a comparison baseline, not to compete with the final NAFNet-SR model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.ReLU(inplace=True),
    )


class BaselineUNet(nn.Module):
    """Small 3-level U-Net operating at GT resolution. Input is the NoisyLR
    image bicubic-upsampled to GT size; output is a residual added back to it.
    """

    def __init__(self, in_channels=1, base_channels=32, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        c = base_channels

        self.enc1 = conv_block(in_channels, c)
        self.enc2 = conv_block(c, c * 2)
        self.enc3 = conv_block(c * 2, c * 4)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = conv_block(c * 4, c * 8)

        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.dec3 = conv_block(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = conv_block(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = conv_block(c * 2, c)

        self.out_conv = nn.Conv2d(c, in_channels, 3, padding=1)

    def forward(self, x_lr):
        x = F.interpolate(x_lr, scale_factor=self.scale_factor, mode="bicubic", align_corners=False)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        residual = self.out_conv(d1)
        return x + residual


if __name__ == "__main__":
    model = BaselineUNet(in_channels=1, base_channels=32, scale_factor=2)
    dummy = torch.randn(2, 1, 128, 128)
    out = model(dummy)
    print(f"input {tuple(dummy.shape)} -> output {tuple(out.shape)}")
    assert out.shape == (2, 1, 256, 256)
