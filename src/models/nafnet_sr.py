"""NAFNet-SR: a NAFNet-style UNet encoder/bottleneck/decoder operating at
NoisyLR resolution, followed by a sub-pixel (PixelShuffle) upsample head that
brings the output to GT resolution.
"""
import torch
import torch.nn as nn

from src.models.nafnet_blocks import NAFBlock


class NAFNetSR(nn.Module):
    def __init__(self, in_channels=1, width=32, enc_blocks=(2, 2, 4, 8),
                 middle_blocks=12, dec_blocks=(2, 2, 2, 2), scale_factor=2):
        super().__init__()
        assert len(enc_blocks) == len(dec_blocks)
        self.scale_factor = scale_factor

        self.intro = nn.Conv2d(in_channels, width, 3, padding=1)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        chan = width
        for num in enc_blocks:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, chan * 2, 2, stride=2))
            chan *= 2

        self.middle = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blocks)])

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for num in dec_blocks:
            self.ups.append(nn.Sequential(
                nn.Conv2d(chan, chan * 2, 1, bias=False),
                nn.PixelShuffle(2),
            ))
            chan //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        # sub-pixel upsample head: LR-resolution features -> GT resolution
        self.sr_head = nn.Sequential(
            nn.Conv2d(chan, in_channels * (scale_factor ** 2), 3, padding=1),
            nn.PixelShuffle(scale_factor),
        )

        self.padder_size = 2 ** len(enc_blocks)

    def _pad_to_multiple(self, x):
        _, _, h, w = x.shape
        pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        if pad_h or pad_w:
            x = nn.functional.pad(x, (0, pad_w, 0, pad_h))
        return x, h, w

    def forward(self, inp):
        x, orig_h, orig_w = self._pad_to_multiple(inp)

        x = self.intro(x)
        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            skips.append(x)
            x = down(x)

        x = self.middle(x)

        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up(x)
            x = x + skip
            x = decoder(x)

        out = self.sr_head(x)
        out = out[:, :, :orig_h * self.scale_factor, :orig_w * self.scale_factor]
        return out


if __name__ == "__main__":
    model = NAFNetSR(in_channels=1, width=32, enc_blocks=[2, 2, 4, 8],
                      middle_blocks=12, dec_blocks=[2, 2, 2, 2], scale_factor=2)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"NAFNetSR params: {n_params / 1e6:.2f}M")

    dummy = torch.randn(1, 1, 128, 128)
    out = model(dummy)
    print(f"input {tuple(dummy.shape)} -> output {tuple(out.shape)}")
    assert out.shape == (1, 1, 256, 256)

    # non-multiple-of-16 input still produces exactly 2x output
    dummy2 = torch.randn(1, 1, 100, 100)
    out2 = model(dummy2)
    print(f"input {tuple(dummy2.shape)} -> output {tuple(out2.shape)}")
    assert out2.shape == (1, 1, 200, 200)
