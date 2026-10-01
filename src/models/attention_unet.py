"""
attention_unet.py
====================
Attention U-Net (Oktay et al. 2018, "Attention U-Net: Learning Where to
Look for the Pancreas") implemented from scratch -- segmentation-models-
pytorch does not provide this specific architecture, and it's simple enough
(standard U-Net encoder/decoder + attention gates on skip connections) that
a from-scratch implementation is lower-risk than forcing a workaround out
of a library that doesn't have it.

Trained from scratch (no pretrained weights) -- this is standard practice
for Attention U-Net in the literature and needs no network access.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)


class AttentionGate(nn.Module):
    """g: gating signal from the coarser decoder level. x: skip connection
    from the encoder at the matching resolution. Returns x reweighted by a
    learned attention coefficient, so the decoder can suppress irrelevant
    background regions in the skip connection -- particularly relevant here
    since TiN occupies a tiny fraction of each patch, and most of the skip
    connection's spatial content is majority-phase background."""
    def __init__(self, gate_ch, skip_ch, inter_ch):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(gate_ch, inter_ch, 1), nn.BatchNorm2d(inter_ch))
        self.W_x = nn.Sequential(nn.Conv2d(skip_ch, inter_ch, 1), nn.BatchNorm2d(inter_ch))
        self.psi = nn.Sequential(nn.Conv2d(inter_ch, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttentionUNet(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 5, base_ch: int = 32):
        super().__init__()
        chs = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 16]

        self.pool = nn.MaxPool2d(2)
        self.enc1 = ConvBlock(in_channels, chs[0])
        self.enc2 = ConvBlock(chs[0], chs[1])
        self.enc3 = ConvBlock(chs[1], chs[2])
        self.enc4 = ConvBlock(chs[2], chs[3])
        self.bottleneck = ConvBlock(chs[3], chs[4])

        self.up4 = UpConv(chs[4], chs[3])
        self.att4 = AttentionGate(chs[3], chs[3], chs[3] // 2)
        self.dec4 = ConvBlock(chs[4], chs[3])

        self.up3 = UpConv(chs[3], chs[2])
        self.att3 = AttentionGate(chs[2], chs[2], chs[2] // 2)
        self.dec3 = ConvBlock(chs[3], chs[2])

        self.up2 = UpConv(chs[2], chs[1])
        self.att2 = AttentionGate(chs[1], chs[1], chs[1] // 2)
        self.dec2 = ConvBlock(chs[2], chs[1])

        self.up1 = UpConv(chs[1], chs[0])
        self.att1 = AttentionGate(chs[0], chs[0], chs[0] // 2)
        self.dec1 = ConvBlock(chs[1], chs[0])

        self.final = nn.Conv2d(chs[0], num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        e4_att = self.att4(d4, e4)
        d4 = self.dec4(torch.cat([e4_att, d4], dim=1))

        d3 = self.up3(d4)
        e3_att = self.att3(d3, e3)
        d3 = self.dec3(torch.cat([e3_att, d3], dim=1))

        d2 = self.up2(d3)
        e2_att = self.att2(d2, e2)
        d2 = self.dec2(torch.cat([e2_att, d2], dim=1))

        d1 = self.up1(d2)
        e1_att = self.att1(d1, e1)
        d1 = self.dec1(torch.cat([e1_att, d1], dim=1))

        return self.final(d1)


def build_attention_unet(num_classes: int, in_channels: int = 1, base_ch: int = 32) -> nn.Module:
    return AttentionUNet(in_channels=in_channels, num_classes=num_classes, base_ch=base_ch)


def predict_probs(model: nn.Module, image_patch, device: str = "cpu"):
    """Same predict_fn adapter interface as deeplabv3plus.predict_probs."""
    import numpy as np
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(image_patch.astype("float32") / 255.0)
        x = x.unsqueeze(0).unsqueeze(0).to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        return probs.squeeze(0).cpu().numpy()
