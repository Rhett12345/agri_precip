"""
model.py
轻量 UNet：7 通道 IR+BTD 输入 → 双头（分类 + 回归）
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """双层卷积块：Conv3x3 → BN → ReLU → Conv3x3 → BN → ReLU"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """
    轻量 UNet，双头输出。

    输入:  (B, 7, 128, 128)  — 4 IR + 3 BTD
    输出:  cls_logit (B, 1, 128, 128)
           reg_log  (B, 1, 128, 128)  — log1p 空间

    推理时:
      cls_prob = sigmoid(cls_logit)
      rate_mmh = expm1(reg_log)  if cls_prob > threshold
    """

    def __init__(self, in_ch: int = 7, base_ch: int = 32):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(in_ch, base_ch)           # 128x128
        self.enc2 = ConvBlock(base_ch, base_ch * 2)      # 64x64
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)  # 32x32
        self.enc4 = ConvBlock(base_ch * 4, base_ch * 8)  # 16x16

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(base_ch * 8, base_ch * 8)  # 8x8

        # Decoder (skip 维度: d4+e4=256+256, d3+e3=128+128, d2+e2=64+64, d1+e1=32+32)
        self.up4 = nn.ConvTranspose2d(base_ch * 8, base_ch * 8, 2, stride=2)
        self.dec4 = ConvBlock(base_ch * 16, base_ch * 4)

        self.up3 = nn.ConvTranspose2d(base_ch * 4, base_ch * 4, 2, stride=2)
        self.dec3 = ConvBlock(base_ch * 8, base_ch * 2)

        self.up2 = nn.ConvTranspose2d(base_ch * 2, base_ch * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_ch * 4, base_ch)

        self.up1 = nn.ConvTranspose2d(base_ch, base_ch, 2, stride=2)
        self.dec1 = ConvBlock(base_ch * 2, base_ch)

        # Dual heads
        self.cls_head = nn.Conv2d(base_ch, 1, kernel_size=1)
        self.reg_head = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, x: torch.Tensor):
        # Encoder
        e1 = self.enc1(x)                   # (B, 32, 128, 128)
        e2 = self.enc2(self.pool(e1))       # (B, 64, 64, 64)
        e3 = self.enc3(self.pool(e2))       # (B, 128, 32, 32)
        e4 = self.enc4(self.pool(e3))       # (B, 256, 16, 16)

        # Bottleneck
        b = self.bottleneck(self.pool(e4))  # (B, 256, 8, 8)

        # Decoder with skip connections
        d4 = self.up4(b)                              # (B, 256, 16, 16)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))    # (B, 128, 16, 16)

        d3 = self.up3(d4)                             # (B, 128, 32, 32)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))    # (B, 64, 32, 32)

        d2 = self.up2(d3)                             # (B, 64, 64, 64)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))    # (B, 32, 64, 64)

        d1 = self.up1(d2)                             # (B, 32, 128, 128)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))    # (B, 32, 128, 128)

        cls_logit = self.cls_head(d1)     # (B, 1, 128, 128)
        reg_log = self.reg_head(d1)       # (B, 1, 128, 128)

        return cls_logit, reg_log
