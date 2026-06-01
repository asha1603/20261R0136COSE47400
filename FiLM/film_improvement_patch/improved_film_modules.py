"""
improved_film_modules.py

Focused improvement for a FiLM-CVAE / U-Net CVAE model.
Use this file to replace or adapt the model part of your current FiLM code.

Main change:
- Original U-Net skip connections can make the model copy the input too strongly.
- This version uses target-conditioned gated skips + FiLM in the decoder.
- During translation, you can pass skip_scale < 1.0 to reduce copying.

Expected input:
    x: Tensor [B, 1, 224, 224], values in [0, 1]
    target_y: Tensor [B], class index: 0, 1, 2

Forward output:
    out, mu, logvar, z = model(x, target_y, skip_scale=...)
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvGNAct(nn.Module):
    """Conv -> GroupNorm -> SiLU. GroupNorm is stable for small Colab batches."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        groups = min(8, out_ch)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """Downsample by 2, then refine."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvGNAct(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            ConvGNAct(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class RefineBlock(nn.Module):
    """Two conv refinement block."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvGNAct(in_ch, out_ch),
            ConvGNAct(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation.

    h' = h * (1 + gamma(label)) + beta(label)

    Zero initialization makes the layer initially behave like identity, which makes
    training more stable when adding FiLM to an existing CVAE.
    """

    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.to_gamma_beta = nn.Linear(cond_dim, channels * 2)
        nn.init.zeros_(self.to_gamma_beta.weight)
        nn.init.zeros_(self.to_gamma_beta.bias)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.to_gamma_beta(cond).chunk(2, dim=1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return h * (1.0 + gamma) + beta


class TargetSkipGate(nn.Module):
    """
    Target-conditioned skip gate.

    Why this helps:
    - Plain U-Net skips pass detailed source anatomy directly into the decoder.
    - That improves reconstruction, but makes target-stage outputs look identical.
    - This gate lets you reduce skip strength for translation samples.
    """

    def __init__(self, cond_dim: int, skip_channels: int, max_gate: float = 0.70):
        super().__init__()
        self.max_gate = max_gate
        self.to_gate = nn.Linear(cond_dim, skip_channels)
        nn.init.zeros_(self.to_gate.weight)
        nn.init.constant_(self.to_gate.bias, -0.5)

    def forward(self, skip: torch.Tensor, cond: torch.Tensor, skip_scale: float = 1.0) -> torch.Tensor:
        gate = torch.sigmoid(self.to_gate(cond))[:, :, None, None]
        return skip * gate * self.max_gate * skip_scale


class ImprovedFiLMUNetCVAE(nn.Module):
    """
    Improved FiLM + gated-skip CVAE.

    This is intentionally close to your current FiLM-CVAE idea, but with one important
    architectural change: skip connections are gated by the target condition.
    """

    def __init__(
        self,
        img_size: int = 224,
        in_channels: int = 1,
        num_classes: int = 3,
        latent_dim: int = 128,
        class_dim: int = 32,
        base_channels: int = 32,
    ):
        super().__init__()
        if img_size != 224:
            raise ValueError("This model template assumes 224x224 images. Resize input to 224x224.")
        if latent_dim <= class_dim:
            raise ValueError("latent_dim must be larger than class_dim.")

        self.img_size = img_size
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.class_dim = class_dim
        self.content_dim = latent_dim - class_dim

        c = base_channels
        self.label_emb = nn.Embedding(num_classes, class_dim)

        # Encoder: 224 -> 112 -> 56 -> 28 -> 14
        self.enc1 = DownBlock(in_channels, c)       # [B, 32, 112, 112]
        self.enc2 = DownBlock(c, c * 2)             # [B, 64, 56, 56]
        self.enc3 = DownBlock(c * 2, c * 4)         # [B, 128, 28, 28]
        self.enc4 = DownBlock(c * 4, c * 8)         # [B, 256, 14, 14]

        self.flat_dim = c * 8 * 14 * 14
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)

        # Decoder starts from content latent + target class embedding.
        # This forces target label to affect generation more directly.
        self.fc_dec = nn.Linear(self.content_dim + class_dim, self.flat_dim)

        self.gate3 = TargetSkipGate(class_dim, c * 4, max_gate=0.65)
        self.gate2 = TargetSkipGate(class_dim, c * 2, max_gate=0.55)
        self.gate1 = TargetSkipGate(class_dim, c, max_gate=0.45)

        self.dec3 = RefineBlock(c * 8 + c * 4, c * 4)
        self.dec2 = RefineBlock(c * 4 + c * 2, c * 2)
        self.dec1 = RefineBlock(c * 2 + c, c)
        self.dec0 = RefineBlock(c, c)

        self.film3 = FiLM(class_dim, c * 4)
        self.film2 = FiLM(class_dim, c * 2)
        self.film1 = FiLM(class_dim, c)
        self.film0 = FiLM(class_dim, c)

        self.out = nn.Sequential(
            nn.Conv2d(c, c // 2, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c // 2, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        h = torch.flatten(e4, start_dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h).clamp(min=-8.0, max=8.0)
        return mu, logvar, (e1, e2, e3, e4)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, target_y: torch.Tensor, skips, skip_scale: float = 1.0) -> torch.Tensor:
        e1, e2, e3, _ = skips
        y_emb = self.label_emb(target_y)

        # Replace class part of z with target label embedding.
        z_content = z[:, : self.content_dim]
        z_cond = torch.cat([z_content, y_emb], dim=1)

        d = self.fc_dec(z_cond).view(z.shape[0], -1, 14, 14)

        d = F.interpolate(d, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        s3 = self.gate3(e3, y_emb, skip_scale=skip_scale)
        d = self.dec3(torch.cat([d, s3], dim=1))
        d = self.film3(d, y_emb)

        d = F.interpolate(d, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        s2 = self.gate2(e2, y_emb, skip_scale=skip_scale)
        d = self.dec2(torch.cat([d, s2], dim=1))
        d = self.film2(d, y_emb)

        d = F.interpolate(d, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        s1 = self.gate1(e1, y_emb, skip_scale=skip_scale)
        d = self.dec1(torch.cat([d, s1], dim=1))
        d = self.film1(d, y_emb)

        d = F.interpolate(d, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        d = self.dec0(d)
        d = self.film0(d, y_emb)
        return self.out(d)

    def forward(self, x: torch.Tensor, target_y: torch.Tensor, skip_scale: float = 1.0):
        mu, logvar, skips = self.encode(x)
        z = self.reparameterize(mu, logvar)
        out = self.decode(z, target_y, skips, skip_scale=skip_scale)
        return out, mu, logvar, z


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
