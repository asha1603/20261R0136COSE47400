from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_down(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class SmallMRIClassifier(nn.Module):
    """Lightweight grayscale classifier used as the frozen guide."""

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            conv_down(1, 32),
            conv_down(32, 64),
            conv_down(64, 128),
            conv_down(128, 256),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class MidUNetCVAE(nn.Module):
    """Classifier-guided mid-version U-Net CVAE.

    The class condition is injected once at the bottleneck. This intentionally
    avoids FiLM, latent splitting, and latent clustering used by the final model.
    """

    def __init__(
        self,
        image_size: int = 224,
        latent_dim: int = 128,
        num_classes: int = 3,
        class_embed_dim: int = 32,
    ) -> None:
        super().__init__()
        if image_size % 16 != 0:
            raise ValueError("image_size must be divisible by 16")

        self.image_size = image_size
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.class_embed_dim = class_embed_dim
        self.bottleneck_size = image_size // 16
        self.flat_dim = 256 * self.bottleneck_size * self.bottleneck_size

        self.enc1 = conv_down(1, 32)
        self.enc2 = conv_down(32, 64)
        self.enc3 = conv_down(64, 128)
        self.enc4 = conv_down(128, 256)

        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)
        self.class_embedding = nn.Embedding(num_classes, class_embed_dim)
        self.dec_fc = nn.Linear(latent_dim + class_embed_dim, self.flat_dim)

        self.dec1 = conv_block(256 + 128, 128)
        self.dec2 = conv_block(128 + 64, 64)
        self.dec3 = conv_block(64 + 32, 32)
        self.out = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        h = e4.flatten(start_dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar, (e1, e2, e3)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, labels: torch.Tensor, skips) -> torch.Tensor:
        e1, e2, e3 = skips
        class_emb = self.class_embedding(labels)
        conditioned = torch.cat([z, class_emb], dim=1)
        d = self.dec_fc(conditioned).view(
            z.size(0), 256, self.bottleneck_size, self.bottleneck_size
        )

        d = F.interpolate(d, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d = self.dec1(torch.cat([d, e3], dim=1))
        d = F.interpolate(d, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d = self.dec2(torch.cat([d, e2], dim=1))
        d = F.interpolate(d, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d = self.dec3(torch.cat([d, e1], dim=1))
        d = F.interpolate(
            d,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        return self.out(d)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
        mu, logvar, skips = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, labels, skips)
        return {"recon": recon, "mu": mu, "logvar": logvar, "z": z}
