from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


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


class ResNet18MRIClassifier(nn.Module):
    """ResNet18 grayscale guide required by the final methodology."""

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.model = models.resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation:
        FiLM(h, e_c) = h * (1 + gamma(e_c)) + beta(e_c)
    """

    def __init__(self, condition_dim: int, channels: int) -> None:
        super().__init__()
        self.affine = nn.Linear(condition_dim, channels * 2)
        # Identity modulation at initialization stabilizes training.
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.affine(condition).chunk(2, dim=1)
        return features * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]


class FinalFiLMDisentangledCVAE(nn.Module):
    """
    Final classifier-guided disentangled CVAE.

    - Encoder sees x only; it does not receive class labels.
    - z_content preserves source anatomy (96 dimensions).
    - z_class is organized by center/separation loss (32 dimensions).
    - Following the final PPT decoder diagram, only z_content is supplied to
      the decoder together with target class embedding e_c.
    - Exactly two skip connections are used: 28x28 and 56x56.
    - FiLM applies the target condition at every decoder resolution.
    """

    def __init__(
        self,
        image_size: int = 224,
        latent_dim: int = 128,
        content_dim: int = 96,
        class_dim: int = 32,
        num_classes: int = 3,
        class_embed_dim: int = 32,
    ) -> None:
        super().__init__()
        if image_size != 224:
            raise ValueError("Final architecture is configured for 224x224 MRI slices.")
        if content_dim + class_dim != latent_dim:
            raise ValueError("content_dim + class_dim must equal latent_dim.")

        self.image_size = image_size
        self.latent_dim = latent_dim
        self.content_dim = content_dim
        self.class_dim = class_dim
        self.num_classes = num_classes
        self.class_embed_dim = class_embed_dim
        self.bottleneck_size = image_size // 16   # 14 for 224x224
        self.flat_dim = 256 * self.bottleneck_size * self.bottleneck_size

        # Encoder q_phi(z|x): no disease label is injected here.
        self.enc1 = conv_down(1, 32)      # 112 x 112
        self.enc2 = conv_down(32, 64)     # 56 x 56: skip 2
        self.enc3 = conv_down(64, 128)    # 28 x 28: skip 1
        self.enc4 = conv_down(128, 256)   # 14 x 14
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)

        self.class_embedding = nn.Embedding(num_classes, class_embed_dim)
        self.dec_fc = nn.Linear(content_dim + class_embed_dim, self.flat_dim)

        # Two U-Net skips only, as selected in the final slide.
        self.dec28 = conv_block(256 + 128, 128)
        self.dec56 = conv_block(128 + 64, 64)
        self.dec112 = conv_block(64, 32)
        self.dec224 = conv_block(32, 16)

        # Condition target disease stage at every decoder resolution.
        self.film28 = FiLM(class_embed_dim, 128)
        self.film56 = FiLM(class_embed_dim, 64)
        self.film112 = FiLM(class_embed_dim, 32)
        self.film224 = FiLM(class_embed_dim, 16)

        self.out = nn.Sequential(
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
        return mu, logvar, (e2, e3)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def split_latent(self, z: torch.Tensor):
        return z[:, : self.content_dim], z[:, self.content_dim :]

    def decode(self, z_content: torch.Tensor, labels: torch.Tensor, skips) -> torch.Tensor:
        e2, e3 = skips
        class_emb = self.class_embedding(labels)
        conditioned = torch.cat([z_content, class_emb], dim=1)
        d = self.dec_fc(conditioned).view(
            z_content.size(0), 256, self.bottleneck_size, self.bottleneck_size
        )

        d = F.interpolate(d, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d = self.film28(self.dec28(torch.cat([d, e3], dim=1)), class_emb)

        d = F.interpolate(d, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d = self.film56(self.dec56(torch.cat([d, e2], dim=1)), class_emb)

        d = F.interpolate(d, size=(112, 112), mode="bilinear", align_corners=False)
        d = self.film112(self.dec112(d), class_emb)

        d = F.interpolate(d, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        d = self.film224(self.dec224(d), class_emb)
        return self.out(d)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
        mu, logvar, skips = self.encode(x)
        z = self.reparameterize(mu, logvar)
        z_content, z_class = self.split_latent(z)
        recon = self.decode(z_content, labels, skips)
        return {
            "recon": recon,
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "z_content": z_content,
            "z_class": z_class,
        }

    @torch.no_grad()
    def translate(
        self,
        x: torch.Tensor,
        target_labels: torch.Tensor,
        deterministic: bool = True,
    ) -> torch.Tensor:
        """Generate a requested target stage while retaining source anatomy."""
        mu, logvar, skips = self.encode(x)
        z = mu if deterministic else self.reparameterize(mu, logvar)
        z_content, _ = self.split_latent(z)
        return self.decode(z_content, target_labels, skips)
