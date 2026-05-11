from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def label_to_onehot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(labels, num_classes=num_classes).float()


def make_label_map(
    labels: torch.Tensor,
    num_classes: int,
    height: int,
    width: int,
) -> torch.Tensor:
    onehot = label_to_onehot(labels, num_classes)
    return onehot[:, :, None, None].repeat(1, 1, height, width)


class VanillaCVAE(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        channels: int = 1,
        num_classes: int = 3,
        latent_dim: int = 128,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.channels = channels
        self.num_classes = num_classes
        self.latent_dim = latent_dim

        encoder_input_channels = channels + num_classes
        self.encoder = nn.Sequential(
            nn.Conv2d(encoder_input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.feature_size = img_size // 16
        self.flatten_dim = 256 * self.feature_size * self.feature_size
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
        self.decoder_input = nn.Linear(latent_dim + num_classes, self.flatten_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor, labels: torch.Tensor):
        batch_size, _, height, width = x.shape
        label_map = make_label_map(labels, self.num_classes, height, width).to(x.device)
        features = self.encoder(torch.cat([x, label_map], dim=1))
        features = features.view(batch_size, -1)
        return self.fc_mu(features), self.fc_logvar(features)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        onehot = label_to_onehot(labels, self.num_classes).to(z.device)
        x = self.decoder_input(torch.cat([z, onehot], dim=1))
        x = x.view(-1, 256, self.feature_size, self.feature_size)
        return self.decoder(x)

    def forward(self, x: torch.Tensor, labels: torch.Tensor):
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, labels), mu, logvar


def main() -> None:
    img_size = 224
    channels = 1
    num_classes = 3
    latent_dim = 128
    beta = 1.0

    model = VanillaCVAE(
        img_size=img_size,
        channels=channels,
        num_classes=num_classes,
        latent_dim=latent_dim,
    )

    x = torch.rand(2, channels, img_size, img_size)
    labels = torch.tensor([0, num_classes - 1], dtype=torch.long)
    recon, mu, logvar = model(x, labels)
    recon_loss = F.binary_cross_entropy(recon, x, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss = recon_loss + beta * kl_loss
    loss.backward()

    assert recon.shape == x.shape
    assert mu.shape == (2, latent_dim)
    assert logvar.shape == (2, latent_dim)
    print("Vanilla CVAE smoke test passed.")
    print(f"Image size: {img_size} | Classes: {num_classes} | Latent dim: {latent_dim}")
    print(f"Loss: {loss.item():.4f} | Recon: {recon_loss.item():.4f} | KL: {kl_loss.item():.4f}")


if __name__ == "__main__":
    main()
