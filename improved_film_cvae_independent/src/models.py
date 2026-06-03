from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import timm
except ImportError:
    timm = None

NUM_CLASSES = 3


def group_norm(channels: int) -> nn.Module:
    groups = min(8, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            group_norm(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            group_norm(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.down = nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.conv = ConvBlock(out_ch, out_ch)

    def forward(self, x):
        return self.conv(self.down(x))


class UpBlock(nn.Module):
    """Bilinear upsampling + convolution. This avoids ConvTranspose2d checkerboard artifacts."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x, target_size: Tuple[int, int]):
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return self.conv(x)


class GentleFiLM(nn.Module):
    """Small FiLM modulation to prevent global distortion artifacts."""
    def __init__(self, embed_dim: int, channels: int, scale: float = 0.05):
        super().__init__()
        self.scale = scale
        self.fc = nn.Linear(embed_dim, channels * 2)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x, emb):
        gamma, beta = self.fc(emb).chunk(2, dim=1)
        gamma = gamma[..., None, None]
        beta = beta[..., None, None]
        return x * (1.0 + self.scale * gamma) + self.scale * beta


class TargetSkipGate(nn.Module):
    """Target-conditioned skip gate so the decoder cannot copy raw encoder features too strongly."""
    def __init__(self, embed_dim: int, channels: int, max_scale: float):
        super().__init__()
        self.max_scale = max_scale
        self.fc = nn.Linear(embed_dim, channels)
        nn.init.zeros_(self.fc.weight)
        nn.init.constant_(self.fc.bias, -0.5)

    def forward(self, skip, emb, global_scale: float = 1.0):
        gate = torch.sigmoid(self.fc(emb))[..., None, None]
        return skip * gate * self.max_scale * global_scale


class ImprovedFiLMGatedCVAE(nn.Module):
    """
    Improved FiLM-Gated U-Net CVAE.

    Main anti-artifact design:
    - No transposed convolution.
    - Gentle FiLM scale.
    - Gated skip connections.
    - Translation can use smaller skip_scale than reconstruction.
    """
    def __init__(
        self,
        image_size: int = 224,
        in_channels: int = 1,
        num_classes: int = NUM_CLASSES,
        latent_dim: int = 128,
        class_dim: int = 32,
        base_channels: int = 32,
        film_scale: float = 0.05,
    ):
        super().__init__()
        if image_size != 224:
            raise ValueError("This implementation assumes image_size=224.")
        if latent_dim <= class_dim:
            raise ValueError("latent_dim must be larger than class_dim.")

        self.image_size = image_size
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.class_dim = class_dim
        self.content_dim = latent_dim - class_dim

        c = base_channels
        self.class_embed = nn.Embedding(num_classes, class_dim)

        self.enc1 = DownBlock(in_channels, c)       # 112
        self.enc2 = DownBlock(c, c * 2)             # 56
        self.enc3 = DownBlock(c * 2, c * 4)         # 28
        self.enc4 = DownBlock(c * 4, c * 8)         # 14

        self.flat_dim = c * 8 * 14 * 14
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)

        self.dec_fc = nn.Linear(latent_dim, self.flat_dim)

        self.gate3 = TargetSkipGate(class_dim, c * 4, max_scale=0.75)
        self.gate2 = TargetSkipGate(class_dim, c * 2, max_scale=0.65)
        self.gate1 = TargetSkipGate(class_dim, c, max_scale=0.55)

        self.up3 = UpBlock(c * 8 + c * 4, c * 4)
        self.up2 = UpBlock(c * 4 + c * 2, c * 2)
        self.up1 = UpBlock(c * 2 + c, c)
        self.up0 = UpBlock(c, c)

        self.film3 = GentleFiLM(class_dim, c * 4, film_scale)
        self.film2 = GentleFiLM(class_dim, c * 2, film_scale)
        self.film1 = GentleFiLM(class_dim, c, film_scale)
        self.film0 = GentleFiLM(class_dim, c, film_scale)

        self.out = nn.Sequential(
            nn.Conv2d(c, c // 2, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c // 2, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        h = e4.flatten(start_dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h).clamp(-8.0, 8.0)
        return mu, logvar, (e1, e2, e3, e4)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, target_label, skips, skip_scale: float = 1.0):
        e1, e2, e3, _ = skips
        emb = self.class_embed(target_label)

        z_content = z[:, : self.content_dim]
        z_cond = torch.cat([z_content, emb], dim=1)

        d = self.dec_fc(z_cond).view(z.size(0), -1, 14, 14)

        s3 = self.gate3(e3, emb, global_scale=skip_scale)
        d = self.up3(torch.cat([F.interpolate(d, size=e3.shape[-2:], mode="bilinear", align_corners=False), s3], dim=1), e3.shape[-2:])
        d = self.film3(d, emb)

        s2 = self.gate2(e2, emb, global_scale=skip_scale)
        d = self.up2(torch.cat([F.interpolate(d, size=e2.shape[-2:], mode="bilinear", align_corners=False), s2], dim=1), e2.shape[-2:])
        d = self.film2(d, emb)

        s1 = self.gate1(e1, emb, global_scale=skip_scale)
        d = self.up1(torch.cat([F.interpolate(d, size=e1.shape[-2:], mode="bilinear", align_corners=False), s1], dim=1), e1.shape[-2:])
        d = self.film1(d, emb)

        d = self.up0(d, (self.image_size, self.image_size))
        d = self.film0(d, emb)
        return self.out(d)

    def forward(self, x, target_label, skip_scale: float = 1.0):
        mu, logvar, skips = self.encode(x)
        z = self.reparameterize(mu, logvar)
        out = self.decode(z, target_label, skips, skip_scale=skip_scale)
        return out, mu, logvar, z


class MRIClassifier(nn.Module):
    def __init__(self, model_name: str = "resnet18", num_classes: int = NUM_CLASSES, pretrained: bool = False):
        super().__init__()
        if timm is None:
            raise ImportError("timm is required for MRIClassifier. Install with: pip install timm")
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=1,
        )

    def forward(self, x):
        return self.model(x)
