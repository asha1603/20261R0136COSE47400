from __future__ import annotations

import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def kl_annealing_beta(epoch: int, beta_max: float, warmup_epochs: int) -> float:
    """Paper equation beta_t = min(beta_max, beta_max * t/T). Epoch starts at 1."""
    if warmup_epochs <= 0:
        return beta_max
    return min(beta_max, beta_max * float(epoch) / float(warmup_epochs))


def normalize_for_classifier(
    images: torch.Tensor,
    mean: float = 0.456,
    std: float = 0.224,
) -> torch.Tensor:
    return (images - mean) / std


class LPIPSLoss(nn.Module):
    """LPIPS feature loss for grayscale MRIs; repeats one channel into RGB."""

    def __init__(self, net: str = "alex") -> None:
        super().__init__()
        try:
            import lpips
        except ImportError as exc:
            raise ImportError("Install the final dependency first: pip install lpips") from exc
        self.metric = lpips.LPIPS(net=net)
        self.metric.eval()
        for parameter in self.metric.parameters():
            parameter.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        self.metric.eval()
        return self

    def forward(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        recon_rgb = recon.repeat(1, 3, 1, 1) * 2.0 - 1.0
        target_rgb = target.repeat(1, 3, 1, 1) * 2.0 - 1.0
        return self.metric(recon_rgb, target_rgb).mean()


class FinalFiLMLoss(nn.Module):
    """
    Final loss from the report/PPT:
      L = L_recon + beta*L_KL + w_center*L_center + w_sep*L_sep + lambda_cls*L_cls
      L_recon = lambda_bce*BCE + lambda_lpips*LPIPS
    """

    def __init__(
        self,
        classifier: nn.Module,
        num_classes: int = 3,
        class_dim: int = 32,
        lambda_bce: float = 1.0,
        lambda_lpips: float = 1.0,
        w_center: float = 10.0,
        w_sep: float = 5.0,
        lambda_cls: float = 2.0,
        margin: float = 2.0,
        classifier_mean: float = 0.456,
        classifier_std: float = 0.224,
    ) -> None:
        super().__init__()
        self.classifier = classifier
        self.perceptual = LPIPSLoss()
        self.centers = nn.Parameter(torch.randn(num_classes, class_dim) * 0.02)
        self.lambda_bce = lambda_bce
        self.lambda_lpips = lambda_lpips
        self.w_center = w_center
        self.w_sep = w_sep
        self.lambda_cls = lambda_cls
        self.margin = margin
        self.classifier_mean = classifier_mean
        self.classifier_std = classifier_std

    def train(self, mode: bool = True):
        super().train(mode)
        self.classifier.eval()
        self.perceptual.eval()
        return self

    def center_loss(self, z_class: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        target_centers = self.centers[labels]
        return ((z_class - target_centers) ** 2).sum(dim=1).mean()

    def separation_loss(self) -> torch.Tensor:
        pair_losses = []
        for i, j in itertools.combinations(range(self.centers.size(0)), 2):
            distance = torch.norm(self.centers[i] - self.centers[j], p=2)
            pair_losses.append(F.relu(self.margin - distance))
        return torch.stack(pair_losses).mean()

    def forward(
        self,
        output: dict[str, torch.Tensor],
        target: torch.Tensor,
        labels: torch.Tensor,
        beta_kl: float,
    ) -> dict[str, torch.Tensor]:
        recon = output["recon"]
        bce = F.binary_cross_entropy(recon, target, reduction="mean")
        lpips_value = self.perceptual(recon, target)
        reconstruction = self.lambda_bce * bce + self.lambda_lpips * lpips_value

        kl = kl_divergence(output["mu"], output["logvar"])
        center = self.center_loss(output["z_class"], labels)
        separation = self.separation_loss()

        logits = self.classifier(
            normalize_for_classifier(recon, self.classifier_mean, self.classifier_std)
        )
        classifier_loss = F.cross_entropy(logits, labels)

        total = (
            reconstruction
            + beta_kl * kl
            + self.w_center * center
            + self.w_sep * separation
            + self.lambda_cls * classifier_loss
        )
        return {
            "total": total,
            "reconstruction": reconstruction.detach(),
            "bce": bce.detach(),
            "lpips": lpips_value.detach(),
            "kl": kl.detach(),
            "center": center.detach(),
            "separation": separation.detach(),
            "classifier": classifier_loss.detach(),
        }
