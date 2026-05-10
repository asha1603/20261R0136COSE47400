from __future__ import annotations

import torch
import torch.nn.functional as F


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def normalize_for_classifier(
    images: torch.Tensor,
    mean: float = 0.456,
    std: float = 0.224,
) -> torch.Tensor:
    return (images - mean) / std


def mid_cvae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    labels: torch.Tensor,
    classifier=None,
    beta_kl: float = 1.0,
    classifier_loss_weight: float = 2.0,
    classifier_mean: float = 0.456,
    classifier_std: float = 0.224,
) -> dict[str, torch.Tensor]:
    reconstruction = F.binary_cross_entropy(recon, target, reduction="mean")
    kl = kl_divergence(mu, logvar)
    classifier_loss = recon.new_tensor(0.0)

    if classifier is not None and classifier_loss_weight > 0:
        logits = classifier(normalize_for_classifier(recon, classifier_mean, classifier_std))
        classifier_loss = F.cross_entropy(logits, labels)

    total = reconstruction + beta_kl * kl + classifier_loss_weight * classifier_loss
    return {
        "total": total,
        "reconstruction": reconstruction.detach(),
        "kl": kl.detach(),
        "classifier": classifier_loss.detach(),
    }
