"""
improved_losses.py

Loss functions for improving FiLM-CVAE training.
The important idea is to separate:
1. same-label reconstruction loss
2. different-label translation control loss

This avoids the common problem where the model only learns identity reconstruction.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_different_labels(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    """For each label y_i, sample a different target class."""
    offset = torch.randint(1, num_classes, size=y.shape, device=y.device)
    return (y + offset) % num_classes


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Mean KL divergence between N(mu, sigma) and N(0, I)."""
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def edge_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Simple gradient/edge preservation loss."""
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    tgt_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    tgt_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(pred_dx, tgt_dx) + F.l1_loss(pred_dy, tgt_dy)


def low_frequency_loss(pred: torch.Tensor, target: torch.Tensor, pool: int = 8) -> torch.Tensor:
    """
    Preserve broad anatomy without forcing pixel-perfect copying.
    Useful for unpaired target-stage translation.
    """
    return F.l1_loss(F.avg_pool2d(pred, pool), F.avg_pool2d(target, pool))


def total_variation_loss(x: torch.Tensor) -> torch.Tensor:
    return (
        torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
        + torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    )


def normalize_for_classifier(x: torch.Tensor, mean: float = 0.456, std: float = 0.224) -> torch.Tensor:
    """
    Use this only if your classifier was trained with Normalize(mean=[0.456], std=[0.224]).
    If your classifier used different normalization, change mean/std here.
    """
    return (x - mean) / std


class FiLMCVAELoss(nn.Module):
    """
    Combined loss for the improved FiLM-CVAE.

    It returns a dictionary so you can log each component.
    """

    def __init__(
        self,
        num_classes: int = 3,
        lambda_l1: float = 8.0,
        lambda_bce: float = 1.0,
        lambda_edge: float = 2.0,
        beta_kl: float = 0.05,
        lambda_trans_lowfreq: float = 1.0,
        lambda_trans_edge: float = 0.5,
        lambda_cls: float = 1.5,
        lambda_tv: float = 0.02,
        lambda_diversity: float = 0.25,
        min_translation_delta: float = 0.015,
        classifier_mean: float = 0.456,
        classifier_std: float = 0.224,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_l1 = lambda_l1
        self.lambda_bce = lambda_bce
        self.lambda_edge = lambda_edge
        self.beta_kl = beta_kl
        self.lambda_trans_lowfreq = lambda_trans_lowfreq
        self.lambda_trans_edge = lambda_trans_edge
        self.lambda_cls = lambda_cls
        self.lambda_tv = lambda_tv
        self.lambda_diversity = lambda_diversity
        self.min_translation_delta = min_translation_delta
        self.classifier_mean = classifier_mean
        self.classifier_std = classifier_std

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        recon: torch.Tensor,
        recon_mu: torch.Tensor,
        recon_logvar: torch.Tensor,
        translated: torch.Tensor,
        trans_y: torch.Tensor,
        trans_mu: torch.Tensor,
        trans_logvar: torch.Tensor,
        classifier: Optional[nn.Module] = None,
        cls_weight_scale: float = 1.0,
        kl_weight_scale: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        # 1) Same-label reconstruction branch
        rec_l1 = F.l1_loss(recon, x)
        rec_bce = F.binary_cross_entropy(recon, x)
        rec_edge = edge_loss(recon, x)
        rec_kl = kl_divergence(recon_mu, recon_logvar)

        # 2) Different-label translation branch
        # No paired ground truth exists, so use weak structure preservation.
        trans_lowfreq = low_frequency_loss(translated, x)
        trans_edge = edge_loss(translated, x)
        trans_kl = kl_divergence(trans_mu, trans_logvar)
        tv = total_variation_loss(translated)

        # Encourage translated output to be different from plain reconstruction.
        # This prevents all target classes from becoming visually identical.
        delta = torch.mean(torch.abs(translated - recon), dim=(1, 2, 3))
        diversity = torch.relu(self.min_translation_delta - delta).mean()

        if classifier is not None and self.lambda_cls > 0:
            with torch.set_grad_enabled(True):
                logits = classifier(
                    normalize_for_classifier(
                        translated,
                        mean=self.classifier_mean,
                        std=self.classifier_std,
                    )
                )
            cls = F.cross_entropy(logits, trans_y)
            target_conf = torch.softmax(logits.detach(), dim=1).gather(1, trans_y[:, None]).mean()
        else:
            cls = translated.new_tensor(0.0)
            target_conf = translated.new_tensor(0.0)

        loss = (
            self.lambda_l1 * rec_l1
            + self.lambda_bce * rec_bce
            + self.lambda_edge * rec_edge
            + self.beta_kl * kl_weight_scale * rec_kl
            + self.lambda_trans_lowfreq * trans_lowfreq
            + self.lambda_trans_edge * trans_edge
            + 0.5 * self.beta_kl * kl_weight_scale * trans_kl
            + self.lambda_tv * tv
            + self.lambda_diversity * diversity
            + self.lambda_cls * cls_weight_scale * cls
        )

        return {
            "loss": loss,
            "rec_l1": rec_l1.detach(),
            "rec_bce": rec_bce.detach(),
            "rec_edge": rec_edge.detach(),
            "rec_kl": rec_kl.detach(),
            "trans_lowfreq": trans_lowfreq.detach(),
            "trans_edge": trans_edge.detach(),
            "trans_kl": trans_kl.detach(),
            "tv": tv.detach(),
            "diversity": diversity.detach(),
            "cls": cls.detach(),
            "target_conf": target_conf.detach(),
            "delta": delta.mean().detach(),
        }
