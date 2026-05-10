from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def mse(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(recon, target, reduction="mean")


def psnr(recon: torch.Tensor, target: torch.Tensor, max_value: float = 1.0) -> torch.Tensor:
    value = mse(recon, target).clamp_min(1e-12)
    return 20 * torch.log10(torch.tensor(max_value, device=recon.device)) - 10 * torch.log10(value)


def ssim(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Batch-level global SSIM for grayscale tensors in [0, 1]."""
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    dims = (1, 2, 3)
    mu_x = recon.mean(dim=dims)
    mu_y = target.mean(dim=dims)
    sigma_x = ((recon - mu_x[:, None, None, None]) ** 2).mean(dim=dims)
    sigma_y = ((target - mu_y[:, None, None, None]) ** 2).mean(dim=dims)
    sigma_xy = (
        (recon - mu_x[:, None, None, None])
        * (target - mu_y[:, None, None, None])
    ).mean(dim=dims)
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x.pow(2) + mu_y.pow(2) + c1) * (sigma_x + sigma_y + c2)
    return (numerator / denominator.clamp_min(1e-12)).mean()


def update_confusion_matrix(matrix: torch.Tensor, labels: torch.Tensor, preds: torch.Tensor) -> None:
    for label, pred in zip(labels.view(-1), preds.view(-1)):
        matrix[int(label), int(pred)] += 1


def format_confusion_matrix(matrix: torch.Tensor) -> str:
    rows = ["[" + ", ".join(str(int(v)) for v in row.tolist()) + "]" for row in matrix]
    return "\n".join(rows)


def safe_mean(total: float, count: int) -> float:
    return total / max(1, count)


def safe_psnr_from_mse(value: float) -> float:
    return 20 * math.log10(1.0) - 10 * math.log10(max(value, 1e-12))
