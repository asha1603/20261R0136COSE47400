import torch
import torch.nn.functional as F


def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def edge_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
    target_dx = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
    pred_dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
    target_dy = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


def total_variation_loss(x: torch.Tensor) -> torch.Tensor:
    return (
        torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
        + torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    )


def low_frequency_structure_loss(pred: torch.Tensor, target: torch.Tensor, kernel_size: int = 8) -> torch.Tensor:
    return F.l1_loss(
        F.avg_pool2d(pred, kernel_size=kernel_size),
        F.avg_pool2d(target, kernel_size=kernel_size),
    )
