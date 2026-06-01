"""
train_step_patch.py

This is the part you should merge into your existing train loop.
It assumes:
    model = ImprovedFiLMUNetCVAE(...)
    loss_fn = FiLMCVAELoss(...)
    classifier = frozen classifier or None

Main training trick:
- recon branch uses original label y and full skip_scale=1.0
- translation branch uses a different target label and weaker skip_scale=0.35
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from improved_losses import FiLMCVAELoss, sample_different_labels


def warmup_weight(epoch: int, max_value: float = 1.0, warmup_epochs: int = 10) -> float:
    if warmup_epochs <= 0:
        return max_value
    return max_value * min(1.0, epoch / warmup_epochs)


def train_one_batch_improved_film(
    model: nn.Module,
    batch,
    optimizer: torch.optim.Optimizer,
    loss_fn: FiLMCVAELoss,
    device: torch.device,
    epoch: int,
    classifier: Optional[nn.Module] = None,
    translation_skip_scale: float = 0.35,
    cls_warmup_epochs: int = 10,
    kl_warmup_epochs: int = 20,
) -> Dict[str, float]:
    """
    Drop-in batch training step.

    batch can be:
        (x, y)
        (x, y, subject_id)
        dictionary with keys "image" and "label"
    """
    model.train()
    if classifier is not None:
        classifier.eval()
        for p in classifier.parameters():
            p.requires_grad = False

    # Support common dataset return formats.
    if isinstance(batch, dict):
        x = batch["image"]
        y = batch["label"]
    else:
        x = batch[0]
        y = batch[1]

    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True).long()

    # Different target labels for the translation branch.
    y_target = sample_different_labels(y, loss_fn.num_classes)

    optimizer.zero_grad(set_to_none=True)

    # Branch A: reconstruction. Full skip scale is okay.
    recon, mu, logvar, _ = model(x, y, skip_scale=1.0)

    # Branch B: translation. Reduce skip scale to prevent copy-only behavior.
    translated, mu_t, logvar_t, _ = model(x, y_target, skip_scale=translation_skip_scale)

    loss_dict = loss_fn(
        x=x,
        y=y,
        recon=recon,
        recon_mu=mu,
        recon_logvar=logvar,
        translated=translated,
        trans_y=y_target,
        trans_mu=mu_t,
        trans_logvar=logvar_t,
        classifier=classifier,
        cls_weight_scale=warmup_weight(epoch, 1.0, cls_warmup_epochs),
        kl_weight_scale=warmup_weight(epoch, 1.0, kl_warmup_epochs),
    )

    loss = loss_dict["loss"]
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return {k: float(v.detach().cpu()) for k, v in loss_dict.items()}
