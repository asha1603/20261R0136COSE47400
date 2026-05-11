from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mid_unet_cvae.losses import mid_cvae_loss
from mid_unet_cvae.models import MidUNetCVAE, SmallMRIClassifier
from mid_unet_cvae.utils import set_seed


def main() -> None:
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.rand(2, 1, 224, 224, device=device)
    y = torch.tensor([0, 2], dtype=torch.long, device=device)

    cvae = MidUNetCVAE(image_size=224, latent_dim=128, num_classes=3).to(device)
    classifier = SmallMRIClassifier(num_classes=3).to(device)
    classifier.eval()

    output = cvae(x, y)
    losses = mid_cvae_loss(
        recon=output["recon"],
        target=x,
        mu=output["mu"],
        logvar=output["logvar"],
        labels=y,
        classifier=classifier,
        beta_kl=1.0,
        classifier_loss_weight=0.1,
    )
    losses["total"].backward()

    assert output["recon"].shape == x.shape
    assert output["mu"].shape == (2, 128)
    print("Smoke test passed.")
    print(f"Device: {device}")
    print(f"Loss: {losses['total'].item():.4f}")


if __name__ == "__main__":
    main()
