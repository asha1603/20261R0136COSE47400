"""Run this first to check that the improved FiLM model works."""

import torch

from improved_film_modules import ImprovedFiLMUNetCVAE, count_parameters
from improved_losses import FiLMCVAELoss, sample_different_labels


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImprovedFiLMUNetCVAE(num_classes=3, latent_dim=64, class_dim=16, base_channels=8).to(device)
    loss_fn = FiLMCVAELoss(num_classes=3)

    x = torch.rand(1, 1, 224, 224).to(device)
    y = torch.tensor([0], dtype=torch.long).to(device)
    y_t = sample_different_labels(y, 3)

    recon, mu, logvar, _ = model(x, y, skip_scale=1.0)
    trans, mu_t, logvar_t, _ = model(x, y_t, skip_scale=0.35)

    loss_dict = loss_fn(
        x=x,
        y=y,
        recon=recon,
        recon_mu=mu,
        recon_logvar=logvar,
        translated=trans,
        trans_y=y_t,
        trans_mu=mu_t,
        trans_logvar=logvar_t,
        classifier=None,
    )

    print("Device:", device)
    print("Trainable parameters:", count_parameters(model))
    print("Recon shape:", tuple(recon.shape))
    print("Translated shape:", tuple(trans.shape))
    print("Loss:", float(loss_dict["loss"].detach().cpu()))
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
