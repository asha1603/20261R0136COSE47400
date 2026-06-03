import torch

from src.models import ImprovedFiLMGatedCVAE


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImprovedFiLMGatedCVAE(image_size=224, latent_dim=32, class_dim=8, base_channels=8, film_scale=0.05).to(device)
    x = torch.rand(1, 1, 224, 224).to(device)
    y = torch.tensor([0], dtype=torch.long).to(device)

    out, mu, logvar, z = model(x, y, skip_scale=0.6)
    assert out.shape == x.shape, out.shape
    assert mu.shape == (1, 32), mu.shape
    assert logvar.shape == (1, 32), logvar.shape
    assert z.shape == (1, 32), z.shape
    print("Smoke test passed.")
    print(f"Output shape: {out.shape}")


if __name__ == "__main__":
    main()
