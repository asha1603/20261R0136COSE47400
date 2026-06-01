"""
example_train_loop.py

Minimal example showing how to use the improved FiLM patch.
This is NOT meant to replace your whole project; copy the relevant parts into your current training file.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from improved_film_modules import ImprovedFiLMUNetCVAE, count_parameters
from improved_losses import FiLMCVAELoss
from train_step_patch import train_one_batch_improved_film


NUM_CLASSES = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # Replace this with your existing OASIS dataset.
    # This example assumes ImageFolder structure:
    # data/Non Demented, data/Very mild Dementia, data/Mild Dementia
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder("data", transform=transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)

    model = ImprovedFiLMUNetCVAE(
        img_size=224,
        in_channels=1,
        num_classes=NUM_CLASSES,
        latent_dim=128,
        class_dim=32,
        base_channels=32,
    ).to(DEVICE)

    print("Trainable parameters:", count_parameters(model))

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    loss_fn = FiLMCVAELoss(
        num_classes=NUM_CLASSES,
        lambda_l1=8.0,
        lambda_bce=1.0,
        lambda_edge=2.0,
        beta_kl=0.05,
        lambda_cls=1.5,
    )

    # Use your frozen classifier here if available.
    # classifier = load_your_classifier(...).to(DEVICE).eval()
    classifier = None

    for epoch in range(1, 51):
        running = {}
        for batch in loader:
            metrics = train_one_batch_improved_film(
                model=model,
                batch=batch,
                optimizer=optimizer,
                loss_fn=loss_fn,
                device=DEVICE,
                epoch=epoch,
                classifier=classifier,
                translation_skip_scale=0.35,
            )
            for k, v in metrics.items():
                running[k] = running.get(k, 0.0) + v

        n = max(1, len(loader))
        print(f"Epoch {epoch}", {k: round(v / n, 5) for k, v in running.items()})

        torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, "improved_film_cvae_latest.pth")


if __name__ == "__main__":
    main()
