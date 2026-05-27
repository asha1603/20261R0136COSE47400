from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from final_film_cvae.data import CLASS_TO_INDEX, MRISliceDataset
from final_film_cvae.losses import FinalFiLMLoss, kl_annealing_beta
from final_film_cvae.models import FinalFiLMDisentangledCVAE, ResNet18MRIClassifier
from final_film_cvae.utils import AverageMeter, ensure_dir, get_device, load_checkpoint, save_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train final FiLM-conditioned disentangled CVAE.")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--classifier-checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="checkpoints/final_film_cvae")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--content-dim", type=int, default=96)
    parser.add_argument("--class-dim", type=int, default=32)
    parser.add_argument("--class-embed-dim", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta-kl-max", type=float, default=2.0)
    parser.add_argument("--kl-warmup-epochs", type=int, default=10)
    parser.add_argument("--lambda-bce", type=float, default=1.0)
    parser.add_argument("--lambda-lpips", type=float, default=1.0)
    parser.add_argument("--w-center", type=float, default=10.0)
    parser.add_argument("--w-sep", type=float, default=5.0)
    parser.add_argument("--lambda-cls", type=float, default=2.0)
    parser.add_argument("--margin", type=float, default=2.0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--mean", type=float, default=0.456)
    parser.add_argument("--std", type=float, default=0.224)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def load_frozen_classifier(path: str, device: torch.device) -> ResNet18MRIClassifier:
    checkpoint = load_checkpoint(path, device)
    classifier = ResNet18MRIClassifier(
        num_classes=int(checkpoint.get("num_classes", len(CLASS_TO_INDEX)))
    ).to(device)
    classifier.load_state_dict(checkpoint["model_state"])
    classifier.eval()
    for parameter in classifier.parameters():
        parameter.requires_grad = False
    return classifier


def run_epoch(model, objective, loader, optimizer, device, beta_kl: float, train: bool):
    model.train(train)
    objective.train(train)
    meters = {
        key: AverageMeter() for key in
        ["total", "reconstruction", "bce", "lpips", "kl", "center", "separation", "classifier"]
    }
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels, _, _ in tqdm(loader, desc="final train" if train else "final val"):
            images, labels = images.to(device), labels.to(device)
            output = model(images, labels)  # known-label reconstruction during training
            losses = objective(output, images, labels, beta_kl)
            if train:
                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + [objective.centers], max_norm=1.0
                )
                optimizer.step()
            batch_size = images.size(0)
            for key in meters:
                meters[key].update(losses[key].item(), batch_size)
    return {key: value.avg for key, value in meters.items()}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(prefer_cuda=not args.cpu)
    output_dir = ensure_dir(args.output_dir)

    train_dataset = MRISliceDataset(
        args.data_dir, split="train", image_size=args.image_size,
        val_ratio=args.val_ratio, seed=args.seed, normalize=False,
    )
    val_dataset = MRISliceDataset(
        args.data_dir, split="val", image_size=args.image_size,
        val_ratio=args.val_ratio, seed=args.seed, normalize=False,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )

    classifier = load_frozen_classifier(args.classifier_checkpoint, device)
    model = FinalFiLMDisentangledCVAE(
        image_size=args.image_size, latent_dim=args.latent_dim,
        content_dim=args.content_dim, class_dim=args.class_dim,
        num_classes=len(CLASS_TO_INDEX), class_embed_dim=args.class_embed_dim,
    ).to(device)
    objective = FinalFiLMLoss(
        classifier=classifier, num_classes=len(CLASS_TO_INDEX), class_dim=args.class_dim,
        lambda_bce=args.lambda_bce, lambda_lpips=args.lambda_lpips,
        w_center=args.w_center, w_sep=args.w_sep, lambda_cls=args.lambda_cls,
        margin=args.margin, classifier_mean=args.mean, classifier_std=args.std,
    ).to(device)

    trainable = list(model.parameters()) + [objective.centers]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    best_val = float("inf")
    print(f"Device: {device}")
    print(f"Train images: {len(train_dataset)} | Val images: {len(val_dataset)}")
    print("Loss weights: BCE=1, LPIPS=1, beta_KL_max=2, center=10, separation=5, cls=2, margin=2")

    for epoch in range(1, args.epochs + 1):
        beta_kl = kl_annealing_beta(epoch, args.beta_kl_max, args.kl_warmup_epochs)
        train_stats = run_epoch(model, objective, train_loader, optimizer, device, beta_kl, True)
        val_stats = run_epoch(model, objective, val_loader, None, device, beta_kl, False)
        scheduler.step()
        print(
            f"Epoch {epoch:03d}/{args.epochs} | beta {beta_kl:.3f} | "
            f"train {train_stats['total']:.4f} | val {val_stats['total']:.4f} "
            f"(BCE {val_stats['bce']:.4f}, LPIPS {val_stats['lpips']:.4f}, "
            f"KL {val_stats['kl']:.4f}, cen {val_stats['center']:.4f}, "
            f"sep {val_stats['separation']:.4f}, cls {val_stats['classifier']:.4f})"
        )
        if val_stats["total"] < best_val:
            best_val = val_stats["total"]
            save_checkpoint(
                output_dir / "best_final_film_cvae.pth",
                model_state=model.state_dict(),
                centers=objective.centers.detach().cpu(),
                epoch=epoch, val_loss=best_val,
                image_size=args.image_size, latent_dim=args.latent_dim,
                content_dim=args.content_dim, class_dim=args.class_dim,
                class_embed_dim=args.class_embed_dim,
                num_classes=len(CLASS_TO_INDEX),
                beta_kl_max=args.beta_kl_max, kl_warmup_epochs=args.kl_warmup_epochs,
                lambda_bce=args.lambda_bce, lambda_lpips=args.lambda_lpips,
                w_center=args.w_center, w_sep=args.w_sep,
                lambda_cls=args.lambda_cls, margin=args.margin,
                classes=CLASS_TO_INDEX,
            )
            print(f"Saved best final model: {output_dir / 'best_final_film_cvae.pth'}")


if __name__ == "__main__":
    main()
