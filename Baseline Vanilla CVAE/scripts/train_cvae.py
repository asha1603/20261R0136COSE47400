from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mid_unet_cvae.data import CLASS_TO_INDEX, MRISliceDataset
from mid_unet_cvae.losses import mid_cvae_loss
from mid_unet_cvae.models import MidUNetCVAE, SmallMRIClassifier
from mid_unet_cvae.utils import AverageMeter, ensure_dir, get_device, load_checkpoint, save_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the mid-presentation U-Net CVAE.")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--classifier-checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="checkpoints/cvae")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--class-embed-dim", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta-kl", type=float, default=1.0)
    parser.add_argument("--classifier-loss-weight", type=float, default=2.0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--mean", type=float, default=0.456)
    parser.add_argument("--std", type=float, default=0.224)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def load_frozen_classifier(path: str, device: torch.device) -> SmallMRIClassifier:
    checkpoint = load_checkpoint(path, device)
    num_classes = int(checkpoint.get("num_classes", len(CLASS_TO_INDEX)))
    classifier = SmallMRIClassifier(num_classes=num_classes).to(device)
    classifier.load_state_dict(checkpoint["model_state"])
    classifier.eval()
    for param in classifier.parameters():
        param.requires_grad = False
    return classifier


def run_epoch(model, classifier, loader, optimizer, args, device, train: bool):
    model.train(train)
    loss_meter = AverageMeter()
    recon_meter = AverageMeter()
    kl_meter = AverageMeter()
    cls_meter = AverageMeter()

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels, _, _ in tqdm(loader, desc="train" if train else "val"):
            images = images.to(device)
            labels = labels.to(device)
            output = model(images, labels)
            losses = mid_cvae_loss(
                recon=output["recon"],
                target=images,
                mu=output["mu"],
                logvar=output["logvar"],
                labels=labels,
                classifier=classifier,
                beta_kl=args.beta_kl,
                classifier_loss_weight=args.classifier_loss_weight,
                classifier_mean=args.mean,
                classifier_std=args.std,
            )

            if train:
                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            batch_size = images.size(0)
            loss_meter.update(losses["total"].item(), batch_size)
            recon_meter.update(losses["reconstruction"].item(), batch_size)
            kl_meter.update(losses["kl"].item(), batch_size)
            cls_meter.update(losses["classifier"].item(), batch_size)

    return {
        "loss": loss_meter.avg,
        "reconstruction": recon_meter.avg,
        "kl": kl_meter.avg,
        "classifier": cls_meter.avg,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(prefer_cuda=not args.cpu)
    output_dir = ensure_dir(args.output_dir)

    train_dataset = MRISliceDataset(
        args.data_dir,
        split="train",
        image_size=args.image_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        normalize=False,
    )
    val_dataset = MRISliceDataset(
        args.data_dir,
        split="val",
        image_size=args.image_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        normalize=False,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    classifier = load_frozen_classifier(args.classifier_checkpoint, device)
    model = MidUNetCVAE(
        image_size=args.image_size,
        latent_dim=args.latent_dim,
        num_classes=len(CLASS_TO_INDEX),
        class_embed_dim=args.class_embed_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    best_val = float("inf")

    print(f"Device: {device}")
    print(f"Train images: {len(train_dataset)} | Val images: {len(val_dataset)}")

    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(model, classifier, train_loader, optimizer, args, device, train=True)
        val_stats = run_epoch(model, classifier, val_loader, optimizer, args, device, train=False)
        scheduler.step()

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train {train_stats['loss']:.4f} | "
            f"val {val_stats['loss']:.4f} "
            f"(rec {val_stats['reconstruction']:.4f}, kl {val_stats['kl']:.4f}, cls {val_stats['classifier']:.4f})"
        )

        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            save_checkpoint(
                output_dir / "best_cvae.pth",
                model_state=model.state_dict(),
                epoch=epoch,
                val_loss=best_val,
                image_size=args.image_size,
                latent_dim=args.latent_dim,
                class_embed_dim=args.class_embed_dim,
                num_classes=len(CLASS_TO_INDEX),
                beta_kl=args.beta_kl,
                classifier_loss_weight=args.classifier_loss_weight,
                classes=CLASS_TO_INDEX,
            )
            print(f"Saved best CVAE: {output_dir / 'best_cvae.pth'}")


if __name__ == "__main__":
    main()
