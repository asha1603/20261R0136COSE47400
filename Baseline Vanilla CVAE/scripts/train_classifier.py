from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mid_unet_cvae.data import CLASS_TO_INDEX, MRISliceDataset
from mid_unet_cvae.metrics import format_confusion_matrix, update_confusion_matrix
from mid_unet_cvae.models import SmallMRIClassifier
from mid_unet_cvae.utils import AverageMeter, ensure_dir, get_device, save_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the frozen MRI classifier guide.")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="checkpoints/classifier")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--mean", type=float, default=0.456)
    parser.add_argument("--std", type=float, default=0.224)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def run_epoch(model, loader, optimizer, device, train: bool, num_classes: int):
    model.train(train)
    loss_meter = AverageMeter()
    correct = 0
    total = 0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels, _, _ in tqdm(loader, desc="train" if train else "val"):
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += batch_size
            loss_meter.update(loss.item(), batch_size)
            update_confusion_matrix(confusion, labels.cpu(), preds.cpu())

    return loss_meter.avg, correct / max(1, total), confusion


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
        normalize=True,
        mean=args.mean,
        std=args.std,
    )
    val_dataset = MRISliceDataset(
        args.data_dir,
        split="val",
        image_size=args.image_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        normalize=True,
        mean=args.mean,
        std=args.std,
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

    model = SmallMRIClassifier(num_classes=len(CLASS_TO_INDEX)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    best_acc = -1.0

    print(f"Device: {device}")
    print(f"Train images: {len(train_dataset)} | Val images: {len(val_dataset)}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, _ = run_epoch(
            model, train_loader, optimizer, device, train=True, num_classes=len(CLASS_TO_INDEX)
        )
        val_loss, val_acc, confusion = run_epoch(
            model, val_loader, optimizer, device, train=False, num_classes=len(CLASS_TO_INDEX)
        )
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                output_dir / "best_classifier.pth",
                model_state=model.state_dict(),
                epoch=epoch,
                val_accuracy=val_acc,
                image_size=args.image_size,
                num_classes=len(CLASS_TO_INDEX),
                mean=args.mean,
                std=args.std,
                classes=CLASS_TO_INDEX,
            )
            print(f"Saved best classifier: {output_dir / 'best_classifier.pth'}")

    print("Final validation confusion matrix:")
    print(format_confusion_matrix(confusion))


if __name__ == "__main__":
    main()
