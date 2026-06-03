import argparse
import os

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import MRISubjectDataset, CLASSES, NUM_CLASSES
from src.models import MRIClassifier
from src.utils import get_device, set_seed, ensure_dir, save_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--output_dir", type=str, default="outputs/classifier")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model_name", type=str, default="resnet18")
    p.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained weights. Keep false if your professor disallows pretrained models.")
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, preds = [], []
    total_loss = 0.0
    n = 0
    for x, y, *_ in tqdm(loader, desc="Val", leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
        ys.extend(y.cpu().tolist())
        preds.extend(logits.argmax(dim=1).cpu().tolist())
    return total_loss / max(1, n), accuracy_score(ys, preds), ys, preds


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    ensure_dir(args.output_dir)

    train_ds = MRISubjectDataset(args.data_dir, "train", args.image_size, classifier_transform=True, seed=args.seed)
    val_ds = MRISubjectDataset(args.data_dir, "val", args.image_size, classifier_transform=True, seed=args.seed)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    print(f"Device: {device}")
    print(f"Train images: {len(train_ds)} | Val images: {len(val_ds)}")
    print(f"Classes: {CLASSES}")

    model = MRIClassifier(model_name=args.model_name, num_classes=NUM_CLASSES, pretrained=args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = -1.0
    best_path = os.path.join(args.output_dir, "best_classifier.pth")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        total = 0
        for x, y, *_ in tqdm(train_loader, desc=f"Train {epoch:03d}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item() * x.size(0)
            total += x.size(0)
        scheduler.step()

        val_loss, val_acc, ys, preds = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:03d}: train_loss={running/max(1,total):.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(best_path, {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "classes": CLASSES,
                "model_name": args.model_name,
            })
            print(f"Saved best classifier: {best_path}")

    _, val_acc, ys, preds = evaluate(model, val_loader, device)
    print("Final validation accuracy:", val_acc)
    print("Confusion matrix:")
    print(confusion_matrix(ys, preds))
    print(classification_report(ys, preds, target_names=[CLASSES[i] for i in range(NUM_CLASSES)]))


if __name__ == "__main__":
    main()
