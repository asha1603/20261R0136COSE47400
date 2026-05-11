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
from mid_unet_cvae.losses import normalize_for_classifier
from mid_unet_cvae.metrics import format_confusion_matrix, mse, psnr, ssim, update_confusion_matrix
from mid_unet_cvae.models import MidUNetCVAE, SmallMRIClassifier
from mid_unet_cvae.utils import AverageMeter, get_device, load_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained mid U-Net CVAE.")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--cvae-checkpoint", type=str, required=True)
    parser.add_argument("--classifier-checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, choices=["train", "val", "all"], default="val")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--mean", type=float, default=0.456)
    parser.add_argument("--std", type=float, default=0.224)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def load_cvae(path: str, device: torch.device) -> MidUNetCVAE:
    checkpoint = load_checkpoint(path, device)
    model = MidUNetCVAE(
        image_size=int(checkpoint.get("image_size", 224)),
        latent_dim=int(checkpoint.get("latent_dim", 128)),
        num_classes=int(checkpoint.get("num_classes", len(CLASS_TO_INDEX))),
        class_embed_dim=int(checkpoint.get("class_embed_dim", 32)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def load_classifier(path: str | None, device: torch.device):
    if path is None:
        return None
    checkpoint = load_checkpoint(path, device)
    classifier = SmallMRIClassifier(num_classes=int(checkpoint.get("num_classes", len(CLASS_TO_INDEX)))).to(device)
    classifier.load_state_dict(checkpoint["model_state"])
    classifier.eval()
    return classifier


@torch.no_grad()
def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(prefer_cuda=not args.cpu)
    cvae = load_cvae(args.cvae_checkpoint, device)
    classifier = load_classifier(args.classifier_checkpoint, device)
    dataset = MRISliceDataset(
        args.data_dir,
        split=args.split,
        image_size=cvae.image_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        normalize=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    mse_meter = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    confidence_meter = AverageMeter()
    correct = 0
    total = 0
    confusion = torch.zeros(len(CLASS_TO_INDEX), len(CLASS_TO_INDEX), dtype=torch.long)

    for images, labels, _, _ in tqdm(loader, desc="evaluate"):
        images = images.to(device)
        labels = labels.to(device)
        output = cvae(images, labels)
        recon = output["recon"]
        batch_size = images.size(0)

        mse_meter.update(mse(recon, images).item(), batch_size)
        psnr_meter.update(psnr(recon, images).item(), batch_size)
        ssim_meter.update(ssim(recon, images).item(), batch_size)

        if classifier is not None:
            logits = classifier(normalize_for_classifier(recon, args.mean, args.std))
            probs = logits.softmax(dim=1)
            preds = logits.argmax(dim=1)
            confidence = probs.gather(1, labels.view(-1, 1)).mean().item()
            confidence_meter.update(confidence, batch_size)
            correct += (preds == labels).sum().item()
            total += batch_size
            update_confusion_matrix(confusion, labels.cpu(), preds.cpu())

    print(f"Split: {args.split}")
    print(f"Images: {len(dataset)}")
    print(f"MSE: {mse_meter.avg:.6f}")
    print(f"PSNR: {psnr_meter.avg:.4f}")
    print(f"SSIM: {ssim_meter.avg:.4f}")

    if classifier is not None:
        print(f"Target confidence: {confidence_meter.avg:.4f}")
        print(f"Classifier accuracy on reconstructions: {correct / max(1, total):.4f}")
        print("Confusion matrix:")
        print(format_confusion_matrix(confusion))


if __name__ == "__main__":
    main()
