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

from final_film_cvae.data import CLASS_TO_INDEX, MRISliceDataset
from final_film_cvae.losses import normalize_for_classifier
from final_film_cvae.metrics import format_confusion_matrix, mse, psnr, ssim, update_confusion_matrix
from final_film_cvae.models import FinalFiLMDisentangledCVAE, ResNet18MRIClassifier
from final_film_cvae.utils import AverageMeter, get_device, load_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate final FiLM CVAE reconstructions and stage control.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--cvae-checkpoint", required=True)
    parser.add_argument("--classifier-checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "val", "all"], default="val")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--mean", type=float, default=0.456)
    parser.add_argument("--std", type=float, default=0.224)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def load_models(args, device):
    ckp = load_checkpoint(args.cvae_checkpoint, device)
    model = FinalFiLMDisentangledCVAE(
        image_size=int(ckp.get("image_size", 224)),
        latent_dim=int(ckp.get("latent_dim", 128)),
        content_dim=int(ckp.get("content_dim", 96)),
        class_dim=int(ckp.get("class_dim", 32)),
        num_classes=int(ckp.get("num_classes", 3)),
        class_embed_dim=int(ckp.get("class_embed_dim", 32)),
    ).to(device)
    model.load_state_dict(ckp["model_state"])
    model.eval()

    cls_ckp = load_checkpoint(args.classifier_checkpoint, device)
    classifier = ResNet18MRIClassifier(
        num_classes=int(cls_ckp.get("num_classes", len(CLASS_TO_INDEX)))
    ).to(device)
    classifier.load_state_dict(cls_ckp["model_state"])
    classifier.eval()
    return model, classifier


@torch.no_grad()
def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(prefer_cuda=not args.cpu)
    model, classifier = load_models(args, device)

    dataset = MRISliceDataset(
        args.data_dir, split=args.split, image_size=model.image_size,
        val_ratio=args.val_ratio, seed=args.seed, normalize=False,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )

    meters = {name: AverageMeter() for name in ["mse", "psnr", "ssim", "bce", "same_conf", "translation_conf"]}
    confusion = torch.zeros(len(CLASS_TO_INDEX), len(CLASS_TO_INDEX), dtype=torch.long)
    correct, total = 0, 0

    for images, labels, _, _ in tqdm(loader, desc="evaluate final"):
        images, labels = images.to(device), labels.to(device)
        recon = model(images, labels)["recon"]
        batch = images.size(0)
        meters["mse"].update(mse(recon, images).item(), batch)
        meters["psnr"].update(psnr(recon, images).item(), batch)
        meters["ssim"].update(ssim(recon, images).item(), batch)
        meters["bce"].update(F.binary_cross_entropy(recon, images).item(), batch)

        logits = classifier(normalize_for_classifier(recon, args.mean, args.std))
        probs = logits.softmax(dim=1)
        preds = logits.argmax(dim=1)
        meters["same_conf"].update(probs.gather(1, labels[:, None]).mean().item(), batch)
        correct += (preds == labels).sum().item()
        total += batch
        update_confusion_matrix(confusion, labels.cpu(), preds.cpu())

        # Counterfactual stage control: target confidence only. There is no paired target MRI.
        per_target = []
        for target_label in range(len(CLASS_TO_INDEX)):
            targets = torch.full_like(labels, target_label)
            generated = model.translate(images, targets, deterministic=True)
            target_probs = classifier(
                normalize_for_classifier(generated, args.mean, args.std)
            ).softmax(dim=1)[:, target_label]
            per_target.append(target_probs)
        meters["translation_conf"].update(torch.cat(per_target).mean().item(), batch)

    print(f"Split: {args.split} | Images: {len(dataset)}")
    print(f"Same-label reconstruction MSE: {meters['mse'].avg:.6f}")
    print(f"Same-label reconstruction PSNR: {meters['psnr'].avg:.4f}")
    print(f"Same-label reconstruction SSIM: {meters['ssim'].avg:.4f}")
    print(f"Same-label BCE: {meters['bce'].avg:.6f}")
    print(f"Classifier confidence on reconstructions: {meters['same_conf'].avg:.4f}")
    print(f"Classifier accuracy on reconstructions: {correct / max(1, total):.4f}")
    print(f"Mean requested-target confidence across translations: {meters['translation_conf'].avg:.4f}")
    print("Reconstruction confusion matrix:")
    print(format_confusion_matrix(confusion))
    print("Note: Translation confidence is reported without pixel comparison because OASIS-1 has no paired future-stage target image.")


if __name__ == "__main__":
    main()
