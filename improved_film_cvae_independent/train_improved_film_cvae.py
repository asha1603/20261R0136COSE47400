import argparse
import os
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import MRISubjectDataset, CLASSES, NUM_CLASSES
from src.losses import kl_loss, edge_loss, total_variation_loss, low_frequency_structure_loss
from src.models import ImprovedFiLMGatedCVAE, MRIClassifier
from src.utils import (
    get_device,
    set_seed,
    ensure_dir,
    save_checkpoint,
    load_state_dict_flexible,
    sample_different_labels,
    classifier_normalize,
)
from src.visualization import save_stage_grid


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--classifier_path", type=str, default="", help="Optional path to trained classifier checkpoint.")
    p.add_argument("--output_dir", type=str, default="outputs/improved_film_cvae")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--class_dim", type=int, default=32)
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--film_scale", type=float, default=0.05)
    p.add_argument("--translation_skip_scale", type=float, default=0.6)

    p.add_argument("--lambda_l1", type=float, default=8.0)
    p.add_argument("--lambda_bce", type=float, default=0.5)
    p.add_argument("--lambda_edge", type=float, default=2.0)
    p.add_argument("--beta_kl", type=float, default=0.02)
    p.add_argument("--lambda_trans_structure", type=float, default=2.0)
    p.add_argument("--lambda_trans_edge", type=float, default=0.5)
    p.add_argument("--lambda_tv", type=float, default=0.01)
    p.add_argument("--lambda_cls", type=float, default=0.2)
    p.add_argument("--lambda_diversity", type=float, default=0.0)
    p.add_argument("--min_translation_delta", type=float, default=0.01)

    p.add_argument("--recon_warmup_epochs", type=int, default=10)
    p.add_argument("--gentle_translation_epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--save_every", type=int, default=5)
    return p.parse_args()


def load_classifier(path, device):
    if not path:
        print("No classifier_path provided. Training will run without classifier guidance.")
        return None
    if not os.path.exists(path):
        print(f"Classifier checkpoint not found: {path}. Training will run without classifier guidance.")
        return None
    model = MRIClassifier(pretrained=False).to(device)
    state, _ = load_state_dict_flexible(path, device)
    model.load_state_dict(state, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"Loaded frozen classifier: {path}")
    return model


def stage_weights(args, epoch):
    """
    Stage-wise training to reduce artifacts.
    Stage 1: reconstruction only.
    Stage 2: gentle translation.
    Stage 3: full training with moderate classifier guidance.
    """
    if epoch <= args.recon_warmup_epochs:
        return {
            "use_translation": False,
            "lambda_cls": 0.0,
            "translation_skip_scale": 1.0,
        }
    if epoch <= args.gentle_translation_epochs:
        return {
            "use_translation": True,
            "lambda_cls": min(0.1, args.lambda_cls),
            "translation_skip_scale": max(0.7, args.translation_skip_scale),
        }
    return {
        "use_translation": True,
        "lambda_cls": args.lambda_cls,
        "translation_skip_scale": args.translation_skip_scale,
    }


def train_one_epoch(model, classifier, loader, optimizer, device, epoch, args):
    model.train()
    stats = defaultdict(float)
    n_batches = 0
    sw = stage_weights(args, epoch)

    pbar = tqdm(loader, desc=f"Train {epoch:03d}")
    for x, y, *_ in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        recon, mu, logvar, _ = model(x, y, skip_scale=1.0)
        rec_l1 = F.l1_loss(recon, x)
        rec_bce = F.binary_cross_entropy(recon, x)
        rec_edge = edge_loss(recon, x)
        kld = kl_loss(mu, logvar)

        loss = (
            args.lambda_l1 * rec_l1
            + args.lambda_bce * rec_bce
            + args.lambda_edge * rec_edge
            + args.beta_kl * kld
        )

        cls_loss = x.new_tensor(0.0)
        trans_structure = x.new_tensor(0.0)
        trans_edge = x.new_tensor(0.0)
        tv = x.new_tensor(0.0)
        diversity = x.new_tensor(0.0)
        delta = x.new_tensor(0.0)
        target_conf = x.new_tensor(0.0)

        if sw["use_translation"]:
            y_tgt = sample_different_labels(y, NUM_CLASSES)
            translated, mu_t, logvar_t, _ = model(x, y_tgt, skip_scale=sw["translation_skip_scale"])

            trans_structure = low_frequency_structure_loss(translated, x, kernel_size=8)
            trans_edge = edge_loss(translated, x)
            tv = total_variation_loss(translated)
            kld_t = kl_loss(mu_t, logvar_t)
            delta = torch.mean(torch.abs(translated - recon))

            loss = loss + (
                args.lambda_trans_structure * trans_structure
                + args.lambda_trans_edge * trans_edge
                + args.lambda_tv * tv
                + 0.5 * args.beta_kl * kld_t
            )

            if args.lambda_diversity > 0:
                diff_per_sample = torch.mean(torch.abs(translated - recon), dim=(1, 2, 3))
                diversity = torch.relu(args.min_translation_delta - diff_per_sample).mean()
                loss = loss + args.lambda_diversity * diversity

            if classifier is not None and sw["lambda_cls"] > 0:
                logits = classifier(classifier_normalize(translated))
                cls_loss = F.cross_entropy(logits, y_tgt)
                target_conf = torch.softmax(logits, dim=1).gather(1, y_tgt[:, None]).mean()
                loss = loss + sw["lambda_cls"] * cls_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        vals = {
            "loss": loss.item(),
            "rec_l1": rec_l1.item(),
            "rec_bce": rec_bce.item(),
            "rec_edge": rec_edge.item(),
            "kl": kld.item(),
            "cls": cls_loss.item(),
            "target_conf": target_conf.item(),
            "delta": delta.item(),
            "trans_structure": trans_structure.item(),
        }
        for k, v in vals.items():
            stats[k] += v
        n_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.3f}", "l1": f"{rec_l1.item():.4f}", "cls": f"{cls_loss.item():.3f}", "conf": f"{target_conf.item():.3f}", "delta": f"{delta.item():.4f}"})

    return {k: v / max(1, n_batches) for k, v in stats.items()}


@torch.no_grad()
def validate(model, classifier, loader, device, args):
    model.eval()
    stats = defaultdict(float)
    n_batches = 0

    for x, y, *_ in tqdm(loader, desc="Val", leave=False):
        x, y = x.to(device), y.to(device)
        recon, mu, logvar, _ = model(x, y, skip_scale=1.0)
        y_tgt = sample_different_labels(y, NUM_CLASSES)
        translated, _, _, _ = model(x, y_tgt, skip_scale=args.translation_skip_scale)

        rec_l1 = F.l1_loss(recon, x)
        rec_bce = F.binary_cross_entropy(recon, x)
        rec_edge = edge_loss(recon, x)
        kld = kl_loss(mu, logvar)
        trans_structure = low_frequency_structure_loss(translated, x, kernel_size=8)
        trans_edge = edge_loss(translated, x)
        delta = torch.mean(torch.abs(translated - recon))

        cls_loss = x.new_tensor(0.0)
        target_conf = x.new_tensor(0.0)
        target_acc = x.new_tensor(0.0)
        if classifier is not None:
            logits = classifier(classifier_normalize(translated))
            cls_loss = F.cross_entropy(logits, y_tgt)
            probs = torch.softmax(logits, dim=1)
            target_conf = probs.gather(1, y_tgt[:, None]).mean()
            target_acc = (logits.argmax(dim=1) == y_tgt).float().mean()

        # Validation score prefers clear reconstruction and no shortcut-heavy classifier domination.
        val_score = (
            args.lambda_l1 * rec_l1
            + args.lambda_bce * rec_bce
            + args.lambda_edge * rec_edge
            + args.beta_kl * kld
            + args.lambda_trans_structure * trans_structure
            + 0.5 * trans_edge
            + 0.2 * cls_loss
        )

        vals = {
            "val_score": val_score.item(),
            "rec_l1": rec_l1.item(),
            "rec_bce": rec_bce.item(),
            "rec_edge": rec_edge.item(),
            "kl": kld.item(),
            "target_conf": target_conf.item(),
            "target_acc": target_acc.item(),
            "delta": delta.item(),
            "trans_structure": trans_structure.item(),
        }
        for k, v in vals.items():
            stats[k] += v
        n_batches += 1

    return {k: v / max(1, n_batches) for k, v in stats.items()}


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    sample_dir = os.path.join(args.output_dir, "samples")
    ensure_dir(ckpt_dir)
    ensure_dir(sample_dir)

    train_ds = MRISubjectDataset(args.data_dir, "train", args.image_size, classifier_transform=False, seed=args.seed)
    val_ds = MRISubjectDataset(args.data_dir, "val", args.image_size, classifier_transform=False, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    print(f"Device: {device}")
    print(f"Train images: {len(train_ds)} | Val images: {len(val_ds)}")
    print(f"Classes: {CLASSES}")

    model = ImprovedFiLMGatedCVAE(
        image_size=args.image_size,
        latent_dim=args.latent_dim,
        class_dim=args.class_dim,
        base_channels=args.base_channels,
        film_scale=args.film_scale,
    ).to(device)

    classifier = load_classifier(args.classifier_path, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_score = float("inf")
    no_improve = 0
    best_path = os.path.join(ckpt_dir, "best_improved_film_cvae.pth")

    for epoch in range(1, args.epochs + 1):
        print("\n" + "=" * 80)
        print(f"Epoch {epoch}/{args.epochs} | stage={stage_weights(args, epoch)}")
        train_stats = train_one_epoch(model, classifier, train_loader, optimizer, device, epoch, args)
        val_stats = validate(model, classifier, val_loader, device, args)
        scheduler.step()

        print("Train:", {k: round(v, 5) for k, v in train_stats.items()})
        print("Val:  ", {k: round(v, 5) for k, v in val_stats.items()})

        if epoch == 1 or epoch % args.save_every == 0:
            batch = next(iter(val_loader))
            out_path = os.path.join(sample_dir, f"stage_grid_epoch_{epoch:03d}.png")
            save_stage_grid(model, batch, device, out_path, args.translation_skip_scale)
            print(f"Saved sample grid: {out_path}")

        if val_stats["val_score"] < best_score:
            best_score = val_stats["val_score"]
            no_improve = 0
            save_checkpoint(best_path, {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_stats": val_stats,
                "args": vars(args),
                "classes": CLASSES,
            })
            print(f"Saved best checkpoint: {best_path}")
        else:
            no_improve += 1
            print(f"No improvement: {no_improve}/{args.patience}")

        if no_improve >= args.patience:
            print("Early stopping.")
            break

    print(f"Training finished. Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
