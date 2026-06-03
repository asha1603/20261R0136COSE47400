import argparse
import os

import torch
from torch.utils.data import DataLoader

from src.data import MRISubjectDataset
from src.models import ImprovedFiLMGatedCVAE
from src.utils import get_device, load_state_dict_flexible, ensure_dir
from src.visualization import save_stage_grid


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output_path", type=str, default="outputs/final_stage_grid.png")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--class_dim", type=int, default=32)
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--film_scale", type=float, default=0.05)
    p.add_argument("--translation_skip_scale", type=float, default=0.6)
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()
    ds = MRISubjectDataset(args.data_dir, "val", args.image_size, classifier_transform=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    batch = next(iter(loader))

    model = ImprovedFiLMGatedCVAE(
        image_size=args.image_size,
        latent_dim=args.latent_dim,
        class_dim=args.class_dim,
        base_channels=args.base_channels,
        film_scale=args.film_scale,
    ).to(device)
    state, ckpt = load_state_dict_flexible(args.checkpoint, device)
    model.load_state_dict(state, strict=True)
    model.eval()

    ensure_dir(os.path.dirname(args.output_path) or ".")
    save_stage_grid(model, batch, device, args.output_path, args.translation_skip_scale, max_rows=args.batch_size)
    print(f"Saved grid: {args.output_path}")


if __name__ == "__main__":
    main()
