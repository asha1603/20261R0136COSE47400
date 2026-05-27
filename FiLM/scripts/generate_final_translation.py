from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from final_film_cvae.data import INDEX_TO_CLASS
from final_film_cvae.models import FinalFiLMDisentangledCVAE
from final_film_cvae.utils import get_device, load_checkpoint, save_translation_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate target-stage MRI images with final FiLM CVAE.")
    parser.add_argument("--image", required=True, help="Path to a source MRI slice.")
    parser.add_argument("--cvae-checkpoint", required=True)
    parser.add_argument("--output", default="outputs/final_translation_grid.png")
    parser.add_argument("--target-label", type=int, choices=[0, 1, 2], default=None,
                        help="Generate one class only. Omit to generate all three classes.")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def load_image(path: str, image_size: int) -> torch.Tensor:
    image = Image.open(path).convert("L").resize((image_size, image_size), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0).unsqueeze(0)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = get_device(prefer_cuda=not args.cpu)
    checkpoint = load_checkpoint(args.cvae_checkpoint, device)
    model = FinalFiLMDisentangledCVAE(
        image_size=int(checkpoint.get("image_size", 224)),
        latent_dim=int(checkpoint.get("latent_dim", 128)),
        content_dim=int(checkpoint.get("content_dim", 96)),
        class_dim=int(checkpoint.get("class_dim", 32)),
        num_classes=int(checkpoint.get("num_classes", 3)),
        class_embed_dim=int(checkpoint.get("class_embed_dim", 32)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    source = load_image(args.image, model.image_size).to(device)
    target_labels = [args.target_label] if args.target_label is not None else [0, 1, 2]
    outputs = []
    for label in target_labels:
        label_tensor = torch.tensor([label], dtype=torch.long, device=device)
        generated = model.translate(source, label_tensor, deterministic=True)[0]
        outputs.append((INDEX_TO_CLASS[label], generated))

    save_translation_grid(
        source[0], outputs, args.output,
        title="Source anatomy conditioned into requested dementia stage(s)",
    )
    print(f"Saved generated image comparison to: {args.output}")


if __name__ == "__main__":
    main()
