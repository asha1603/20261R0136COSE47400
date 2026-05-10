from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer_cuda: bool = True) -> torch.device:
    return torch.device("cuda" if prefer_cuda and torch.cuda.is_available() else "cpu")


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def save_checkpoint(path: str | Path, **payload) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, device: torch.device):
    return torch.load(Path(path), map_location=device)


class AverageMeter:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    array = image.detach().cpu().squeeze(0).clamp(0, 1).numpy()
    array = (array * 255).astype(np.uint8)
    return Image.fromarray(array, mode="L")


def save_reconstruction_pair(
    original: torch.Tensor,
    reconstruction: torch.Tensor,
    path: str | Path,
    title: str = "",
) -> None:
    original_img = tensor_to_pil(original)
    recon_img = tensor_to_pil(reconstruction)
    width, height = original_img.size
    canvas = Image.new("L", (width * 2, height + 24), color=255)
    canvas.paste(original_img, (0, 24))
    canvas.paste(recon_img, (width, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 4), f"Original | Reconstruction {title}", fill=0)
    ensure_dir(Path(path).parent)
    canvas.save(path)


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
