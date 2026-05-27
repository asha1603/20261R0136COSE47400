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
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_checkpoint(path: str | Path, **payload) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, device: torch.device):
    try:
        return torch.load(Path(path), map_location=device, weights_only=False)
    except TypeError:
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
    return Image.fromarray((array * 255).astype(np.uint8), mode="L")


def save_translation_grid(
    original: torch.Tensor,
    outputs: list[tuple[str, torch.Tensor]],
    path: str | Path,
    title: str = "",
) -> None:
    images = [("Original", original)] + outputs
    pil_images = [(label, tensor_to_pil(tensor)) for label, tensor in images]
    width, height = pil_images[0][1].size
    top = 42
    canvas = Image.new("L", (width * len(pil_images), height + top), color=255)
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 4), title, fill=0)
    for index, (label, image) in enumerate(pil_images):
        x = index * width
        canvas.paste(image, (x, top))
        draw.text((x + 4, 22), label, fill=0)
    ensure_dir(Path(path).parent)
    canvas.save(path)


def write_json(path: str | Path, payload) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
