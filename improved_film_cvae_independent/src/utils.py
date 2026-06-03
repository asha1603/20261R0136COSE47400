import os
import random
from typing import Dict

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_checkpoint(path: str, payload: Dict) -> None:
    ensure_dir(os.path.dirname(path))
    torch.save(payload, path)


def load_state_dict_flexible(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict):
        for key in ["model_state_dict", "state_dict", "classifier_state_dict"]:
            if key in ckpt:
                return ckpt[key], ckpt
    return ckpt, ckpt


def sample_different_labels(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    offset = torch.randint(1, num_classes, y.shape, device=y.device)
    return (y + offset) % num_classes


def classifier_normalize(x: torch.Tensor, mean: float = 0.456, std: float = 0.224) -> torch.Tensor:
    return (x - mean) / std
