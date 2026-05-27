from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# Kept identical to the existing unet dataset-folder convention.
CLASS_TO_INDEX = {
    "Non Demented": 0,
    "Very mild Dementia": 1,
    "Mild Dementia": 2,
}
INDEX_TO_CLASS = {value: key for key, value in CLASS_TO_INDEX.items()}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: int
    subject_id: str


def extract_subject_id(path: str | Path) -> str:
    """Extract an OASIS subject id such as OAS1_0001 from a slice filename."""
    stem = Path(path).stem
    parts = stem.split("_")
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return stem


def discover_records(data_dir: str | Path) -> list[ImageRecord]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Data directory does not exist: {root}")

    records: list[ImageRecord] = []
    for class_name, label in CLASS_TO_INDEX.items():
        class_dir = root / class_name
        if not class_dir.exists():
            continue
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                records.append(ImageRecord(path, label, extract_subject_id(path)))

    if not records:
        expected = ", ".join(CLASS_TO_INDEX)
        raise ValueError(f"No images found under {root}. Expected folders: {expected}")
    return records


def subject_stratified_split(
    records: list[ImageRecord],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    """Prevent slices from the same person appearing in both train and validation."""
    subject_to_records: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        subject_to_records[record.subject_id].append(record)

    subject_to_label: dict[str, int] = {}
    for subject, items in subject_to_records.items():
        subject_to_label[subject] = Counter(item.label for item in items).most_common(1)[0][0]

    label_to_subjects: dict[int, list[str]] = defaultdict(list)
    for subject, label in subject_to_label.items():
        label_to_subjects[label].append(subject)

    rng = random.Random(seed)
    train_subjects: set[str] = set()
    val_subjects: set[str] = set()
    for subjects in label_to_subjects.values():
        subjects = list(subjects)
        rng.shuffle(subjects)
        n_val = int(round(len(subjects) * val_ratio))
        if len(subjects) > 1:
            n_val = max(1, min(n_val, len(subjects) - 1))
        else:
            n_val = 0
        val_subjects.update(subjects[:n_val])
        train_subjects.update(subjects[n_val:])

    return (
        [record for record in records if record.subject_id in train_subjects],
        [record for record in records if record.subject_id in val_subjects],
    )


class MRISliceDataset(Dataset):
    """
    Data loader compatible with your current GitHub unet folder.

    CVAE images must remain in [0, 1] because the final reconstruction loss uses BCE.
    For the frozen classifier only, use normalize=True with its training mean/std.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        image_size: int = 224,
        val_ratio: float = 0.2,
        seed: int = 42,
        normalize: bool = False,
        mean: float = 0.456,
        std: float = 0.224,
    ) -> None:
        if split not in {"train", "val", "all"}:
            raise ValueError("split must be one of: train, val, all")
        all_records = discover_records(data_dir)
        train_records, val_records = subject_stratified_split(all_records, val_ratio, seed)
        self.records = (
            train_records if split == "train"
            else val_records if split == "val"
            else all_records
        )
        self.image_size = image_size
        self.normalize = normalize
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = Image.open(record.path).convert("L")
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).unsqueeze(0)
        if self.normalize:
            tensor = (tensor - self.mean) / self.std
        label = torch.tensor(record.label, dtype=torch.long)
        return tensor, label, record.subject_id, str(record.path)
