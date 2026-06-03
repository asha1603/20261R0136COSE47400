import os
import random
from collections import defaultdict
from glob import glob
from typing import Dict, List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

CLASSES: Dict[int, str] = {
    0: "Non Demented",
    1: "Very mild Dementia",
    2: "Mild Dementia",
}
CLASS_TO_IDX: Dict[str, int] = {v: k for k, v in CLASSES.items()}
NUM_CLASSES = len(CLASSES)


def default_transform(image_size: int = 224, classifier: bool = False):
    ops = [
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ]
    if classifier:
        ops.append(transforms.Normalize(mean=[0.456], std=[0.224]))
    return transforms.Compose(ops)


class MRISubjectDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        image_size: int = 224,
        val_ratio: float = 0.2,
        seed: int = 42,
        classifier_transform: bool = False,
        extensions: Tuple[str, ...] = ("*.jpg", "*.jpeg", "*.png", "*.bmp"),
    ):
        assert split in {"train", "val", "all"}
        self.root_dir = root_dir
        self.split = split
        self.transform = default_transform(image_size, classifier=classifier_transform)
        self.items: List[Tuple[str, str, int]] = []

        subject_to_items = defaultdict(list)
        for class_name, label in CLASS_TO_IDX.items():
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                print(f"Warning: missing class folder: {class_dir}")
                continue

            image_paths = []
            for ext in extensions:
                image_paths.extend(glob(os.path.join(class_dir, ext)))

            for path in image_paths:
                subject_id = self.extract_subject_id(os.path.basename(path))
                subject_to_items[subject_id].append((path, label))

        if not subject_to_items:
            raise RuntimeError(
                f"No images found in {root_dir}. Expected class folders: {list(CLASS_TO_IDX.keys())}"
            )

        rng = random.Random(seed)
        class_to_subjects = defaultdict(list)
        for subject_id, vals in subject_to_items.items():
            labels = [label for _, label in vals]
            majority_label = max(set(labels), key=labels.count)
            class_to_subjects[majority_label].append(subject_id)

        self.train_subjects, self.val_subjects = [], []
        for _, subjects in class_to_subjects.items():
            subjects = sorted(subjects)
            rng.shuffle(subjects)
            n_val = max(1, int(len(subjects) * val_ratio)) if len(subjects) > 1 else 0
            self.val_subjects.extend(subjects[:n_val])
            self.train_subjects.extend(subjects[n_val:])

        if split == "train":
            selected = set(self.train_subjects)
        elif split == "val":
            selected = set(self.val_subjects)
        else:
            selected = set(subject_to_items.keys())

        for subject_id in selected:
            for path, label in subject_to_items[subject_id]:
                self.items.append((subject_id, path, label))

        if not self.items:
            raise RuntimeError(f"No items in split={split}. Check dataset size and folder names.")

    @staticmethod
    def extract_subject_id(filename: str) -> str:
        stem = os.path.splitext(filename)[0]
        parts = stem.split("_")
        if len(parts) >= 2:
            return "_".join(parts[:2])
        return stem

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        subject_id, path, label = self.items[idx]
        img = Image.open(path).convert("L")
        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long), subject_id, path
