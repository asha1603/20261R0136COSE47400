from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mid_unet_cvae.data import CLASS_TO_INDEX, INDEX_TO_CLASS, MRISliceDataset
from mid_unet_cvae.models import MidUNetCVAE
from mid_unet_cvae.utils import ensure_dir, get_device, load_checkpoint, save_reconstruction_pair, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save same-label reconstruction samples.")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--cvae-checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/reconstructions")
    parser.add_argument("--split", type=str, choices=["train", "val", "all"], default="val")
    parser.add_argument("--samples-per-stage", type=int, default=5)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
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


def select_balanced_indices(
    dataset: MRISliceDataset,
    samples_per_stage: int,
    seed: int,
) -> list[int]:
    rng = random.Random(seed)
    by_label_subject: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))

    for index, record in enumerate(dataset.records):
        by_label_subject[record.label][record.subject_id].append(index)

    selected: list[int] = []
    for label in sorted(INDEX_TO_CLASS):
        subject_to_indices = by_label_subject.get(label, {})
        subjects = list(subject_to_indices)
        rng.shuffle(subjects)

        label_selected: list[int] = []
        for subject in subjects[:samples_per_stage]:
            label_selected.append(rng.choice(subject_to_indices[subject]))

        if len(label_selected) < samples_per_stage:
            already_selected = set(label_selected)
            remaining = [
                index
                for indices in subject_to_indices.values()
                for index in indices
                if index not in already_selected
            ]
            rng.shuffle(remaining)
            label_selected.extend(remaining[: samples_per_stage - len(label_selected)])

        if len(label_selected) < samples_per_stage:
            print(
                f"Warning: requested {samples_per_stage} samples for "
                f"{INDEX_TO_CLASS[label]}, but only found {len(label_selected)}."
            )

        rng.shuffle(label_selected)
        selected.extend(label_selected[:samples_per_stage])

    return selected


@torch.no_grad()
def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(prefer_cuda=not args.cpu)
    output_dir = ensure_dir(args.output_dir)
    model = load_cvae(args.cvae_checkpoint, device)
    dataset = MRISliceDataset(
        args.data_dir,
        split=args.split,
        image_size=model.image_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        normalize=False,
    )
    selected_indices = select_balanced_indices(
        dataset=dataset,
        samples_per_stage=args.samples_per_stage,
        seed=args.seed,
    )
    selected_counts = Counter(dataset.records[index].label for index in selected_indices)
    print("Selected reconstruction samples:")
    for label in sorted(INDEX_TO_CLASS):
        print(f"  {INDEX_TO_CLASS[label]}: {selected_counts.get(label, 0)}")

    saved = 0
    for dataset_index in tqdm(selected_indices, desc="reconstruct"):
        image, label, subject, _ = dataset[dataset_index]
        images = image.unsqueeze(0).to(device)
        labels = label.unsqueeze(0).to(device)
        recon = model(images, labels)["recon"]
        class_name = INDEX_TO_CLASS[int(label.item())].replace(" ", "_")
        filename = f"{saved:04d}_{subject}_{class_name}.png"
        save_reconstruction_pair(
            original=image.cpu(),
            reconstruction=recon[0].cpu(),
            path=output_dir / filename,
            title=f"| {INDEX_TO_CLASS[int(label.item())]}",
        )
        saved += 1

    print(f"Saved {saved} reconstruction samples to {output_dir}")


if __name__ == "__main__":
    main()
