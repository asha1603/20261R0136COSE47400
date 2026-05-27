"""
Download and prepare the OASIS Alzheimer's Detection Kaggle dataset for the final FiLM model.

Kaggle dataset:
    https://www.kaggle.com/datasets/ninadaithal/imagesoasis
    slug: ninadaithal/imagesoasis

This script:
1. Downloads the dataset through the Kaggle CLI.
2. Extracts the downloaded archive.
3. Copies only the three classes used by the final methodology:
       Non Demented, Very mild Dementia, Mild Dementia
4. Ignores Moderate Dementia.
5. Applies the slides 100--160 ROI rule when a reliable slice number can be
   parsed from OASIS-style filenames. When filenames do not contain a
   recognizable slice index, images are kept and a warning is printed.
"""

from __future__ import annotations

import argparse
from collections import Counter
import re
import shutil
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

DATASET_SLUG = "ninadaithal/imagesoasis"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

CLASS_ALIASES = {
    "nondemented": "Non Demented",
    "nondementia": "Non Demented",
    "verymilddemented": "Very mild Dementia",
    "verymilddementia": "Very mild Dementia",
    "milddemented": "Mild Dementia",
    "milddementia": "Mild Dementia",
}
IGNORED_CLASS_ALIASES = {"moderatedemented", "moderatedementia"}


def normalized_name(value: str) -> str:
    return re.sub(r"[^a-z]", "", value.lower())


def find_class_name(folder_name: str) -> str | None:
    return CLASS_ALIASES.get(normalized_name(folder_name))


def is_ignored_class(folder_name: str) -> bool:
    return normalized_name(folder_name) in IGNORED_CLASS_ALIASES


def parse_oasis_slice_index(path: Path) -> int | None:
    """
    Read a slice number only from a filename that looks like an MRI slice name.
    This avoids treating an arbitrary image identifier as an axial slice index.
    Common supported endings include: _slice_100, -slice-100, _mpr-1_100.
    """
    stem = path.stem.lower()
    patterns = [
        r"(?:^|[_-])slice[_-]?(\d{2,3})$",
        r"(?:^|[_-])mpr[_-]?\d+[_-](\d{2,3})$",
        r"(?:^|[_-])mpr[_-]?\d+[_-].*[_-](\d{2,3})$",
    ]
    for pattern in patterns:
        match = re.search(pattern, stem)
        if match:
            return int(match.group(1))
    return None


def run_kaggle_download(dataset_slug: str, download_dir: Path, force: bool) -> Path:
    download_dir.mkdir(parents=True, exist_ok=True)
    command = ["kaggle", "datasets", "download", "-d", dataset_slug, "-p", str(download_dir)]
    if force:
        command.append("--force")
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise SystemExit(
            "The Kaggle CLI is not installed. Run: pip install kaggle"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            "Kaggle download failed. Check that your Kaggle API credential is configured "
            "and that you accepted any dataset access conditions."
        ) from exc

    zip_files = sorted(download_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not zip_files:
        raise FileNotFoundError(f"No ZIP file was downloaded into {download_dir}.")
    return zip_files[0]


def find_labeled_images(raw_dir: Path) -> tuple[dict[str, list[Path]], int]:
    selected: dict[str, list[Path]] = {name: [] for name in CLASS_ALIASES.values()}
    moderate_count = 0
    for folder in raw_dir.rglob("*"):
        if not folder.is_dir():
            continue
        final_class = find_class_name(folder.name)
        ignored = is_ignored_class(folder.name)
        if final_class is None and not ignored:
            continue

        images = [
            path for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if ignored:
            moderate_count += len(images)
        else:
            selected[final_class].extend(images)
    return selected, moderate_count


def copy_selected_images(
    selected: dict[str, list[Path]],
    dataset_dir: Path,
    slice_min: int,
    slice_max: int,
    clean: bool,
) -> Counter:
    if clean and dataset_dir.exists():
        for class_name in selected:
            shutil.rmtree(dataset_dir / class_name, ignore_errors=True)

    counters: Counter = Counter()
    for class_name, image_paths in selected.items():
        destination_dir = dataset_dir / class_name
        destination_dir.mkdir(parents=True, exist_ok=True)

        parsed_indices = [parse_oasis_slice_index(path) for path in image_paths]
        can_filter_roi = any(index is not None for index in parsed_indices)
        if not can_filter_roi:
            print(
                f"WARNING: Could not identify slice indices for '{class_name}'. "
                f"Keeping all {len(image_paths)} images for this class."
            )

        for source_path, slice_index in zip(image_paths, parsed_indices):
            if can_filter_roi and slice_index is not None and not (slice_min <= slice_index <= slice_max):
                counters[f"{class_name}: removed outside ROI"] += 1
                continue
            # Preserve uniqueness even if Kaggle has nested folders with repeated image names.
            destination = destination_dir / source_path.name
            if destination.exists():
                destination = destination_dir / f"{source_path.parent.name}_{source_path.name}"
            shutil.copy2(source_path, destination)
            counters[f"{class_name}: copied"] += 1
    return counters


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare Kaggle OASIS images for FiLM model.")
    parser.add_argument("--dataset", default=DATASET_SLUG)
    parser.add_argument("--dataset-dir", default="Dataset")
    parser.add_argument("--download-dir", default=".kaggle_download")
    parser.add_argument("--slice-min", type=int, default=100)
    parser.add_argument("--slice-max", type=int, default=160)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--no-clean", action="store_true", help="Do not remove existing selected-class images.")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    download_dir = Path(args.download_dir)
    raw_dir = download_dir / "raw"

    archive = run_kaggle_download(args.dataset, download_dir, args.force_download)
    shutil.rmtree(raw_dir, ignore_errors=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {archive} ...")
    with ZipFile(archive, "r") as archive_file:
        archive_file.extractall(raw_dir)

    selected, moderate_count = find_labeled_images(raw_dir)
    missing = [name for name, paths in selected.items() if not paths]
    if missing:
        raise SystemExit(
            "Could not find image folders for: " + ", ".join(missing) +
            ". Inspect the extracted archive structure under " + str(raw_dir)
        )

    counters = copy_selected_images(
        selected=selected,
        dataset_dir=dataset_dir,
        slice_min=args.slice_min,
        slice_max=args.slice_max,
        clean=not args.no_clean,
    )

    print("\nDataset preparation complete.")
    print(f"Source Kaggle dataset: {args.dataset}")
    print(f"Prepared dataset directory: {dataset_dir.resolve()}")
    print(f"Moderate Dementia images ignored: {moderate_count}")
    for class_name in ["Non Demented", "Very mild Dementia", "Mild Dementia"]:
        print(f"{class_name}: {counters[f'{class_name}: copied']} prepared images")
        removed = counters[f"{class_name}: removed outside ROI"]
        if removed:
            print(f"  Removed outside slice range {args.slice_min}-{args.slice_max}: {removed}")

    if sum(counters[f"{name}: copied"] for name in selected) == 0:
        raise SystemExit("No images were prepared. Check the slice range or extracted dataset structure.")


if __name__ == "__main__":
    main()
