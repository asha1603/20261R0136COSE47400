#!/usr/bin/env python3
"""
Create a Team-I-style OASIS subset from the Kaggle ImageOASIS dataset.

Target setup:
- 41 subjects total
- ROI slices 100–160
- subject-level train/validation split
- balanced subject selection across classes as much as possible
- default: 3 classes to match the uploaded ResNet18 classifier checkpoint:
    Non Demented, Very mild Dementia, Mild Dementia

Example:
python make_team_i_style_dataset.py \
  --source_dir /content/imagesoasis \
  --output_dir /content/OASIS_TeamI_Style \
  --num_subjects 41 \
  --slice_min 100 \
  --slice_max 160 \
  --classes "Non Demented" "Very mild Dementia" "Mild Dementia"
"""

import argparse
import os
import random
import re
import shutil
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


CLASS_ALIASES = {
    # Non Demented
    "Non Demented": "Non Demented",
    "NonDemented": "Non Demented",
    "Non_Demented": "Non Demented",
    "Non-Demented": "Non Demented",
    "Nondemented": "Non Demented",

    # Very mild Dementia
    "Very mild Dementia": "Very mild Dementia",
    "Very Mild Dementia": "Very mild Dementia",
    "Very_Mild_Dementia": "Very mild Dementia",
    "VeryMildDemented": "Very mild Dementia",
    "Very Mild Demented": "Very mild Dementia",
    "VeryMildDementia": "Very mild Dementia",

    # Mild Dementia
    "Mild Dementia": "Mild Dementia",
    "Mild_Dementia": "Mild Dementia",
    "MildDemented": "Mild Dementia",
    "Mild Demented": "Mild Dementia",

    # Moderate Dementia
    "Moderate Dementia": "Moderate Dementia",
    "Moderate_Dementia": "Moderate Dementia",
    "ModerateDemented": "Moderate Dementia",
    "Moderate Demented": "Moderate Dementia",
}


def normalize_class_name(name: str) -> str:
    return CLASS_ALIASES.get(name, name)


def extract_subject_id(filename: str) -> str:
    """
    Extract OASIS subject ID from filename.

    Expected examples:
      OAS1_0001_MR1_mpr_n4_anon_111.jpg -> OAS1_0001
      OAS1-0001-slice-111.png -> OAS1_0001

    Fallback:
      remove the final numeric component, assumed to be slice index.
    """
    stem = Path(filename).stem

    match = re.search(r"(OAS1[_-]\d+)", stem, flags=re.IGNORECASE)
    if match:
        return match.group(1).replace("-", "_").upper()

    nums = re.findall(r"\d+", stem)
    parts = stem.split("_")

    if len(parts) > 1 and nums:
        # Remove last number-like slice part if possible.
        last_num = nums[-1]
        cleaned = re.sub(rf"[_-]?{last_num}$", "", stem)
        return cleaned

    return stem


def extract_slice_index(filename: str):
    """
    Extract slice number from filename.
    Assumption: slice index is the last number in the filename.
    """
    stem = Path(filename).stem
    nums = re.findall(r"\d+", stem)
    if not nums:
        return None
    return int(nums[-1])


def detect_class_from_path(path: Path, valid_classes: set):
    """
    Find class from any parent folder name.
    """
    for parent in path.parents:
        cname = normalize_class_name(parent.name)
        if cname in valid_classes:
            return cname
    return None


def find_images(source_dir: Path, valid_classes: list[str]) -> pd.DataFrame:
    valid_class_set = set(valid_classes)
    rows = []

    for path in source_dir.rglob("*"):
        if path.suffix.lower() not in IMAGE_EXTS:
            continue

        class_name = detect_class_from_path(path, valid_class_set)
        if class_name is None:
            continue

        slice_idx = extract_slice_index(path.name)
        if slice_idx is None:
            continue

        subject_id = extract_subject_id(path.name)

        rows.append({
            "image_path": str(path),
            "filename": path.name,
            "class_name": class_name,
            "subject_id": subject_id,
            "slice_index": slice_idx,
        })

    return pd.DataFrame(rows)


def choose_balanced_subjects(df: pd.DataFrame, num_subjects: int, seed: int) -> list[str]:
    """
    Choose subjects fairly across classes.

    Because each OASIS subject should belong to one dementia class, we assign each subject
    to the class that appears most frequently for that subject.

    For 3 classes and 41 subjects, this aims for 14 / 14 / 13 subjects.
    If a class has fewer available subjects, it takes all available and fills the rest from other classes.
    """
    rng = random.Random(seed)

    subj_class = (
        df.groupby("subject_id")["class_name"]
        .agg(lambda x: Counter(x).most_common(1)[0][0])
        .reset_index()
    )

    subj_counts = (
        df.groupby("subject_id")
        .size()
        .rename("num_images")
        .reset_index()
    )

    subj_info = subj_class.merge(subj_counts, on="subject_id", how="left")

    classes = sorted(subj_info["class_name"].unique().tolist())
    n_classes = len(classes)

    base = num_subjects // n_classes
    remainder = num_subjects % n_classes

    # Distribute remainder to classes with fewer subjects first, then alphabetically.
    class_available = {
        c: int((subj_info["class_name"] == c).sum())
        for c in classes
    }
    class_order = sorted(classes, key=lambda c: (class_available[c], c))

    targets = {c: base for c in classes}
    for c in class_order[:remainder]:
        targets[c] += 1

    selected = []
    leftovers = []

    for c in classes:
        candidates = subj_info[subj_info["class_name"] == c].copy()
        # Prefer subjects with complete/more ROI slices, but shuffle ties.
        candidates["rand"] = [rng.random() for _ in range(len(candidates))]
        candidates = candidates.sort_values(["num_images", "rand"], ascending=[False, True])

        take = min(targets[c], len(candidates))
        chosen = candidates.head(take)["subject_id"].tolist()
        selected.extend(chosen)

        remaining = candidates.iloc[take:]["subject_id"].tolist()
        leftovers.extend(remaining)

    # Fill if one class did not have enough subjects.
    if len(selected) < num_subjects:
        rng.shuffle(leftovers)
        for sid in leftovers:
            if sid not in selected:
                selected.append(sid)
            if len(selected) == num_subjects:
                break

    return selected[:num_subjects]


def subject_level_split(df: pd.DataFrame, val_ratio: float, seed: int) -> pd.DataFrame:
    """
    Subject-level split, approximately stratified by class.

    All slices from one subject go either to train or val.
    """
    rng = random.Random(seed)

    subj_class = (
        df.groupby("subject_id")["class_name"]
        .agg(lambda x: Counter(x).most_common(1)[0][0])
        .reset_index()
    )

    val_subjects = set()
    train_subjects = set()

    for class_name, group in subj_class.groupby("class_name"):
        subjects = group["subject_id"].tolist()
        rng.shuffle(subjects)

        n_val = max(1, round(len(subjects) * val_ratio)) if len(subjects) > 1 else 0
        class_val = set(subjects[:n_val])
        class_train = set(subjects[n_val:])

        val_subjects |= class_val
        train_subjects |= class_train

    df = df.copy()
    df["split"] = df["subject_id"].apply(lambda sid: "val" if sid in val_subjects else "train")

    overlap = set(df[df["split"] == "train"]["subject_id"]) & set(df[df["split"] == "val"]["subject_id"])
    if overlap:
        raise RuntimeError(f"Data leakage detected! Subjects in both train and val: {sorted(overlap)[:5]}")

    return df


def copy_dataset(df: pd.DataFrame, output_dir: Path):
    """
    Copy into:
      output_dir/all/<class>/*.jpg
      output_dir/train/<class>/*.jpg
      output_dir/val/<class>/*.jpg
    """
    for subset_name in ["all", "train", "val"]:
        subset_dir = output_dir / subset_name
        if subset_dir.exists():
            shutil.rmtree(subset_dir)

    for _, row in df.iterrows():
        src = Path(row["image_path"])
        class_name = row["class_name"]
        subject_id = row["subject_id"]
        slice_idx = int(row["slice_index"])
        split = row["split"]

        clean_name = f"{subject_id}_slice_{slice_idx:03d}{src.suffix.lower()}"

        # all folder
        dst_all = output_dir / "all" / class_name / clean_name
        dst_all.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_all)

        # train/val folder
        dst_split = output_dir / split / class_name / clean_name
        dst_split.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_split)


def print_summary(df: pd.DataFrame, title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print("Images:", len(df))
    print("Subjects:", df["subject_id"].nunique())
    print("\nImages per class:")
    print(df["class_name"].value_counts())
    print("\nSubjects per class:")
    subj_class = df.groupby("subject_id")["class_name"].agg(lambda x: Counter(x).most_common(1)[0][0])
    print(subj_class.value_counts())

    if "split" in df.columns:
        print("\nImages by split/class:")
        print(pd.crosstab(df["split"], df["class_name"]))
        print("\nSubjects by split/class:")
        tmp = df.drop_duplicates("subject_id")[["subject_id", "class_name", "split"]]
        print(pd.crosstab(tmp["split"], tmp["class_name"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_subjects", type=int, default=41)
    parser.add_argument("--slice_min", type=int, default=100)
    parser.add_argument("--slice_max", type=int, default=160)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["Non Demented", "Very mild Dementia", "Mild Dementia"],
        help="Use 3 classes by default to match the uploaded 3-class classifier checkpoint."
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)

    if not source_dir.exists():
        raise FileNotFoundError(f"source_dir does not exist: {source_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Source:", source_dir)
    print("Output:", output_dir)
    print("Classes:", args.classes)
    print(f"ROI slices: {args.slice_min}–{args.slice_max}")
    print(f"Target subjects: {args.num_subjects}")

    df = find_images(source_dir, args.classes)
    if df.empty:
        raise RuntimeError(
            "No images found. Check source_dir and class folder names. "
            "The script expects class names in parent folders."
        )

    print_summary(df, "Raw detected data")

    # ROI slice filtering
    df_roi = df[df["slice_index"].between(args.slice_min, args.slice_max)].copy()
    if df_roi.empty:
        raise RuntimeError(
            "No images left after slice filtering. "
            "Maybe the Kaggle dataset is already cropped and filenames do not contain slice indices."
        )

    print_summary(df_roi, f"After ROI filtering: slices {args.slice_min}–{args.slice_max}")

    # Select 41 fair subjects
    selected_subjects = choose_balanced_subjects(df_roi, args.num_subjects, args.seed)
    df_subset = df_roi[df_roi["subject_id"].isin(selected_subjects)].copy()

    if df_subset["subject_id"].nunique() < args.num_subjects:
        print(
            f"WARNING: Requested {args.num_subjects} subjects, but only selected "
            f"{df_subset['subject_id'].nunique()} subjects."
        )

    # Subject-level train/val split
    df_subset = subject_level_split(df_subset, args.val_ratio, args.seed)

    print_summary(df_subset, "Final Team-I-style subset")

    # Save metadata
    df_subset = df_subset.sort_values(["split", "class_name", "subject_id", "slice_index"]).reset_index(drop=True)
    df_subset.to_csv(output_dir / "Deep learning metadata.csv", index=False)

    with open(output_dir / "Deep learning selected subjects.txt", "w", encoding="utf-8") as f:
        for sid in sorted(selected_subjects):
            f.write(sid + "\n")

    train_subjects = sorted(df_subset[df_subset["split"] == "train"]["subject_id"].unique().tolist())
    val_subjects = sorted(df_subset[df_subset["split"] == "val"]["subject_id"].unique().tolist())

    with open(output_dir / "Deep learning train subjects.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(train_subjects) + "\n")

    with open(output_dir / "Deep learning val subjects.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(val_subjects) + "\n")

    # Copy image folders
    copy_dataset(df_subset, output_dir)

    print("\nSaved:")
    print(output_dir / "Deep learning metadata.csv")
    print(output_dir / "Deep learning selected subjects.txt")
    print(output_dir / "Deep learning train subjects.txt")
    print(output_dir / "Deep learning val subjects.txt")
    print(output_dir / "all")
    print(output_dir / "train")
    print(output_dir / "val")

    print("\nLeakage check:")
    overlap = set(train_subjects) & set(val_subjects)
    print("Train/val subject overlap:", overlap)
    if not overlap:
        print("OK: subject-level split has no leakage.")


if __name__ == "__main__":
    main()
