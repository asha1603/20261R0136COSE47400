import argparse
import os
import zipfile
import subprocess
from pathlib import Path


def setup_kaggle_from_colab_secrets():
    """
    Use Kaggle username/key stored in Google Colab Secrets.

    In Colab, create secrets:
    - KAGGLE_USERNAME
    - KAGGLE_KEY
    """
    try:
        from google.colab import userdata
    except ImportError:
        raise RuntimeError(
            "google.colab is not available. "
            "Use --kaggle_json instead or run this inside Colab."
        )

    username = userdata.get("KAGGLE_USERNAME")
    key = userdata.get("KAGGLE_KEY")

    if username is None or key is None:
        raise RuntimeError(
            "Kaggle secrets not found. Please add KAGGLE_USERNAME and KAGGLE_KEY "
            "to Colab Secrets."
        )

    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)

    kaggle_json = kaggle_dir / "kaggle.json"
    kaggle_json.write_text(
        f'{{"username":"{username}","key":"{key}"}}'
    )

    os.chmod(kaggle_json, 0o600)
    print(f"Kaggle API key saved to {kaggle_json}")


def setup_kaggle_from_json(kaggle_json_path):
    kaggle_json_path = Path(kaggle_json_path)
    if not kaggle_json_path.exists():
        raise FileNotFoundError(f"Cannot find {kaggle_json_path}")

    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)

    target = kaggle_dir / "kaggle.json"
    target.write_bytes(kaggle_json_path.read_bytes())
    os.chmod(target, 0o600)

    print(f"Kaggle API key copied to {target}")


def unzip_all_zip_files(output_dir):
    output_dir = Path(output_dir)
    zip_files = list(output_dir.glob("*.zip"))

    if not zip_files:
        print("No zip file found to extract.")
        return

    for zip_path in zip_files:
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(output_dir)

    print("Extraction complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Kaggle dataset slug, for example: username/dataset-name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Where the dataset should be downloaded/extracted.",
    )
    parser.add_argument(
        "--use_colab_secrets",
        action="store_true",
        help="Use KAGGLE_USERNAME and KAGGLE_KEY from Colab Secrets.",
    )
    parser.add_argument(
        "--kaggle_json",
        type=str,
        default=None,
        help="Path to local kaggle.json, if not using Colab Secrets.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.use_colab_secrets:
        setup_kaggle_from_colab_secrets()
    elif args.kaggle_json is not None:
        setup_kaggle_from_json(args.kaggle_json)
    else:
        print(
            "No Kaggle credential setup selected. "
            "Assuming Kaggle API is already configured."
        )

    command = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        args.dataset,
        "-p",
        args.output_dir,
    ]

    print("Running:", " ".join(command))
    subprocess.run(command, check=True)

    unzip_all_zip_files(args.output_dir)

    print("\nDone.")
    print(f"Dataset downloaded/extracted to: {args.output_dir}")


if __name__ == "__main__":
    main()