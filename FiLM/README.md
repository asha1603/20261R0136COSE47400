# Final_FiLM_Model — Standalone Final Model

This is a complete independent final-model folder. It does **not** need to be inside `unet/` and it does not import the previous midterm U-Net code.

## GitHub structure

Upload this folder directly to your repository root:

```text
20261R0136COSE47400/
  unet/                         # previous/midterm model, kept separately
  Final_FiLM_Model/             # final model on its own
    Dataset/
    checkpoints/
    outputs/
    configs/
    scripts/
    src/
    requirements.txt
    README.md
```

## Model implemented

- Classifier-guided disentangled CVAE with FiLM decoder conditioning.
- 224 × 224 grayscale MRI slices and three classes.
- `z_content = 96`, `z_class = 32`, total latent dimension `128`.
- Two U-Net skip connections.
- FiLM modulation at every decoder resolution.
- Frozen ResNet18 classifier guidance.
- BCE + LPIPS reconstruction loss.
- KL annealing, center loss, separation loss, and classifier-guided loss.

Selected loss settings:

```text
lambda_BCE   = 1.0
lambda_LPIPS = 1.0
beta_KL_max  = 2.0
w_center     = 10.0
w_sep        = 5.0
lambda_cls   = 2.0
margin       = 2.0
```

## Kaggle dataset used

This project is prepared for:

```text
OASIS Alzheimer's Detection
Kaggle dataset slug: ninadaithal/imagesoasis
```

The downloaded dataset contains dementia-stage image classes. The final model prepares only:

```text
Non Demented
Very mild Dementia
Mild Dementia
```

`Moderate Dementia` is not used because the final methodology is defined for the three classes above.

The dataset download script also applies the stated axial ROI range of slices `100–160` when the downloaded OASIS filenames contain a recognizable slice index. When a slice number cannot be reliably detected from filenames, it keeps the available class images and prints a warning instead of silently deleting valid data.

## Run in Google Colab

### Step 1: Clone the repository and enter this standalone folder

```python
!git clone https://github.com/asha1603/20261R0136COSE47400.git
%cd /content/20261R0136COSE47400/Final_FiLM_Model
!pip -q install -r requirements.txt
```

### Step 2: Set up Kaggle authentication

#### Option A — Upload `kaggle.json`

In Kaggle, open **Settings → API → Create Legacy API Key**. This downloads `kaggle.json`.

Then run in Colab:

```python
from google.colab import files
files.upload()   # choose kaggle.json
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/kaggle.json
!chmod 600 ~/.kaggle/kaggle.json
```

Do not upload `kaggle.json` to GitHub.

#### Option B — Use a Colab secret with the current Kaggle API token

Create a Colab secret named `KAGGLE_API_TOKEN`, enable notebook access, then run:

```python
from google.colab import userdata
import os
os.environ["KAGGLE_API_TOKEN"] = userdata.get("KAGGLE_API_TOKEN")
```

### Step 3: Download and prepare the dataset

```python
!python scripts/download_dataset_from_kaggle.py \
    --dataset ninadaithal/imagesoasis \
    --dataset-dir Dataset \
    --slice-min 100 \
    --slice-max 160 \
    --force-download
```

After preparation, the model reads the following independent local structure:

```text
Final_FiLM_Model/
  Dataset/
    Non Demented/
    Very mild Dementia/
    Mild Dementia/
```

### Step 4: Verify the architecture

```python
!python scripts/smoke_test_final_film.py
```

### Step 5: Train the ResNet18 guidance classifier

```python
!python scripts/train_resnet18_classifier_final.py \
    --data-dir Dataset \
    --output-dir checkpoints/final_classifier \
    --epochs 20 \
    --batch-size 32
```

### Step 6: Train the final FiLM CVAE

```python
!python scripts/train_final_film_cvae.py \
    --data-dir Dataset \
    --classifier-checkpoint checkpoints/final_classifier/best_resnet18_classifier.pth \
    --output-dir checkpoints/final_film_cvae \
    --epochs 50 \
    --batch-size 8 \
    --beta-kl-max 2.0 \
    --kl-warmup-epochs 10 \
    --lambda-bce 1.0 \
    --lambda-lpips 1.0 \
    --w-center 10.0 \
    --w-sep 5.0 \
    --lambda-cls 2.0 \
    --margin 2.0
```

### Step 7: Generate condition-controlled MRI outputs

Replace `YOUR_IMAGE_FILENAME.jpg` with a real downloaded filename.

```python
!python scripts/generate_final_translation.py \
    --image "Dataset/Non Demented/YOUR_IMAGE_FILENAME.jpg" \
    --cvae-checkpoint checkpoints/final_film_cvae/best_final_film_cvae.pth \
    --output outputs/final_translation_grid.png
```

### Step 8: Evaluate

```python
!python scripts/evaluate_final_film.py \
    --data-dir Dataset \
    --cvae-checkpoint checkpoints/final_film_cvae/best_final_film_cvae.pth \
    --classifier-checkpoint checkpoints/final_classifier/best_resnet18_classifier.pth \
    --split val
```

## Important evaluation note

BCE, PSNR, and SSIM are valid for same-stage reconstruction because the original input slice is the reconstruction target.

For a generated different dementia stage, OASIS-1 does not provide a paired true future-stage MRI of the same individual. Therefore, the evaluation code reports requested-target classifier confidence for translations and does not incorrectly claim pixel-wise target-stage ground-truth comparison.
