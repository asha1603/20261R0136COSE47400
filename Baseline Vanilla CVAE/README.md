# Deep Learning Mid: Classifier-Guided U-Net CVAE

This repository implements the model described by `deep learning mid.pdf` as a reproducible project.

The target is the mid-presentation version, not the later final-paper version:

- U-Net style Conditional VAE for MRI slice reconstruction.
- Same-label reconstruction only: input MRI is reconstructed with its own class label.
- Frozen classifier-guided loss to improve disease-stage consistency.
- No FiLM conditioning.
- No content/class latent split.
- No latent center loss or separation loss.
- No target-stage translation helper.

## Project Structure

```text
deep_learning_mid_unet_cvae/
  configs/default.yaml
  scripts/
    train_classifier.py
    train_cvae.py
    evaluate.py
    reconstruct.py
    smoke_test.py
  src/mid_unet_cvae/
    data.py
    losses.py
    metrics.py
    models.py
    utils.py
```

## Setup

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you need a CUDA-specific PyTorch build, install PyTorch from the official selector first, then install the remaining requirements.

## Data Format

Put MRI slice images in class-name folders:

```text
data/
  Non Demented/
    OAS1_0001_slice_001.jpg
  Very mild Dementia/
    OAS1_0002_slice_001.jpg
  Mild Dementia/
    OAS1_0003_slice_001.jpg
```

Supported extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`.

The train/validation split is subject-based. By default, the subject id is extracted from the first two underscore-separated filename parts, e.g. `OAS1_0001_slice_001.jpg` becomes `OAS1_0001`.

## 1. Smoke Test

Run this first on any computer. It does not require data.

```bash
python scripts/smoke_test.py
```

## 2. Train the Frozen Classifier

```bash
python scripts/train_classifier.py --data-dir data --output-dir checkpoints/classifier
```

This saves the best classifier checkpoint to:

```text
checkpoints/classifier/best_classifier.pth
```

## 3. Train the Mid U-Net CVAE

```bash
python scripts/train_cvae.py `
  --data-dir data `
  --classifier-checkpoint checkpoints/classifier/best_classifier.pth `
  --output-dir checkpoints/cvae
```

On macOS/Linux:

```bash
python scripts/train_cvae.py \
  --data-dir data \
  --classifier-checkpoint checkpoints/classifier/best_classifier.pth \
  --output-dir checkpoints/cvae
```

The saved model is:

```text
checkpoints/cvae/best_cvae.pth
```

## 4. Evaluate

```bash
python scripts/evaluate.py \
  --data-dir data \
  --cvae-checkpoint checkpoints/cvae/best_cvae.pth \
  --classifier-checkpoint checkpoints/classifier/best_classifier.pth
```

Reported metrics:

- Reconstruction MSE
- PSNR
- SSIM
- Classifier target confidence on reconstructions
- Classifier accuracy on reconstructions
- Confusion matrix

## 5. Save Reconstruction Samples

```bash
python scripts/reconstruct.py \
  --data-dir data \
  --cvae-checkpoint checkpoints/cvae/best_cvae.pth \
  --output-dir outputs/reconstructions \
  --samples-per-stage 5
```

Each output image shows the original MRI next to the same-label reconstruction.
Samples are selected randomly and evenly by stage instead of taking the first
validation images in file order. With the default setting, the script saves:

```text
5 Non Demented samples
5 Very mild Dementia samples
5 Mild Dementia samples
```

The sampler uses `--seed` for reproducibility and chooses different subjects
within each stage when possible. If a stage has fewer available subjects or
images than requested, the script saves as many as it can and prints a warning.

## Model Definition

The CVAE encodes a grayscale MRI slice into a single latent vector `z`. The class label is embedded once and concatenated with `z` at the bottleneck before decoding. Skip connections preserve spatial anatomy.

Training loss:

```text
L = L_reconstruction + beta * L_KL + lambda_cls * L_classifier
```

This matches the mid-presentation scope: reconstruction quality plus classifier-guided stage consistency.
