# Improved FiLM-Gated CVAE for MRI Stage-Conditioned Generation

This folder is a **standalone improved FiLM-CVAE project**. You do **not** need to run your old FiLM folder first.

It is designed to fix the artifact problem where every target-conditioned output receives the same stripe/checkerboard distortion. The main fixes are:

1. **Bilinear upsampling instead of transposed convolution** to reduce checkerboard artifacts.
2. **Gentle FiLM conditioning** using a small FiLM scale instead of strong feature modulation.
3. **Target-conditioned gated skip connections** so the decoder cannot simply copy the input.
4. **Stage-wise training** so the model first learns reconstruction, then gradually learns target conditioning.
5. **Lower classifier-guided loss** to reduce shortcut learning.
6. **Structure-preserving translation loss** to keep MRI anatomy stable.
7. **Optional Grad-CAM script** to visualize what the classifier focuses on.

---

## Folder structure

```text
improved_film_cvae_independent/
├── README.md
├── requirements.txt
├── config.yaml
├── smoke_test.py
├── train_classifier.py
├── train_improved_film_cvae.py
├── generate_stage_grid.py
├── generate_gradcam.py
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── models.py
│   ├── losses.py
│   ├── metrics.py
│   ├── utils.py
│   ├── visualization.py
│   └── gradcam.py
└── scripts/
    └── push_to_github.md
```

---

## Download Dataset from Kaggle

This project expects the dataset to be arranged like this:

```text
data/
├── Non Demented/
├── Very mild Dementia/
└── Mild Dementia/
```
---

## Expected dataset structure

Put your MRI dataset like this:

```text
data/
├── Non Demented/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Very mild Dementia/
│   ├── image1.jpg
│   └── ...
└── Mild Dementia/
    ├── image1.jpg
    └── ...
```

The class names must match these folder names exactly:

```text
Non Demented
Very mild Dementia
Mild Dementia
```

If your folder names are different, edit `src/data.py` and change `CLASSES`.

---

## Step 1: Install requirements

```bash
pip install -r requirements.txt
```

For Google Colab, run:

```bash
!pip install timm scikit-learn matplotlib tqdm pyyaml pillow
```

Torch and torchvision are usually already installed in Colab.

---

## Step 2: Test the code before training

Run:

```bash
python smoke_test.py
```

Expected output:

```text
Smoke test passed.
```

This only checks that the model forward pass works. It does not train on your dataset.

---

## Step 3: Train the classifier

The improved CVAE can use a frozen classifier for target-stage guidance. Train the classifier first:

```bash
python train_classifier.py \
  --data_dir data \
  --output_dir outputs/classifier \
  --epochs 15 \
  --batch_size 32
```

For Colab T4, use:

```bash
python train_classifier.py \
  --data_dir /content/data \
  --output_dir /content/outputs/classifier \
  --epochs 15 \
  --batch_size 32
```

The best classifier checkpoint will be saved at:

```text
outputs/classifier/best_classifier.pth
```

---

## Step 4: Train the improved FiLM-CVAE

Run:

```bash
python train_improved_film_cvae.py \
  --data_dir data \
  --classifier_path outputs/classifier/best_classifier.pth \
  --output_dir outputs/improved_film_cvae \
  --epochs 50 \
  --batch_size 16
```

For Colab T4, recommended:

```bash
python train_improved_film_cvae.py \
  --data_dir /content/data \
  --classifier_path /content/outputs/classifier/best_classifier.pth \
  --output_dir /content/outputs/improved_film_cvae \
  --epochs 50 \
  --batch_size 8 \
  --num_workers 2
```

The best CVAE checkpoint will be saved at:

```text
outputs/improved_film_cvae/checkpoints/best_improved_film_cvae.pth
```

Sample generation grids will be saved at:

```text
outputs/improved_film_cvae/samples/
```

---

## Step 5: Generate target-stage grid after training

```bash
python generate_stage_grid.py \
  --data_dir data \
  --checkpoint outputs/improved_film_cvae/checkpoints/best_improved_film_cvae.pth \
  --output_path outputs/improved_film_cvae/final_stage_grid.png
```

This creates a grid like:

```text
Original | Non Demented | Very mild Dementia | Mild Dementia
```

---

## Step 6: Generate Grad-CAM visualization

Grad-CAM uses the trained classifier, not the CVAE directly.

```bash
python generate_gradcam.py \
  --data_dir data \
  --classifier_path outputs/classifier/best_classifier.pth \
  --output_dir outputs/gradcam \
  --num_images 8
```

This helps you inspect whether the classifier is focusing on brain regions or artifacts/background.

---

## Important hyperparameters for artifact reduction

The default settings are intentionally conservative:

```text
film_scale = 0.05
translation_skip_scale = 0.6
lambda_cls = 0.2
lambda_diversity = 0.0
beta_kl = 0.02
lambda_trans_structure = 2.0
```

These are chosen to reduce the repeated stripe/checkerboard artifact.

If artifacts still appear, try:

```bash
python train_improved_film_cvae.py \
  --data_dir data \
  --classifier_path outputs/classifier/best_classifier.pth \
  --lambda_cls 0.1 \
  --film_scale 0.03 \
  --translation_skip_scale 0.7
```

If all target outputs look too similar, try:

```bash
python train_improved_film_cvae.py \
  --data_dir data \
  --classifier_path outputs/classifier/best_classifier.pth \
  --lambda_cls 0.3 \
  --translation_skip_scale 0.55
```

---

## How this version is different from the previous FiLM code

The previous FiLM-CVAE likely learned strong reconstruction, but target-conditioned outputs showed the same artifact. This version changes the training strategy:

### Stage 1: Reconstruction warm-up
The model first learns stable MRI reconstruction. Classifier guidance is disabled.

### Stage 2: Gentle translation
The model begins target conditioning, but classifier loss remains weak.

### Stage 3: Full training
The model balances reconstruction, structure preservation, and moderate classifier guidance.

This reduces the chance that the generator learns a fake texture shortcut.

---

## What to monitor during training

Good signs:

- Reconstruction image is clear.
- Stripe/checkerboard artifact decreases.
- Target images are not identical, but also not heavily distorted.
- Classifier confidence improves without obvious artificial texture.

Bad signs:

- Same stripe appears for every target class.
- Classifier confidence is high but image looks fake.
- Target outputs differ only by noise or texture.
- Brain anatomy changes too much.

---

## Suggested report explanation

You can write:

> We improved the FiLM-CVAE by replacing artifact-prone decoder upsampling with bilinear upsampling, applying weaker FiLM modulation, and using target-conditioned gated skip connections. We also used a stage-wise training strategy where the model first learns stable reconstruction before classifier-guided target translation. This was done to reduce shortcut learning, where the generator previously produced similar distortion artifacts for all target dementia stages.

---

## GitHub push summary

Copy this folder into your repo, for example:

```text
20261R0136COSE47400/FiLM/improved_film_cvae_independent/
```

Then push:

```bash
git add FiLM/improved_film_cvae_independent
git commit -m "Add independent improved FiLM CVAE implementation"
git push origin main
```
