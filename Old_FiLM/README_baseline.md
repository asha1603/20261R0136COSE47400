# Vanilla CVAE Baseline — Old_FiLM

This README explains how to run the **Vanilla CVAE baseline / ablation model** in the `Old_FiLM` folder using the dataset created by `Deep learning dataset updated.py`.

> Note: In this project, `vanilla_cvae_no_disentanglement.py` is not a completely plain CVAE. It is a CVAE + FiLM ablation model trained **without disentanglement loss**. It is used as a baseline comparison against the full FiLM-CVAE model.

---

## 1. Files Used

The baseline run mainly uses these files:

```text
Old_FiLM/
├── vanilla_cvae_no_disentanglement.py
├── Deep learning dataset updated.py
├── best_classifier_resnet18_weights_42.pth
└── ...
```

### File descriptions

| File | Purpose |
|---|---|
| `Deep learning dataset updated.py` | Creates the 41-subject OASIS subset. |
| `vanilla_cvae_no_disentanglement.py` | Trains and evaluates the baseline model. |
| `best_classifier_resnet18_weights_42.pth` | Pretrained 3-class classifier checkpoint used for classifier-guided loss. |

---

## 2. Dataset Setting

The dataset script creates a Team-I-style OASIS subset with:

```text
Dataset: ImageOASIS / OASIS-1 MRI slices
Number of subjects: 41
Classes: Non Demented, Very mild Dementia, Mild Dementia
ROI slices: 100–160
Split: subject-level train/validation split
```

The dataset script creates this structure:

```text
/content/deep_learning_41_subject_dataset/
├── all/
│   ├── Non Demented/
│   ├── Very mild Dementia/
│   └── Mild Dementia/
├── train/
├── val/
├── Deep learning metadata.csv
├── Deep learning selected subjects.txt
├── Deep learning train subjects.txt
└── Deep learning val subjects.txt
```

For the baseline training, use:

```text
/content/deep_learning_41_subject_dataset/all
```

This is because `vanilla_cvae_no_disentanglement.py` expects the class folders directly inside the dataset path.

---

## 3. Run in Google Colab

### Step 1 — Enable GPU

In Colab:

```text
Runtime → Change runtime type → GPU
```

Check GPU:

```python
!nvidia-smi
```

---

### Step 2 — Clone the repository

```python
%cd /content
!git clone https://github.com/asha1603/20261R0136COSE47400.git
%cd /content/20261R0136COSE47400
```

---

### Step 3 — Install packages

```python
!pip install -q timm lpips scikit-learn pandas tqdm matplotlib pillow kaggle
```

---

## 4. Download ImageOASIS Dataset from Kaggle

### Step 1 — Add Kaggle API key to Colab Secrets

In Colab, open the **Secrets** tab and add:

```text
KAGGLE_USERNAME
KAGGLE_KEY
```

### Step 2 — Configure Kaggle API

```python
import os
from google.colab import userdata

os.makedirs('/root/.kaggle', exist_ok=True)

kaggle_username = userdata.get('KAGGLE_USERNAME')
kaggle_key = userdata.get('KAGGLE_KEY')

with open('/root/.kaggle/kaggle.json', 'w') as f:
    f.write(f'{{"username":"{kaggle_username}","key":"{kaggle_key}"}}')

!chmod 600 /root/.kaggle/kaggle.json
```

### Step 3 — Download and unzip dataset

```python
!mkdir -p /content/imagesoasis
!kaggle datasets download -d ninadaithal/imagesoasis -p /content/imagesoasis --unzip
```

Check the dataset folders:

```python
!find /content/imagesoasis -maxdepth 3 -type d | head -30
```

---

## 5. Create the 41-Subject Dataset

Run the dataset script:

```python
!python "Old_FiLM/Deep learning dataset updated.py" \
  --source_dir "/content/imagesoasis" \
  --output_dir "/content/deep_learning_41_subject_dataset" \
  --num_subjects 41 \
  --slice_min 100 \
  --slice_max 160 \
  --classes "Non Demented" "Very mild Dementia" "Mild Dementia"
```

Check that the dataset was created correctly:

```python
!find "/content/deep_learning_41_subject_dataset" -maxdepth 2 -type d
!cat "/content/deep_learning_41_subject_dataset/Deep learning selected subjects.txt" | wc -l
```

Expected subject count:

```text
41
```

---

## 6. Run the Vanilla CVAE Baseline

You do **not** need to edit `vanilla_cvae_no_disentanglement.py`.

Run:

```python
!OLD_FILM_DATA_DIR="/content/deep_learning_41_subject_dataset/all" \
 OLD_FILM_CLASSIFIER_PATH="/content/20261R0136COSE47400/Old_FiLM/best_classifier_resnet18_weights_42.pth" \
 OLD_FILM_RESULTS_DIR="/content/20261R0136COSE47400/Old_FiLM/baseline_results" \
 OLD_FILM_CHECKPOINT_DIR="/content/20261R0136COSE47400/Old_FiLM/baseline_checkpoints" \
 python Old_FiLM/vanilla_cvae_no_disentanglement.py
```

---

## 7. Output Locations

### Checkpoint

The best baseline checkpoint will be saved in:

```text
/content/20261R0136COSE47400/Old_FiLM/baseline_checkpoints/best_cvae_baseline_ablation_hyperparameter.pth
```

### Generated images and metrics

The generated samples and evaluation outputs will be saved in:

```text
/content/20261R0136COSE47400/Old_FiLM/baseline_results/
```

The sample images are usually inside:

```text
/content/20261R0136COSE47400/Old_FiLM/baseline_results/GEN_SAMPLES_BASELINE_HYPERPARAMETER/
```

Check outputs:

```python
!find /content/20261R0136COSE47400/Old_FiLM/baseline_results -maxdepth 3 -type f
!find /content/20261R0136COSE47400/Old_FiLM/baseline_checkpoints -maxdepth 2 -type f
```

---

## 8. Download Results

Zip the baseline results:

```python
!zip -r /content/vanilla_cvae_baseline_results.zip \
  /content/20261R0136COSE47400/Old_FiLM/baseline_results \
  /content/20261R0136COSE47400/Old_FiLM/baseline_checkpoints
```

Download:

```python
from google.colab import files
files.download('/content/vanilla_cvae_baseline_results.zip')
```

---

## 9. Common Errors and Fixes

### Error: `FileNotFoundError` for dataset

Make sure this path exists:

```text
/content/deep_learning_41_subject_dataset/all
```

Check:

```python
!find /content/deep_learning_41_subject_dataset/all -maxdepth 2 -type d
```

---

### Error: classifier checkpoint not found

Make sure the checkpoint exists:

```python
!ls /content/20261R0136COSE47400/Old_FiLM/best_classifier_resnet18_weights_42.pth
```

If the file is missing, upload or pull it again.

---

### Error: no images found

This usually means the dataset path is wrong.

The baseline expects this structure:

```text
OLD_FILM_DATA_DIR/
├── Non Demented/
├── Very mild Dementia/
└── Mild Dementia/
```

So `OLD_FILM_DATA_DIR` should point to:

```text
/content/deep_learning_41_subject_dataset/all
```

not:

```text
/content/deep_learning_41_subject_dataset
```

---

### Error: CUDA out of memory

Open `vanilla_cvae_no_disentanglement.py` and reduce:

```python
BATCH_SIZE = 32
```

to:

```python
BATCH_SIZE = 16
```

or:

```python
BATCH_SIZE = 8
```

---

## 10. Important Notes

- The dataset script already creates `train/` and `val/` folders, but this baseline uses the `all/` folder and performs its own subject-level split internally.
- The baseline uses 3 dementia classes because the classifier checkpoint is a 3-class ResNet18 model.
- Moderate Dementia is not used in this baseline run.
- For a fair comparison, make sure the proposed model and baseline are both trained using the same 41-subject dataset.
