# FiLM-CVAE for Dementia Stage MRI Generation

This folder contains the FiLM-based Conditional Variational Autoencoder implementation used for dementia-stage MRI generation using the OASIS-1 MRI dataset.

The model is designed to preserve anatomical structure while conditioning the generated MRI image on a target dementia stage.

---

## Main Files

* `cvae.py`
  Main FiLM-based Conditional VAE model.

* `vanilla_cvae_no_disentanglement.py`
  Vanilla CVAE baseline model used for comparison.

* `classifier.py`
  Classifier model used for classifier-guided loss.

* `Deep learning dataset updated.py`
  Dataset preparation script. It creates a Team-I-style OASIS subset using 41 subjects and ROI slices 100–160.

* `best_classifier_resnet18_weights_42.pth`
  Pretrained ResNet18 classifier checkpoint used for classifier-guided loss in the FiLM-CVAE.

---

## Dataset Setting

This project uses the ImageOASIS dataset from Kaggle:

```text
ninadaithal/imagesoasis
```
the dataset preparation script creates a subset using:

```text
Dataset: OASIS-1
Subjects: 41 individuals
Classes: Non Demented, Very mild Dementia, Mild Dementia
Format: 2D axial grayscale MRI slices
ROI selection: slices 100–160
Split: subject-level train/validation split
```

The subject-level split is used to avoid data leakage between training and validation sets. This is important because MRI slices from the same subject are highly similar, so random slice-level splitting can produce misleading results.

---

## How to Run the Code

### 1. Open Google Colab

It is recommended to run this project using Google Colab with GPU enabled.

Go to:

```text
Runtime → Change runtime type → GPU
```

Check the GPU:

```python
!nvidia-smi
```

---

### 2. Clone the Repository

```python
!git clone https://github.com/asha1603/20261R0136COSE47400
%cd 20261R0136COSE47400/Old_FiLM
```

If the folder name is different, change the `%cd` path based on your repository structure.

---

### 3. Install Required Libraries

```python
!pip install -q torch torchvision torchaudio
!pip install -q numpy pandas matplotlib seaborn scikit-learn tqdm
!pip install -q opencv-python lpips timm kaggle
```

---

### 4. Prepare Kaggle API Key

If the dataset is downloaded from Kaggle, save your Kaggle username and key in Colab Secrets.

Use these secret names:

```text
KAGGLE_USERNAME
KAGGLE_KEY
```

Then run:

```python
import os
from google.colab import userdata

os.makedirs("/root/.kaggle", exist_ok=True)

kaggle_username = userdata.get("KAGGLE_USERNAME")
kaggle_key = userdata.get("KAGGLE_KEY")

with open("/root/.kaggle/kaggle.json", "w") as f:
    f.write(f'{{"username":"{kaggle_username}","key":"{kaggle_key}"}}')

!chmod 600 /root/.kaggle/kaggle.json
```

---

### 5. Download the Dataset

```python
!mkdir -p /content/imagesoasis
!kaggle datasets download -d ninadaithal/imagesoasis -p /content/imagesoasis --unzip
```

Check the extracted folders:

```python
!find /content/imagesoasis -maxdepth 3 -type d | head -30
```

---

### 6. Create the Dataset

Make sure `Deep learning dataset updated.py` is uploaded or available in `/content`.

Check the file:

```python
!ls /content
```

Then run:

```python
!python "/content/Deep learning dataset updated.py" \
  --source_dir /content/imagesoasis \
  --output_dir "/content/Deep learning dataset" \
  --num_subjects 41 \
  --slice_min 100 \
  --slice_max 160 \
  --classes "Non Demented" "Very mild Dementia" "Mild Dementia"
```

This creates:

```text
/content/Deep learning dataset/
├── all/
├── train/
├── val/
├── Deep learning metadata.csv
├── Deep learning selected subjects.txt
├── Deep learning train subjects.txt
└── Deep learning val subjects.txt
```

Check the created dataset:

```python
!find "/content/Deep learning dataset" -maxdepth 2 -type d
!cat "/content/Deep learning dataset/Deep learning selected subjects.txt" | wc -l
```

The selected subject count should be:

```text
41
```

---

### 7. Prepare the Classifier Checkpoint

The FiLM-CVAE uses the pretrained classifier checkpoint:

```text
best_classifier_resnet18_weights_42.pth
```

Make sure this file is located in the same folder as `cvae.py`.

If the code expects the checkpoint inside `classification_results/`, create the folder and copy the file:

```python
!mkdir -p classification_results
!cp "best_classifier_resnet18_weights_42.pth" "classification_results/best_classifier_resnet18_weights_42.pth"
```

---

### 8. Run the FiLM-CVAE Model

Use the `all/` dataset folder for CVAE training:

```python
!OLD_FILM_DATA_DIR="/content/Deep learning dataset/all" \
 OLD_FILM_RESULTS_DIR="/content/Old_FiLM/evaluation_results" \
 OLD_FILM_CHECKPOINT_DIR="/content/Old_FiLM/checkpoints" \
 python cvae.py
```

If your repository path is different, adjust `/content/Old_FiLM` to match your actual folder path.

---

### 9. Run the Vanilla CVAE Baseline

Use this only for comparison.

```python
!OLD_FILM_DATA_DIR="/content/Deep learning dataset/all" \
 OLD_FILM_RESULTS_DIR="/content/Old_FiLM/vanilla_results" \
 OLD_FILM_CHECKPOINT_DIR="/content/Old_FiLM/vanilla_checkpoints" \
 python vanilla_cvae_no_disentanglement.py
```

The FiLM model is the main proposed model, while the Vanilla CVAE is used as a baseline.

---

### 10. Run Evaluation and Visualization

Run latent-space metrics:

```python
!OLD_FILM_DATA_DIR="/content/Deep learning dataset/all" \
 OLD_FILM_RESULTS_DIR="/content/Old_FiLM/evaluation_results" \
 OLD_FILM_CHECKPOINT_DIR="/content/Old_FiLM/checkpoints" \
 python metrics/latent_metric.py
```

Run latent-space visualization:

```python
!OLD_FILM_DATA_DIR="/content/Deep learning dataset/all" \
 OLD_FILM_RESULTS_DIR="/content/Old_FiLM/evaluation_results" \
 OLD_FILM_CHECKPOINT_DIR="/content/Old_FiLM/checkpoints" \
 python visualization/latent_viz.py
```

Run Grad-CAM visualization:

```python
!python visualization/gradcam_viz.py
```

---

### 11. Check Outputs

After training and evaluation, check the output folders:

```python
!find /content/Old_FiLM/evaluation_results -maxdepth 3 -type f
!find /content/Old_FiLM/checkpoints -maxdepth 2 -type f
```

Common outputs include:

```text
evaluation_results/
checkpoints/
GEN_SAMPLES/
pca_z_class.png
pca_z_content.png
tsne_z_class.png
tsne_z_content.png
comparison_sample.png
```

---

### 12. Download Results

Zip the outputs:

```python
!zip -r /content/film_team_i_style_results.zip /content/Old_FiLM/evaluation_results /content/Old_FiLM/checkpoints
```

Download:

```python
from google.colab import files
files.download("/content/film_team_i_style_results.zip")
```

---

## Notes

* The dataset is prepared using 41 selected subjects and slices 100–160 to follow the Team-I-style dataset setting.
* The classifier checkpoint is a 3-class ResNet18 classifier, so the dataset script uses only:

  * Non Demented
  * Very mild Dementia
  * Mild Dementia
* Moderate Dementia is not included in the current run because the provided classifier checkpoint uses 3 output classes.
* Subject-level splitting is used for the prepared train/validation folders to avoid data leakage.
* The FiLM-CVAE is trained using the `all/` folder to follow the pooled-slice CVAE training setting used in the original Team I experiment.
