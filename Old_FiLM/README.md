@'
## Main Files

- `cvae.py`  
  Main FiLM-based Conditional VAE model.

- `vanilla_cvae_no_disentanglement.py`  
  Vanilla CVAE baseline model.

- `classifier.py`  
  Classifier model used for classifier-guided loss.

## How to Run the Code

### 1. Open Google Colab

It is recommended to run this project using Google Colab with GPU enabled.

Go to:

Runtime → Change runtime type → GPU

---

### 2. Clone the Repository

```
python
!git clone https://github.com/asha1603/20261R0136COSE47400
%cd YOUR_REPO_NAME/Old_FiLM

```
### 3. Install Required Libraries
!pip install torch torchvision torchaudio
!pip install numpy pandas matplotlib seaborn scikit-learn tqdm
!pip install opencv-python lpips


### 4. Prepare Kaggle API Key

If the dataset is downloaded from Kaggle, save your Kaggle username and key in Colab Secrets.

Use these secret names:

KAGGLE_USERNAME
KAGGLE_KEY

Then run:

import os
from google.colab import userdata

os.makedirs("/root/.kaggle", exist_ok=True)

kaggle_username = userdata.get("KAGGLE_USERNAME")
kaggle_key = userdata.get("KAGGLE_KEY")

with open("/root/.kaggle/kaggle.json", "w") as f:
    f.write(f'{{"username":"{kaggle_username}","key":"{kaggle_key}"}}')

!chmod 600 /root/.kaggle/kaggle.json

### 5. Download the Dataset

!pip install kaggle
!mkdir -p /content/data

!kaggle datasets download -d ninadaithal/imagesoasis -p /content/data
!unzip -q /content/data/*.zip -d /content/data

### 6. Check Dataset Path

Before running the model, check the dataset folder:

!ls /content/data

If the code has a hardcoded dataset path, update the path inside cvae.py.

Example:

DATA_DIR = "/content/data"

### 7. Run the FiLM CVAE Model

To run the FiLM-based model:

!python cvae.py

### 8. Run the Vanilla Baseline

To run the baseline model:

!python vanilla_cvae_no_disentanglement.py

Use this only for comparison. The FiLM model is in cvae.py.

### 9. Check Outputs

After training, check the generated output folders:

!ls

Look for folders such as:

outputs
checkpoints
results

depending on how the code saves files.