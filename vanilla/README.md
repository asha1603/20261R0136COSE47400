# Vanilla CVAE Baseline

This folder contains the baseline vanilla CVAE experiment.

## Contents

```text
vanilla/
  Baseline_Vanilla_CVAE.ipynb
  scripts/
    smoke_test.py
  checkpoints_vanilla_cvae/
    best_vanilla_cvae.pth          # created after training the aligned baseline
  outputs_vanilla_cvae/            # created after training the aligned baseline
  legacy_4class_run/
    checkpoints_vanilla_cvae/
    outputs_vanilla_cvae/
```

## Model Shape

The notebook defines `VanillaCVAE` with:

- grayscale MRI input
- image size `224`
- three dementia classes: `Non Demented`, `Very mild Dementia`, `Mild Dementia`
- latent dimension `128`
- label conditioning through encoder label maps and decoder one-hot vectors
- binary cross entropy reconstruction loss
- KL regularization weight `1.0`
- no U-Net skip connections
- no classifier-guided loss

This is the baseline comparison target for the main U-Net CVAE experiment in
`../unet`.

`legacy_4class_run/` contains the previous unaligned 128-resolution, four-class
run artifacts. They are kept for reference only and are not compatible with the
current aligned baseline model.

## Run

Open `Baseline_Vanilla_CVAE.ipynb` in Jupyter or Colab and run the cells from
top to bottom. The notebook writes datasets, outputs, and checkpoints inside
this `vanilla/` folder. It downloads the OASIS dataset from Kaggle when Kaggle
credentials are configured.

For a local Jupyter environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter notebook Baseline_Vanilla_CVAE.ipynb
```

To verify the aligned baseline model structure without re-running training:

```powershell
python scripts\smoke_test.py
```
