# Deep Learning Mid Experiments

This repository is split by experiment so each model can be run from its own
folder.

## Folders

```text
deep_learning_mid_unet_cvae/
  vanilla/
    Baseline_Vanilla_CVAE.ipynb
    scripts/
    legacy_4class_run/

  unet/
    configs/
    scripts/
    src/mid_unet_cvae/
```

## Which Folder To Use

- `unet/` is the main mid-presentation experiment: classifier-guided U-Net CVAE.
- `vanilla/` is the baseline experiment: vanilla CVAE notebook aligned to the
  main U-Net CVAE data/loss settings, except it intentionally has no U-Net skip
  connections and no classifier-guided loss.

Run commands from inside the model folder you are using.

```powershell
cd unet
python scripts\smoke_test.py
```

```powershell
cd vanilla
python scripts\smoke_test.py
# Open Baseline_Vanilla_CVAE.ipynb in Jupyter or Colab for training.
```
