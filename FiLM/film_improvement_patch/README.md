# FiLM-CVAE Improvement Patch

This patch is meant to improve your **existing FiLM-CVAE code**. It is **not** a full replacement repository.

The Team I GitHub repo is used only as a reference for the general idea: **CVAE + FiLM conditioning + classifier-guided MRI generation**. This patch focuses on fixing the common problem where the model reconstructs MRI slices very well, but the outputs for different target dementia stages look almost identical.

---

## 1. What this patch improves

Your current FiLM-CVAE probably works like this:

```python
output = model(input_image, target_label)
loss = reconstruction_loss(output, input_image) + kl_loss + classifier_loss
```

The problem is that U-Net skip connections can make the decoder copy the source MRI too strongly. This gives very good reconstruction scores, but weak target-stage translation.

This patch changes the training idea into two branches:

```python
# Branch 1: same-label reconstruction
recon = model(x, original_label, skip_scale=1.0)

# Branch 2: different-label translation
translated = model(x, different_target_label, skip_scale=0.35)
```

Main changes:

1. **Target-conditioned gated skip connections**
2. **Full skip strength for reconstruction**
3. **Reduced skip strength for translation**
4. **Random different target labels during training**
5. **Classifier-guided loss for translated images**
6. **Weak structure-preserving loss instead of fake paired ground truth**

---

## 2. Files included

```text
film_improvement_patch/
├── README.md                    # this file
├── README_MERGE.md              # shorter explanation of how to merge
├── improved_film_modules.py     # improved FiLM + gated skip CVAE model
├── improved_losses.py           # reconstruction + translation losses
├── train_step_patch.py          # one-batch training step for your loop
├── example_train_loop.py        # minimal example loop
├── smoke_test_film_patch.py     # quick forward/loss test
└── requirements_patch.txt       # minimal required packages
```

---

## 3. Install requirements

Inside the patch folder, run:

```bash
pip install -r requirements_patch.txt
```

The minimal requirements are:

```text
torch
torchvision
tqdm
numpy
```

If you already use PyTorch in Colab, you may not need to install anything extra.

---

## 4. First test: smoke test

Before merging into your real code, run:

```bash
python smoke_test_film_patch.py
```

Expected output should look like this:

```text
Device: cuda
Trainable parameters: 2567329
Recon shape: (1, 1, 224, 224)
Translated shape: (1, 1, 224, 224)
Loss: ...
Smoke test passed.
```

If you see `Device: cpu`, it is still okay. It just means the test ran without GPU.

---

## 5. How to copy the files into your project

Copy these three files into the same folder as your current training script, for example beside your current `cvae.py`:

```text
improved_film_modules.py
improved_losses.py
train_step_patch.py
```

Example project structure after copying:

```text
your_project/
├── cvae.py
├── train.py
├── improved_film_modules.py
├── improved_losses.py
└── train_step_patch.py
```

Or, if your project uses a `src/` folder:

```text
your_project/
├── src/
│   ├── improved_film_modules.py
│   ├── improved_losses.py
│   └── train_step_patch.py
└── train.py
```

If you put the files inside `src/`, change the imports like this:

```python
from src.improved_film_modules import ImprovedFiLMUNetCVAE
from src.improved_losses import FiLMCVAELoss
from src.train_step_patch import train_one_batch_improved_film
```

If you put them beside your training script, use:

```python
from improved_film_modules import ImprovedFiLMUNetCVAE
from improved_losses import FiLMCVAELoss
from train_step_patch import train_one_batch_improved_film
```

---

## 6. Replace your model creation

In your current training code, replace your old FiLM-CVAE model with:

```python
from improved_film_modules import ImprovedFiLMUNetCVAE

model = ImprovedFiLMUNetCVAE(
    img_size=224,
    in_channels=1,
    num_classes=3,
    latent_dim=128,
    class_dim=32,
    base_channels=32,
).to(device)
```

Important: this template assumes your MRI slices are resized to:

```text
1 × 224 × 224
```

So your transform should include something like:

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
```

---

## 7. Replace your loss setup

Add this to your training script:

```python
from improved_losses import FiLMCVAELoss

loss_fn = FiLMCVAELoss(
    num_classes=3,
    lambda_l1=8.0,
    lambda_bce=1.0,
    lambda_edge=2.0,
    beta_kl=0.05,
    lambda_trans_lowfreq=1.0,
    lambda_trans_edge=0.5,
    lambda_cls=1.5,
    lambda_tv=0.02,
    lambda_diversity=0.25,
    min_translation_delta=0.015,
)
```

Recommended starting values:

```text
beta_kl = 0.01 to 0.10
lambda_cls = 1.0 to 2.0
translation_skip_scale = 0.35
```

---

## 8. Replace your batch training step

Import the patched training step:

```python
from train_step_patch import train_one_batch_improved_film
```

Inside your training loop, replace your old single-batch training code with:

```python
for epoch in range(1, num_epochs + 1):
    model.train()
    running = {}

    for batch in train_loader:
        metrics = train_one_batch_improved_film(
            model=model,
            batch=batch,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
            classifier=frozen_classifier,      # use None if you do not have it yet
            translation_skip_scale=0.35,
            cls_warmup_epochs=10,
            kl_warmup_epochs=20,
        )

        for key, value in metrics.items():
            running[key] = running.get(key, 0.0) + value

    n = max(1, len(train_loader))
    print(f"Epoch {epoch}", {k: round(v / n, 5) for k, v in running.items()})
```

If your frozen classifier is not ready yet, use:

```python
classifier=None
```

Example:

```python
metrics = train_one_batch_improved_film(
    model=model,
    batch=batch,
    optimizer=optimizer,
    loss_fn=loss_fn,
    device=device,
    epoch=epoch,
    classifier=None,
    translation_skip_scale=0.35,
)
```

This will still train the improved reconstruction + translation structure, but without classifier guidance.

---

## 9. How to use your frozen classifier

If you already trained a classifier, load it before the CVAE training loop:

```python
classifier = YourClassifierClass(num_classes=3).to(device)
checkpoint = torch.load("path/to/best_classifier.pth", map_location=device)

# Depending on how your checkpoint was saved:
classifier.load_state_dict(checkpoint["model_state_dict"])
# or:
# classifier.load_state_dict(checkpoint)

classifier.eval()
for p in classifier.parameters():
    p.requires_grad = False
```

Then pass it here:

```python
classifier=classifier
```

Important: the loss file assumes your classifier expects normalized input using:

```python
Normalize(mean=[0.456], std=[0.224])
```

If your classifier used a different mean/std, edit this part in `improved_losses.py`:

```python
def normalize_for_classifier(x, mean=0.456, std=0.224):
    return (x - mean) / std
```

---

## 10. Minimal runnable example

The file `example_train_loop.py` gives a small example using `torchvision.datasets.ImageFolder`.

Run it with:

```bash
python example_train_loop.py
```

It expects your data folder to look like:

```text
data/
├── Non Demented/
├── Very mild Dementia/
└── Mild Dementia/
```

Warning: `ImageFolder` assigns labels alphabetically by folder name. Make sure your classifier and CVAE use the same class order. If your current project already has a custom OASIS dataset loader, it is better to keep your loader and only copy the model/loss/training-step patch.

---

## 11. Colab usage

Upload or copy the patch files into your Colab working directory:

```python
!ls
```

You should see:

```text
improved_film_modules.py
improved_losses.py
train_step_patch.py
```

Then run:

```python
!python smoke_test_film_patch.py
```

For training, use your existing dataset mounting code, then import the patch files in your notebook:

```python
from improved_film_modules import ImprovedFiLMUNetCVAE
from improved_losses import FiLMCVAELoss
from train_step_patch import train_one_batch_improved_film
```

For Colab T4, recommended starting settings:

```python
batch_size = 8
latent_dim = 128
class_dim = 32
translation_skip_scale = 0.35
num_epochs = 50
```

If you get CUDA out-of-memory, reduce:

```python
batch_size = 4
```

---

## 12. How to save checkpoints

Inside your epoch loop, save like this:

```python
torch.save(
    {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss_config": loss_fn.__dict__,
    },
    "improved_film_cvae_latest.pth",
)
```

For the best checkpoint, use your validation score:

```python
if val_score < best_val_score:
    best_val_score = val_score
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        "best_improved_film_cvae.pth",
    )
```

Do not push `.pth` files to GitHub unless your professor specifically allows it. Usually, put checkpoints in Google Drive or release assets instead.

---

## 13. What to monitor during training

The training step returns these metrics:

```text
loss
rec_l1
rec_bce
rec_edge
rec_kl
trans_lowfreq
trans_edge
trans_kl
tv
diversity
cls
target_conf
delta
```

Important ones:

```text
rec_l1        lower is better reconstruction
target_conf   higher means classifier sees translated image as target class
delta         average difference between reconstruction and translation
cls           lower means better target-class guidance
```

If `delta` is almost zero, the generated stages are still too similar. Try:

```python
translation_skip_scale = 0.20
lambda_cls = 2.0
lambda_diversity = 0.5
```

If outputs look too artificial, try:

```python
translation_skip_scale = 0.50
lambda_cls = 1.0
lambda_diversity = 0.1
```

---

## 14. Suggested evaluation for your report

Do not only report reconstruction metrics like MSE, PSNR, and SSIM. Those mainly prove that the model can copy the input.

Use two groups of metrics:

### Reconstruction quality

```text
MSE
PSNR
SSIM
L1 reconstruction loss
```

### Target-stage controllability

```text
Classifier confidence for requested target class
Classifier predicted target accuracy
Mean absolute difference between reconstruction and translation
Stage-grid visual comparison
Difference maps
```

For your final explanation, you can say:

> The improved FiLM-CVAE uses target-conditioned gated skip connections to reduce copy-only behavior. During reconstruction, the model uses the original class label and full skip strength to preserve anatomical structure. During translation, the model receives a different target dementia label and reduced skip strength, forcing the target condition to influence the generated output more strongly.

---

## 15. Important limitation wording

Since you probably do not have paired longitudinal ground truth for the same patient at every dementia stage, do not claim that the translated MRI is a verified real disease progression.

Use this wording instead:

> The translated images are stage-conditioned synthetic MRI outputs. They should be interpreted as model-generated visualizations guided by the target class label, not as clinically verified longitudinal disease progression for the same patient.

---

## 16. Quick GitHub push reminder

After copying the files into your repo:

```bash
git status
git add improved_film_modules.py improved_losses.py train_step_patch.py README.md
git commit -m "Add improved FiLM-CVAE gated skip training patch"
git push origin main
```

If your branch is not `main`, replace `main` with your branch name.
