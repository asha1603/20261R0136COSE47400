from pathlib import Path
import torch
import numpy as np
from PIL import Image, ImageDraw
from improved_film_modules import ImprovedFiLMUNetCVAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = Path("data")
out_dir = Path("outputs")
out_dir.mkdir(exist_ok=True)

image_paths = sorted(
    list(data_dir.rglob("*.jpg")) +
    list(data_dir.rglob("*.jpeg")) +
    list(data_dir.rglob("*.png"))
)

if not image_paths:
    raise SystemExit("No images found under data/. Check data link.")

image_path = image_paths[0]
img = Image.open(image_path).convert("L").resize((224, 224))
arr = np.asarray(img, dtype=np.float32) / 255.0
x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)

model = ImprovedFiLMUNetCVAE(
    img_size=224,
    in_channels=1,
    num_classes=3,
    latent_dim=128,
    class_dim=32,
    base_channels=32,
).to(device)

ckpt = torch.load("improved_film_cvae_latest.pth", map_location=device)
state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
model.load_state_dict(state)
model.eval()

class_names = [p.name for p in sorted(data_dir.iterdir()) if p.is_dir()]
if len(class_names) < 3:
    class_names = ["class_0", "class_1", "class_2"]

with torch.no_grad():
    mu, logvar, skips = model.encode(x)

    outputs = []
    for label in [0, 1, 2]:
        y = torch.tensor([label], dtype=torch.long, device=device)
        generated = model.decode(mu, y, skips, skip_scale=0.35)[0, 0].cpu().numpy()
        im = Image.fromarray((generated * 255).clip(0, 255).astype("uint8"))
        name = class_names[label] if label < len(class_names) else f"class_{label}"
        filename = f"improved_target_{label}_{name.replace(' ', '_')}.png"
        im.save(out_dir / filename)
        outputs.append((name, im))

tile_w, tile_h = 224, 254
canvas = Image.new("L", (tile_w * 4, tile_h), color=255)
draw = ImageDraw.Draw(canvas)

canvas.paste(img, (0, 30))
draw.text((5, 5), "Original", fill=0)

for i, (name, im) in enumerate(outputs, start=1):
    canvas.paste(im, (tile_w * i, 30))
    draw.text((tile_w * i + 5, 5), name, fill=0)

canvas.save(out_dir / "improved_all_targets_grid.png")

print("Source:", image_path)
print("Saved outputs/improved_all_targets_grid.png")
for p in sorted(out_dir.glob("improved_target_*.png")):
    print("Saved", p)
