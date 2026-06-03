import argparse
import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from src.data import MRISubjectDataset, CLASSES
from src.models import MRIClassifier
from src.utils import get_device, load_state_dict_flexible, ensure_dir
from src.gradcam import GradCAM, find_last_conv_layer, overlay_heatmap


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--classifier_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="outputs/gradcam")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--num_images", type=int, default=8)
    p.add_argument("--model_name", type=str, default="resnet18")
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()
    ensure_dir(args.output_dir)

    ds = MRISubjectDataset(args.data_dir, "val", args.image_size, classifier_transform=True)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    model = MRIClassifier(model_name=args.model_name, pretrained=False).to(device)
    state, _ = load_state_dict_flexible(args.classifier_path, device)
    model.load_state_dict(state, strict=True)
    model.eval()

    target_layer = find_last_conv_layer(model)
    cam_fn = GradCAM(model, target_layer)

    count = 0
    for x_norm, y, subject_id, path in loader:
        if count >= args.num_images:
            break
        x_norm = x_norm.to(device)
        y = y.to(device)

        # Convert normalized image back to [0,1] for display.
        x_disp = (x_norm * 0.224 + 0.456).clamp(0, 1)

        cam, logits = cam_fn(x_norm, class_idx=y)
        pred = logits.argmax(dim=1).item()
        overlay = overlay_heatmap(x_disp[0], cam[0])

        plt.figure(figsize=(5, 5))
        plt.imshow(overlay)
        plt.axis("off")
        plt.title(f"True: {CLASSES[int(y.item())]} | Pred: {CLASSES[pred]}")
        out_path = os.path.join(args.output_dir, f"gradcam_{count:03d}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved: {out_path}")
        count += 1

    cam_fn.remove_hooks()


if __name__ == "__main__":
    main()
