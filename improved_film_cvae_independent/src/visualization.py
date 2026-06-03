import os
import torch
from torchvision.utils import make_grid, save_image

from .data import NUM_CLASSES


@torch.no_grad()
def save_stage_grid(model, batch, device, output_path: str, translation_skip_scale: float = 0.6, max_rows: int = 4):
    model.eval()
    x, y = batch[0].to(device), batch[1].to(device)
    x = x[:max_rows]
    y = y[:max_rows]

    all_imgs = []
    for i in range(x.size(0)):
        xi = x[i : i + 1]
        row = [xi.cpu()]
        for target in range(NUM_CLASSES):
            t = torch.tensor([target], dtype=torch.long, device=device)
            scale = 1.0 if target == int(y[i].item()) else translation_skip_scale
            out, _, _, _ = model(xi, t, skip_scale=scale)
            row.append(out.cpu())
        all_imgs.append(torch.cat(row, dim=0))

    grid = make_grid(torch.cat(all_imgs, dim=0), nrow=NUM_CLASSES + 1, pad_value=1.0)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_image(grid, output_path)
