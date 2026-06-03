import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()

    def __call__(self, x, class_idx=None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1)
        score = logits.gather(1, class_idx[:, None]).sum()
        score.backward(retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam_min = cam.amin(dim=(2, 3), keepdim=True)
        cam_max = cam.amax(dim=(2, 3), keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam, logits


def find_last_conv_layer(model):
    last_conv = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise RuntimeError("No Conv2d layer found for Grad-CAM.")
    return last_conv


def overlay_heatmap(gray_img, heatmap):
    """Return simple RGB overlay as numpy array in [0, 1]."""
    img = gray_img.detach().cpu().squeeze().numpy()
    hm = heatmap.detach().cpu().squeeze().numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    rgb = np.stack([img, img, img], axis=-1)
    overlay = rgb.copy()
    overlay[..., 0] = np.maximum(overlay[..., 0], hm)
    overlay[..., 1] *= (1 - 0.35 * hm)
    overlay[..., 2] *= (1 - 0.35 * hm)
    return np.clip(overlay, 0, 1)
