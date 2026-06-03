import torch
import torch.nn.functional as F


def mse(pred, target):
    return F.mse_loss(pred, target).item()


def psnr(pred, target, max_val: float = 1.0):
    mse_val = F.mse_loss(pred, target).clamp_min(1e-10)
    return (20 * torch.log10(torch.tensor(max_val, device=pred.device)) - 10 * torch.log10(mse_val)).item()


@torch.no_grad()
def classifier_stats(classifier, images, labels):
    logits = classifier(images)
    probs = torch.softmax(logits, dim=1)
    preds = probs.argmax(dim=1)
    acc = (preds == labels).float().mean().item()
    conf = probs.gather(1, labels[:, None]).mean().item()
    return acc, conf
