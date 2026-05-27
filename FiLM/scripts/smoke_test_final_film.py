from __future__ import annotations

import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from final_film_cvae.models import FinalFiLMDisentangledCVAE, ResNet18MRIClassifier


def main() -> None:
    model = FinalFiLMDisentangledCVAE()
    classifier = ResNet18MRIClassifier()
    x = torch.rand(2, 1, 224, 224)
    labels = torch.tensor([0, 2])
    output = model(x, labels)
    translated = model.translate(x, torch.tensor([1, 1]))
    logits = classifier(output["recon"])
    assert output["recon"].shape == (2, 1, 224, 224)
    assert output["z_content"].shape == (2, 96)
    assert output["z_class"].shape == (2, 32)
    assert translated.shape == (2, 1, 224, 224)
    assert logits.shape == (2, 3)
    print("Final FiLM model smoke test passed.")
    print("Recon:", tuple(output["recon"].shape), "| content:", tuple(output["z_content"].shape), "| class:", tuple(output["z_class"].shape))


if __name__ == "__main__":
    main()
