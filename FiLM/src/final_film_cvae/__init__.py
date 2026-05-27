"""Final FiLM-conditioned disentangled CVAE package."""
from .data import CLASS_TO_INDEX, INDEX_TO_CLASS, MRISliceDataset
from .models import FinalFiLMDisentangledCVAE, ResNet18MRIClassifier
