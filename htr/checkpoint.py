"""Versioned checkpoints for the refactored pipeline."""
import torch
from .data import labels_for
from .model import HandwritingCNN

FORMAT_VERSION = 1
PREPROCESSING = "upright-28x28-mnist-normalization-v1"


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported checkpoint. Train with python -m htr.train.")
    if checkpoint.get("preprocessing") != PREPROCESSING:
        raise ValueError("Checkpoint preprocessing is incompatible")
    labels = labels_for(checkpoint["config"]["dataset"])
    if checkpoint.get("labels") != labels:
        raise ValueError("Checkpoint class mapping is incompatible")
    model = HandwritingCNN(len(labels), checkpoint["config"]["dropout"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint
