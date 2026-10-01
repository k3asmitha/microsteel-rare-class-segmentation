"""
predict.py
============
Loads a trained checkpoint and builds a predict_fn(image_patch) -> probs
matching the FROZEN evaluator's production interface exactly (image-only,
no label -- see src/eval/evaluator.py's docstring on why this matters).
This is the bridge between "a trained model exists" and "the already-
verified reconstruct -> metrics -> aggregate pipeline can evaluate it".
"""

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.models.deeplabv3plus import predict_probs as deeplab_predict_probs
from src.models.attention_unet import predict_probs as aunet_predict_probs
from src.models.build_model import build_model


def load_trained_predict_fn(checkpoint_path: str, config: dict):
    """Returns (predict_fn, checkpoint_metadata). predict_fn has the exact
    image_patch -> probs signature evaluate_dataset expects -- it is
    architecturally impossible for this to receive a label, since the
    checkpoint/model have no access to one at inference time."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    architecture = checkpoint["architecture"]
    num_classes = checkpoint["num_classes"]

    model, adapter = build_model(architecture, num_classes, config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    def predict_fn(image_patch):
        return adapter(model, image_patch, device="cpu")

    return predict_fn, {
        "architecture": architecture,
        "configuration": checkpoint.get("configuration"),
        "true_type": checkpoint.get("true_type"),
        "trained_epoch": checkpoint.get("epoch"),
        "val_loss_at_checkpoint": checkpoint.get("val_loss"),
    }
