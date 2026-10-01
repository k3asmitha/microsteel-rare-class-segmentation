"""
build_model.py
================
Factory: given an architecture name and config, returns the right model
plus its matching predict_probs adapter (for the frozen evaluator).

CPU-SIZE NOTE: Attention U-Net's base_ch defaults to 16, not the more
common 32-64, specifically because there is no GPU available anywhere in
this project (neither here nor on your machine). base_ch=32 quadruples
parameter count and roughly quadruples compute vs base_ch=16 at the
bottleneck. Raise it only if CPU training time is acceptable at the larger
size on your machine -- there's no correctness reason to prefer either.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.models.deeplabv3plus import build_deeplabv3plus, predict_probs as deeplab_predict_probs
from src.models.attention_unet import build_attention_unet, predict_probs as aunet_predict_probs


def build_model(architecture: str, num_classes: int, config: dict = None):
    """Returns (model, predict_probs_fn). predict_probs_fn has signature
    predict_probs_fn(model, image_patch, device) -> (num_classes,ps,ps) probs."""
    training_cfg = (config or {}).get("training", {})

    if architecture == "DeepLabV3Plus":
        encoder_weights = training_cfg.get("deeplab_encoder_weights", "imagenet")
        model = build_deeplabv3plus(num_classes, encoder_weights=encoder_weights, in_channels=1)
        return model, deeplab_predict_probs

    elif architecture == "AttentionUNet":
        base_ch = training_cfg.get("attention_unet_base_ch", 16)
        model = build_attention_unet(num_classes, in_channels=1, base_ch=base_ch)
        return model, aunet_predict_probs

    else:
        raise ValueError(f"Unknown architecture '{architecture}' -- must be "
                          f"'DeepLabV3Plus' or 'AttentionUNet'")
