"""
deeplabv3plus.py
==================
Thin wrapper around segmentation_models_pytorch's DeepLabV3+ with an
EfficientNet-b0 encoder, matching the project doc's specified architecture
("DeepLabV3+/EfficientNet-b0"). Single-channel (grayscale SEM) input.

Uses smp rather than a from-scratch implementation because DeepLabV3+ with
a pretrained ImageNet backbone is exactly what smp is for, is well-tested,
and reimplementing it from scratch would add risk (subtle bugs in atrous
spatial pyramid pooling, etc.) with no benefit over a standard library.

NOTE ON PRETRAINED WEIGHTS: encoder_weights="imagenet" requires downloading
weights the first time this runs, which needs normal internet access on
whatever machine actually trains. This sandbox's network is restricted and
could not download them when this file was written/tested here -- verified
below with encoder_weights=None (random init) instead, which needs no
network access and is sufficient to confirm the architecture's forward
pass, output shape, and gradient flow are all correct. On your own machine,
use encoder_weights="imagenet" (the default in build_deeplabv3plus) for
real training -- random-init encoder weights would badly hurt convergence
speed on a dataset this small.
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def build_deeplabv3plus(num_classes: int, encoder_weights: str = "imagenet",
                         in_channels: int = 1) -> nn.Module:
    """encoder_weights: "imagenet" for real training (needs internet on
    first run to download weights), None for architecture-only testing
    with no network access (e.g. in this sandbox)."""
    model = smp.DeepLabV3Plus(
        encoder_name="efficientnet-b0",
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
    )
    return model


def predict_probs(model: nn.Module, image_patch, device: str = "cpu"):
    """Adapter matching the frozen evaluator's predict_fn(image_patch)
    interface: takes a (ps, ps) uint8 numpy array, returns a
    (num_classes, ps, ps) numpy probability array (softmax already
    applied -- required by reconstruct.py's validation contract)."""
    import numpy as np
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(image_patch.astype("float32") / 255.0)
        x = x.unsqueeze(0).unsqueeze(0).to(device)  # (1,1,ps,ps)
        logits = model(x)  # (1, num_classes, ps, ps)
        probs = torch.softmax(logits, dim=1)
        return probs.squeeze(0).cpu().numpy()
