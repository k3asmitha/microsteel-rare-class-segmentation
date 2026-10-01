"""
losses.py
===========
Builds the actual torch loss criterion for a given configuration
("baseline"/"patch" use plain CE, "loss"/"both" use class-weighted CE),
using the weight VALUES from the already-verified src/loss/class_weights.py
-- this file does not recompute or reinvent weighting logic, only converts
the verified {class_name: weight} dict into a torch tensor aligned with
class-ID ordering and hands it to nn.CrossEntropyLoss.
"""

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.loss.class_weights import compute_class_weights


def build_loss(configuration: str, manifest_rows: list, class_names: list,
               loss_scheme: str, true_type: str, type_class_membership: dict,
               effective_number_beta: float = None,
               split: str = "train", device: str = "cpu") -> nn.Module:
    """configuration: one of 'baseline', 'patch', 'loss', 'both'. Only
    'loss' and 'both' use class weighting -- 'baseline' and 'patch' use
    plain (unweighted) CE, since patch-level oversampling is a SAMPLING
    change, not a loss change, and mixing the two into one config would
    make the 4-way ablation meaningless.

    true_type and type_class_membership are REQUIRED (no default): class
    weights must be computed from the train pixels of the SAME processing
    type the model is training on, and only over the classes that exist in
    that type. An earlier version pooled all three types' pixels for every
    run -- a real bug (see class_weights.compute_class_weights docstring).
    Making these required arguments means that mistake can't silently
    return through a forgotten default."""
    if configuration not in ("baseline", "patch", "loss", "both"):
        raise ValueError(f"Unknown configuration '{configuration}'")
    uses_weighting = configuration in ("loss", "both")

    if not uses_weighting:
        return nn.CrossEntropyLoss()

    kwargs = {}
    if loss_scheme == "effective_number":
        kwargs["beta"] = effective_number_beta

    weights_dict = compute_class_weights(manifest_rows, class_names, loss_scheme,
                                          split=split, true_type=true_type,
                                          type_class_membership=type_class_membership,
                                          **kwargs)
    # class_names order defines class-ID alignment -- must match the
    # manifest's class_codebook.json ordering exactly, or weights will be
    # silently misapplied to the wrong classes.
    weight_tensor = torch.tensor([weights_dict[c] for c in class_names],
                                  dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=weight_tensor)
