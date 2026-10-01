"""
sampler_torch.py
===================
Wraps the ALREADY-VERIFIED weight computation from src/patching/sampler.py
into a torch WeightedRandomSampler. Does not reimplement or recompute the
weighting logic -- only converts the verified per-patch weight list into
the object torch's DataLoader expects.
"""

import os
import sys

from torch.utils.data import WeightedRandomSampler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.patching.sampler import compute_weights_pixel_weighted, resolve_type_specific_target


def build_weighted_sampler(patch_rows: list, target_tin_pixel_multiplier: float,
                            max_tin_pixel_fraction: float, patch_size: int,
                            cap_percentile: float = 90.0):
    """patch_rows: the train-split patch_index rows for ONE true_type
    (already filtered by the caller -- e.g. MicroSteelTorchDataset.patch_rows).
    The target is derived from this exact type's natural TiN pixel fraction,
    then capped at the configured maximum. Returns a replacement sampler
    with num_samples == len(patch_rows)."""
    target_info = resolve_type_specific_target(
        patch_rows,
        target_multiplier=target_tin_pixel_multiplier,
        max_target_fraction=max_tin_pixel_fraction,
    )
    solved = compute_weights_pixel_weighted(
        patch_rows, target_info["target_tin_pixel_fraction"],
        patch_size, cap_percentile
    )
    weights = solved["weights"]
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
