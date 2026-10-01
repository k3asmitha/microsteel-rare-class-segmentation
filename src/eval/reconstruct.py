"""
reconstruct.py
================
Reconstructs a full-image class prediction from per-patch predictions,
correctly handling the 50% overlap in patch_index.csv.

CRITICAL DESIGN POINT: overlapping regions are combined by averaging
per-class PROBABILITIES, never by averaging or voting on hard class labels
directly, and NEVER raw logits either. Class IDs are categorical, not
ordinal -- averaging "class 0" and "class 3" as numbers is meaningless.
Averaging logits is ALSO wrong: softmax is nonlinear, so
softmax(mean(logits)) != mean(softmax(logits)) in general, and a model
producing confident-but-opposite logits on two overlapping patches (e.g.
[10,1] and [1,10]) would average to an artificial midpoint that doesn't
reflect either patch's actual confidence. Every patch prediction passed in
here MUST already be a valid per-pixel probability distribution (softmax
already applied, summing to ~1.0 along the class axis) -- this is checked
and enforced (see reconstruct_prediction's validation below), not just
documented and hoped for.

This module is intentionally decoupled from any specific model -- it only
knows how to combine probability maps it's given. B and C's trained models
plug in by producing these probability arrays (post-softmax) at inference
time.
"""

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.common import load_config


def get_patch_coords_for_stem(patch_rows: list, stem: str) -> list:
    """Returns [(y0, x0, patch_size), ...] for every patch belonging to
    this source image, in patch_index.csv's own recorded order."""
    return [
        (int(r["y0"]), int(r["x0"]), int(r["patch_size"]))
        for r in patch_rows if r["source_stem"] == stem
    ]


def reconstruct_prediction(patch_probs: dict, patch_coords: list,
                            image_shape: tuple, num_classes: int,
                            validate_probabilities: bool = True) -> np.ndarray:
    """
    patch_probs: dict mapping (y0, x0) -> probability array of shape
                 (num_classes, patch_size, patch_size), summing to ~1 along
                 axis 0 at every pixel (a valid per-pixel class
                 distribution -- softmax already applied by the caller,
                 NOT raw logits; see module docstring for why).
    patch_coords: [(y0, x0, patch_size), ...] -- must cover every pixel of
                  image_shape at least once (grid.py's guarantee); this
                  function verifies that guarantee rather than assuming it.
    validate_probabilities: if True (default), checks every patch's
                  probability array is finite (no NaN/Inf), within [0,1]
                  elementwise, and sums to ~1.0 along the class axis before
                  accumulating it -- raises ValueError immediately if any
                  check fails. This is what prevents someone from silently
                  passing raw logits, unnormalized scores, or a numerically
                  broken model output and getting a plausible-looking but
                  mathematically meaningless result. Only disable this for
                  performance in a tight inference loop AFTER independently
                  confirming your model's output is correctly normalized.
    Returns: (H, W) int64 array of reconstructed class predictions.
    """
    H, W = image_shape
    prob_accum = np.zeros((num_classes, H, W), dtype=np.float64)
    count_accum = np.zeros((H, W), dtype=np.int32)

    for (y0, x0, ps) in patch_coords:
        key = (y0, x0)
        if key not in patch_probs:
            raise KeyError(f"Missing prediction for patch at (y0={y0}, x0={x0}) -- "
                            f"every patch in patch_coords must have a prediction.")
        probs = patch_probs[key]
        if probs.shape != (num_classes, ps, ps):
            raise ValueError(f"Patch at ({y0},{x0}): expected shape "
                              f"({num_classes},{ps},{ps}), got {probs.shape}")
        if validate_probabilities:
            if not np.all(np.isfinite(probs)):
                raise ValueError(
                    f"Patch at ({y0},{x0}): probability array contains "
                    f"NaN/Inf values -- this usually means a broken softmax "
                    f"or numerical explosion upstream in the model. Fix the "
                    f"model output before reconstruction, don't let it "
                    f"silently propagate into the reconstructed image."
                )
            if np.any(probs < 0) or np.any(probs > 1):
                raise ValueError(
                    f"Patch at ({y0},{x0}): probability array has values "
                    f"outside [0,1] (min={probs.min():.4f}, max={probs.max():.4f}) "
                    f"-- this is not a valid per-pixel probability "
                    f"distribution. Likely raw logits or unnormalized scores "
                    f"passed where post-softmax probabilities are required."
                )
            sums = probs.sum(axis=0)
            if not np.allclose(sums, 1.0, atol=1e-4):
                raise ValueError(
                    f"Patch at ({y0},{x0}): probability array does not sum "
                    f"to 1.0 along the class axis (min={sums.min():.4f}, "
                    f"max={sums.max():.4f}). This function requires "
                    f"post-softmax probabilities, NOT raw logits or "
                    f"unnormalized scores -- averaging those directly is "
                    f"mathematically wrong (see module docstring). Apply "
                    f"softmax before calling reconstruct_prediction."
                )
        prob_accum[:, y0:y0 + ps, x0:x0 + ps] += probs
        count_accum[y0:y0 + ps, x0:x0 + ps] += 1

    uncovered = (count_accum == 0)
    if uncovered.any():
        n = int(uncovered.sum())
        raise ValueError(
            f"{n} pixel(s) in the ({H},{W}) image are not covered by any "
            f"patch -- grid coverage is broken (should never happen given "
            f"grid.py's coverage guarantee; check patch_coords matches the "
            f"image this reconstruction is for)."
        )

    avg_probs = prob_accum / count_accum[np.newaxis, :, :]
    reconstructed = np.argmax(avg_probs, axis=0).astype(np.int64)

    # Output invariants -- cheap insurance against a future refactor
    # silently breaking one of these guarantees (argmax naturally satisfies
    # them today, but asserting explicitly catches regressions immediately
    # instead of producing a subtly-wrong downstream metric).
    assert reconstructed.shape == (H, W), f"shape invariant violated: {reconstructed.shape} != {(H, W)}"
    assert np.issubdtype(reconstructed.dtype, np.integer), f"dtype invariant violated: {reconstructed.dtype}"
    assert reconstructed.min() >= 0, f"class ID invariant violated: min={reconstructed.min()}"
    assert reconstructed.max() < num_classes, f"class ID invariant violated: max={reconstructed.max()} >= {num_classes}"

    return reconstructed


def one_hot_from_labels(label_arr: np.ndarray, num_classes: int) -> np.ndarray:
    """Convenience: turn a (H,W) integer label array into a
    (num_classes,H,W) one-hot probability array (prob=1.0 for the true
    class, 0.0 elsewhere). Used to build 'perfect model' and other synthetic
    predictors for testing the reconstruction and evaluation pipeline
    without needing an actual trained model."""
    H, W = label_arr.shape
    one_hot = np.zeros((num_classes, H, W), dtype=np.float64)
    for c in range(num_classes):
        one_hot[c] = (label_arr == c).astype(np.float64)
    return one_hot
