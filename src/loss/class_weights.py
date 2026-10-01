"""
class_weights.py
==================
Computes per-class weights for class-weighted cross-entropy, per the team's
decision to use CE only (not Tversky/focal, which the original project doc
had specified -- see README's decision log for that change).

Three standard schemes, all computed from REAL per-pixel class counts in
the train split (never hardcoded), because the correct weight values
depend entirely on the actual class imbalance in this dataset, which is
extreme (TiN is ~0.14% of train pixels vs Alpha's ~74%):

  - inverse_freq: classic "balanced" weighting, w_c = N / (K * n_c).
    Verified real ratio: TiN gets 513.5x Alpha's weight. This is a REAL
    STABILITY RISK, not a hypothetical one -- a 513x weight differential on
    a per-pixel loss can produce large gradient spikes whenever a TiN pixel
    appears in a batch, and is worth testing carefully (watch for loss
    spikes / NaN) before committing it across all 24 runs.
  - inverse_sqrt_freq: w_c = sqrt(inverse_freq). Real ratio: TiN gets 22.7x
    Alpha's weight -- a much gentler correction, standard practice when raw
    inverse frequency is judged too aggressive.
  - effective_number: Cui et al. 2019 "Class-Balanced Loss", w_c =
    (1-beta)/(1-beta^n_c), normalized to mean 1. CRITICAL FINDING: the
    beta values commonly cited in papers (0.99, 0.999, 0.9999) were
    calibrated for CLASS-level sample counts (tens to low-thousands), not
    PER-PIXEL counts in the tens of millions this dataset has. At those
    beta values, beta^n_c underflows to exactly 0.0 in float64 for EVERY
    class here, degenerating the formula into (1-beta) for all of them --
    i.e. NO differentiation at all, silently, with no error. Verified
    directly: beta=0.999 gives beta^n_TiN=4.5e-41 (fine) but
    beta^n_Alpha=0.0 (underflowed) -- and even beta=0.9999 mostly
    underflows too. Beta must be pushed to ~0.999999-0.99999999 at this
    scale to get real differentiation (11x-394x ratios at those values).
    Do not copy a beta value from a paper or tutorial without checking it
    against actual class pixel counts the way this module does.

DEFAULT: inverse_sqrt_freq, chosen as the standard middle ground between
"no correction" and the potentially destabilizing 513x raw inverse-frequency
ratio. Documented here as a reasoned choice, not asserted as objectively
correct -- if training with raw inverse_freq turns out stable, that's a
legitimate alternative; if inverse_sqrt still causes instability, an even
gentler correction or a max-weight cap should be considered.
"""

import csv
import json
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.common import load_config


def load_rows(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def get_class_pixel_totals(manifest_rows: list, class_names: list, split: str = None) -> dict:
    """Sums px_<ClassName> columns across manifest rows, optionally filtered
    to one split. This is the single source of truth for class balance --
    every weighting scheme below is derived from this, never from assumed
    percentages."""
    rows = manifest_rows if split is None else [r for r in manifest_rows if r["split"] == split]
    return {c: sum(int(r[f"px_{c}"]) for r in rows) for c in class_names}


def inverse_freq_weights(totals: dict) -> dict:
    grand_total = sum(totals.values())
    n_classes = len(totals)
    return {c: grand_total / (n_classes * n) for c, n in totals.items()}


def inverse_sqrt_freq_weights(totals: dict) -> dict:
    inv = inverse_freq_weights(totals)
    return {c: math.sqrt(w) for c, w in inv.items()}


def effective_number_weights(totals: dict, beta: float) -> dict:
    """Cui et al. 2019. Normalized so mean weight == 1 (matches the
    convention inverse_freq/inverse_sqrt_freq implicitly follow when class
    counts are balanced). Raises if beta is so small that ALL classes'
    beta^n_c underflow to 0 (the degenerate case documented above) -- fails
    loudly instead of silently returning uniform weights."""
    eff_num = {c: (1 - beta) / (1 - beta ** n) for c, n in totals.items()}
    if all(w == eff_num[list(totals.keys())[0]] for w in eff_num.values()):
        raise ValueError(
            f"effective_number_weights: all classes produced identical "
            f"weight {list(eff_num.values())[0]} at beta={beta} -- this "
            f"means beta^n_c underflowed to 0 for every class (degenerate "
            f"case documented in this module's docstring). Increase beta "
            f"(closer to 1, e.g. 0.999999+) to get real differentiation at "
            f"this dataset's per-pixel count scale."
        )
    mean_w = sum(eff_num.values()) / len(eff_num)
    return {c: w / mean_w for c, w in eff_num.items()}


def _scheme_weights(totals: dict, scheme: str, beta: float = None) -> dict:
    if scheme == "inverse_freq":
        return inverse_freq_weights(totals)
    elif scheme == "inverse_sqrt_freq":
        return inverse_sqrt_freq_weights(totals)
    elif scheme == "effective_number":
        if beta is None:
            raise ValueError("scheme='effective_number' requires beta to be specified")
        return effective_number_weights(totals, beta)
    raise ValueError(f"Unknown scheme '{scheme}' -- must be one of "
                      f"inverse_freq, inverse_sqrt_freq, effective_number")


def compute_class_weights(manifest_rows: list, class_names: list, scheme: str,
                           split: str = "train", beta: float = None,
                           true_type: str = None, type_class_membership: dict = None) -> dict:
    """Class weights computed EXCLUSIVELY from the training population the
    model will actually see.

    true_type=None  -> pooled over ALL processing types (only meaningful if
                       the model trains on all types at once; NOT what any
                       of the per-type experiments do).
    true_type="II"  -> only rows of that processing type, and only the
                       classes that EXIST in that type
                       (type_class_membership[true_type]). K in the formula
                       is the number of classes present in that type, not
                       the global 5.

    BUG HISTORY (real, fixed): train.py used to call this with every
    manifest row regardless of type, so a model trained only on Type II
    received weights computed from Type I + II + III pixels (TiN weight
    11.787 instead of the correct 10.752, TiB2 1.916 instead of 1.392),
    and K counted FeTiB/Fe2B which don't exist in Type II at all. Found by
    an outside reviewer recomputing Type II weights by hand.

    Classes absent from the requested type get weight 1.0. This value is
    irrelevant to the loss: no target pixel ever has an absent class, and
    PyTorch's weighted-mean CE only sums weights of TARGET classes. It only
    exists so the weight tensor has one entry per class ID.
    """
    if true_type is None:
        totals = get_class_pixel_totals(manifest_rows, class_names, split=split)
        if any(n == 0 for n in totals.values()):
            zero_classes = [c for c, n in totals.items() if n == 0]
            raise ValueError(
                f"compute_class_weights: class(es) {zero_classes} have ZERO "
                f"pixels in split='{split}' across the pooled rows -- cannot "
                f"compute a weight for a class with no examples."
            )
        return _scheme_weights(totals, scheme, beta)

    if type_class_membership is None or true_type not in type_class_membership:
        raise ValueError(
            f"compute_class_weights: true_type='{true_type}' requires "
            f"type_class_membership (e.g. config['expected_type_class_membership']) "
            f"containing that type, so absent classes can be identified "
            f"explicitly instead of guessed from zero counts."
        )

    type_rows = [r for r in manifest_rows if r["true_type"] == true_type]
    if not type_rows:
        raise ValueError(f"compute_class_weights: no manifest rows for true_type='{true_type}'")
    totals = get_class_pixel_totals(type_rows, class_names, split=split)

    present = [c for c in class_names if c in type_class_membership[true_type]]
    absent = [c for c in class_names if c not in type_class_membership[true_type]]

    # Verify the declared membership against the actual pixels -- fail loudly
    # if the config and the data disagree, in either direction.
    bad_absent = {c: totals[c] for c in absent if totals[c] != 0}
    if bad_absent:
        raise ValueError(f"Type {true_type}: classes declared absent but found pixels: {bad_absent}")
    bad_present = [c for c in present if totals[c] == 0]
    if bad_present:
        raise ValueError(f"Type {true_type}: classes declared present but have ZERO "
                          f"pixels in split='{split}': {bad_present}")

    weights = _scheme_weights({c: totals[c] for c in present}, scheme, beta)
    for c in absent:
        weights[c] = 1.0
    return weights


def weighted_ce_loss_numpy(logits, targets, class_weights):
    """Pure-numpy reference implementation of PyTorch's documented
    nn.CrossEntropyLoss(weight=...) formula:
        L_i = -w_{y_i} * log( exp(x_{i,y_i}) / sum_c exp(x_{i,c}) )
    Used ONLY to verify the class_weights values actually behave as
    intended (rarer class gets amplified loss for the same confidence
    error) without requiring torch to be installed. logits: (N, C) array,
    targets: (N,) int array of true class indices, class_weights: (C,)
    array aligned to class index."""
    import numpy as np
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    class_weights = np.asarray(class_weights, dtype=np.float64)

    # numerically stable log-softmax
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))

    n = logits.shape[0]
    per_sample_log_prob = log_probs[np.arange(n), targets]
    per_sample_weight = class_weights[targets]
    per_sample_loss = -per_sample_weight * per_sample_log_prob
    # PyTorch's default reduction='mean' divides by SUM OF WEIGHTS, not N --
    # this is a common source of subtle mismatches if reimplemented naively
    return per_sample_loss.sum() / per_sample_weight.sum()


if __name__ == "__main__":
    cfg = load_config("configs/config.yaml")
    out_dir = cfg["paths"]["output_dir"]
    manifest_rows = load_rows(os.path.join(out_dir, "manifest.csv"))
    class_names = ["Alpha", "TiB2", "TiN", "FeTiB", "Fe2B"]
    membership = cfg["expected_type_class_membership"]

    print("=== POOLED across all types (informational scale reference only) ===\n")
    for scheme, kwargs in [("inverse_freq", {}), ("inverse_sqrt_freq", {}),
                            ("effective_number", {"beta": 0.999999})]:
        w = compute_class_weights(manifest_rows, class_names, scheme, split="train", **kwargs)
        print(f"{scheme} {kwargs}: TiN/Alpha ratio={w['TiN'] / w['Alpha']:.1f}x")
        print(f"  {[(c, round(v, 3)) for c, v in w.items()]}\n")

    print("=== PER TYPE (what training actually uses) ===\n")
    for t in ["I", "II", "III"]:
        for scheme, kwargs in [("inverse_freq", {}), ("inverse_sqrt_freq", {}),
                                ("effective_number", {"beta": 0.999999})]:
            w = compute_class_weights(manifest_rows, class_names, scheme, split="train",
                                       true_type=t, type_class_membership=membership, **kwargs)
            present = membership[t]
            print(f"Type {t} {scheme}: TiN/Alpha={w['TiN'] / w['Alpha']:.1f}x  "
                  f"{[(c, round(w[c], 4)) for c in present]}")
        print()
