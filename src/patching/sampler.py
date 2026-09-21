"""
sampler.py
===========
Computes per-patch sampling weights for TiN oversampling, and VERIFIES
(by simulating actual weighted draws, not just asserting) that the
resulting sample distribution achieves the configured target_tin_fraction.

Two modes, both usable from the 4-config ablation grid described in the
project overview:
  - baseline / weighted-loss-only configs: uniform sampling (no oversampling)
  - patch-oversampling / both configs: weighted sampling toward TiN patches

This module does NOT depend on torch -- it returns plain per-patch weights
(a list of floats aligned with patch_index rows) that a torch
WeightedRandomSampler (or any other framework's equivalent) can consume
directly. Keeping it framework-agnostic makes it testable here without
requiring GPU infra, and reusable by whichever architecture (DeepLabV3+ or
Attention U-Net) actually trains on it.
"""

import csv
import json
import os
import sys
import random
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.common import load_config


def load_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def compute_weights(patch_rows: list, target_tin_fraction: float or None) -> list:
    """Returns a list of sampling weights, one per row in patch_rows (same order).

    If target_tin_fraction is None: uniform weight 1.0 for every patch
    (baseline / no oversampling).

    Otherwise: weight TiN-containing patches and non-TiN patches such that,
    under sampling WITH REPLACEMENT proportional to these weights, the
    expected fraction of TiN patches per draw equals target_tin_fraction.

    Derivation: let n_tin, n_other be counts. We want
        w_tin * n_tin / (w_tin * n_tin + w_other * n_other) = target_tin_fraction
    Fix w_other = 1. Solve for w_tin:
        w_tin = target_tin_fraction * n_other / ((1 - target_tin_fraction) * n_tin)
    """
    if target_tin_fraction is None:
        return [1.0] * len(patch_rows)

    if not (0.0 < target_tin_fraction < 1.0):
        raise ValueError(f"target_tin_fraction must be in (0,1), got {target_tin_fraction}")

    tin_flags = [r["tin_present"] in ("True", "true", "1") for r in patch_rows]
    n_tin = sum(tin_flags)
    n_other = len(patch_rows) - n_tin

    if n_tin == 0:
        raise ValueError("No TiN-containing patches found -- cannot compute oversampling weights.")
    if n_other == 0:
        raise ValueError("No non-TiN patches found -- cannot compute oversampling weights.")

    w_tin = target_tin_fraction * n_other / ((1 - target_tin_fraction) * n_tin)
    w_other = 1.0

    return [w_tin if flag else w_other for flag in tin_flags]


def simulate_sampling(patch_rows: list, weights: list, n_draws: int, seed: int = 0) -> dict:
    """Actually draws n_draws samples (with replacement) using the given
    weights and measures the achieved TiN fraction -- verification by
    simulation, not by trusting the weight-derivation algebra alone."""
    rng = random.Random(seed)
    tin_flags = [r["tin_present"] in ("True", "true", "1") for r in patch_rows]

    drawn_indices = rng.choices(range(len(patch_rows)), weights=weights, k=n_draws)
    n_tin_drawn = sum(1 for i in drawn_indices if tin_flags[i])

    unique_drawn = len(set(drawn_indices))
    return {
        "n_draws": n_draws,
        "achieved_tin_fraction": round(n_tin_drawn / n_draws, 4),
        "n_unique_patches_drawn": unique_drawn,
        "pct_unique": round(100.0 * unique_drawn / len(patch_rows), 2),
    }


def run(config: dict) -> dict:
    out_dir = config["paths"]["output_dir"]
    patch_index_path = os.path.join(out_dir, "patch_index.csv")
    target = config["sampler"]["target_tin_fraction"]

    patch_rows = load_rows(patch_index_path)
    train_rows = [r for r in patch_rows if r["split"] == "train"]

    if not train_rows:
        return {"check": "sampler", "error": "no train-split patches found"}

    n_tin = sum(1 for r in train_rows if r["tin_present"] in ("True", "true", "1"))
    natural_fraction = n_tin / len(train_rows)

    results = {}
    for label, tgt in [("uniform_baseline", None), (f"target_{target}", target)]:
        weights = compute_weights(train_rows, tgt)
        sim = simulate_sampling(train_rows, weights, n_draws=50000, seed=42)
        results[label] = {
            "target_tin_fraction": tgt,
            "natural_tin_fraction_in_train_split": round(natural_fraction, 4),
            "simulated_achieved_tin_fraction": sim["achieved_tin_fraction"],
            "close_to_target": (
                True if tgt is None else
                abs(sim["achieved_tin_fraction"] - tgt) < 0.01
            ),
            "pct_unique_patches_sampled_in_50k_draws": sim["pct_unique"],
        }

    all_close = all(v["close_to_target"] for v in results.values())

    # This is a substantive finding, not a formality: if the NATURAL
    # (unweighted) TiN-patch fraction is already close to the configured
    # oversampling target, then "patch-oversampling" vs "baseline" configs
    # in the ablation grid would barely differ -- undermining the whole
    # comparison the project is built around. Flag it loudly rather than
    # letting "target met: True" quietly hide it.
    oversampling_is_meaningful = abs(natural_fraction - target) >= 0.15 if target else None
    warning = None
    if target is not None and not oversampling_is_meaningful:
        warning = (
            f"WARNING: natural (unweighted) TiN-patch fraction in the train "
            f"split is already {round(natural_fraction, 4)}, close to the "
            f"configured target_tin_fraction={target}. This is a direct "
            f"consequence of heavy patch overlap (50%) smearing 'contains "
            f"any TiN pixel' across most of the patch population, even "
            f"though pixel-level TiN share is only ~0.17%. A "
            f"'patch-oversampling' config at this target would barely "
            f"differ from baseline uniform sampling -- the ablation would "
            f"not actually test what it claims to. Consider either raising "
            f"target_tin_fraction substantially (e.g. 0.8-0.9), or "
            f"redefining oversampling by per-patch TiN PIXEL PERCENTAGE "
            f"rather than binary presence, before running the 24-config grid."
        )

    findings = {
        "check": "sampler",
        "train_split_size": len(train_rows),
        "train_split_natural_tin_fraction": round(natural_fraction, 4),
        "configured_target_tin_fraction": target,
        "oversampling_is_meaningful_vs_baseline": oversampling_is_meaningful,
        "warning": warning,
        "simulation_results": results,
        "passed": all_close,
        "conclusion": (
            f"Natural TiN-patch fraction in train split is "
            f"{round(natural_fraction, 4)} ({n_tin}/{len(train_rows)}). "
            f"With target_tin_fraction={target}, weighted sampling achieves "
            f"{results[f'target_{target}']['simulated_achieved_tin_fraction']} "
            f"over 50,000 simulated draws (target met: "
            f"{results[f'target_{target}']['close_to_target']}). "
            f"This confirms the weight-derivation formula actually produces "
            f"the intended oversampling ratio, not just algebraically."
        ),
    }
    return findings


if __name__ == "__main__":
    cfg = load_config("configs/config.yaml")
    result = run(cfg)
    print(json.dumps(result, indent=2))
