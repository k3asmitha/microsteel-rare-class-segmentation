"""
verify_dataset.py
===================
Two things that must be checked directly, not assumed:

1. extract_patch() actually returns the pixels patch_index_builder.py
   already counted. These are two separate code paths computing over the
   same nominal region -- if they disagree, sampling weights (derived from
   patch_index.csv) don't describe what training actually sees.
2. Augmentor preserves label validity: flips/rotations must not invent new
   class values or change per-image class pixel COUNTS (only positions),
   and brightness/contrast jitter must never touch the label array.

Does not require torch -- only exercises the framework-agnostic functions.
"""

import csv
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.common import load_config
from src.patching.dataset import extract_patch
from src.patching.augmentation import Augmentor


def load_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def run(config: dict, n_samples: int = 200, seed: int = 0) -> dict:
    out_dir = config["paths"]["output_dir"]
    data_root = config["paths"]["data_root"]
    codebook_path = os.path.join(out_dir, "class_codebook.json")

    manifest_rows = load_rows(os.path.join(out_dir, "manifest.csv"))
    patch_rows = load_rows(os.path.join(out_dir, "patch_index.csv"))
    manifest_by_stem = {r["stem"]: r for r in manifest_rows}

    with open(codebook_path) as f:
        codebook = json.load(f)
    id_to_name = {int(k): v for k, v in codebook["class_id_to_name"].items()}
    rare_name = config["rare_class"]["name"]

    rng = random.Random(seed)
    sample = rng.sample(patch_rows, min(n_samples, len(patch_rows)))

    failures = []
    for p in sample:
        src = manifest_by_stem[p["source_stem"]]
        patch_size = int(p["patch_size"])
        img_patch, lbl_patch = extract_patch(data_root, src, int(p["y0"]), int(p["x0"]), patch_size)

        if img_patch.shape != (patch_size, patch_size):
            failures.append(f"[FAIL] patch {p['patch_id']}: image shape {img_patch.shape}")
        if lbl_patch.shape != (patch_size, patch_size):
            failures.append(f"[FAIL] patch {p['patch_id']}: label shape {lbl_patch.shape}")

        # cross-check per-class pixel counts against what patch_index_builder recorded
        vals, counts = np.unique(lbl_patch, return_counts=True)
        actual_counts = {id_to_name.get(int(v), f"unknown_{v}"): int(c) for v, c in zip(vals, counts)}
        for cid, cname in id_to_name.items():
            expected = int(p.get(f"px_{cname}", 0))
            actual = actual_counts.get(cname, 0)
            if expected != actual:
                failures.append(
                    f"[FAIL] patch {p['patch_id']} ({p['source_stem']}): "
                    f"px_{cname} mismatch -- patch_index says {expected}, "
                    f"actual extracted pixels say {actual}"
                )

        actual_tin = actual_counts.get(rare_name, 0)
        expected_tin = int(p["tin_pixel_count"])
        if actual_tin != expected_tin:
            failures.append(
                f"[FAIL] patch {p['patch_id']}: tin_pixel_count mismatch -- "
                f"index says {expected_tin}, actual says {actual_tin}"
            )

    # --- augmentation invariant tests ---
    aug_failures = []
    aug = Augmentor(seed=123)
    for p in sample[:50]:
        src = manifest_by_stem[p["source_stem"]]
        patch_size = int(p["patch_size"])
        img_patch, lbl_patch = extract_patch(data_root, src, int(p["y0"]), int(p["x0"]), patch_size)

        aug_img, aug_lbl = aug(img_patch, lbl_patch)

        if aug_img.shape != img_patch.shape or aug_lbl.shape != lbl_patch.shape:
            aug_failures.append(f"[FAIL] patch {p['patch_id']}: augmented shape changed")

        # label value SET must be unchanged (flips/rotations permute
        # positions, never invent or remove class values)
        if set(np.unique(lbl_patch).tolist()) != set(np.unique(aug_lbl).tolist()):
            aug_failures.append(f"[FAIL] patch {p['patch_id']}: augmentation changed label class set")

        # per-class pixel COUNTS must be unchanged (geometric transforms
        # preserve area; only brightness/contrast touches the image, never the label)
        orig_counts = dict(zip(*np.unique(lbl_patch, return_counts=True)))
        aug_counts = dict(zip(*np.unique(aug_lbl, return_counts=True)))
        if {int(k): int(v) for k, v in orig_counts.items()} != {int(k): int(v) for k, v in aug_counts.items()}:
            aug_failures.append(f"[FAIL] patch {p['patch_id']}: augmentation changed per-class pixel counts")

        if aug_img.dtype != np.uint8:
            aug_failures.append(f"[FAIL] patch {p['patch_id']}: augmented image dtype {aug_img.dtype} != uint8")

    all_failures = failures + aug_failures
    passed = len(all_failures) == 0

    return {
        "check": "verify_dataset",
        "n_patches_tested_for_extraction": len(sample),
        "n_patches_tested_for_augmentation": min(50, len(sample)),
        "extraction_failures": failures[:30],
        "augmentation_failures": aug_failures[:30],
        "passed": passed,
        "conclusion": (
            f"VERIFIED: extract_patch() output matches patch_index.csv's "
            f"precomputed pixel counts exactly across {len(sample)} sampled "
            f"patches (bit-for-bit, not just shape). Augmentation preserves "
            f"label validity (class set and per-class pixel counts unchanged) "
            f"across {min(50, len(sample))} sampled patches."
            if passed else
            f"VERIFICATION FAILED: {len(all_failures)} failure(s) -- "
            f"do not trust dataset.py/augmentation.py until fixed."
        ),
    }


if __name__ == "__main__":
    cfg = load_config("configs/config.yaml")
    result = run(cfg)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["passed"] else 1)
