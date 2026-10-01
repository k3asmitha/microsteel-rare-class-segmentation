"""
verify_reconstruct.py
=======================
1. GROUND-TRUTH EXACT RECOVERY on genuine TiN-containing images, one per
   available processing type (I, II, III) -- not an arbitrary first image,
   which previously happened to contain no TiN at all. TiN is the entire
   reason this evaluation pipeline exists and its features are the
   smallest/most boundary-sensitive in the dataset, so it's the case most
   likely to reveal an off-by-one or coverage bug. Take a real label
   array, slice it into the actual overlapping patches from
   patch_index.csv, one-hot encode each patch as a "perfect model"
   prediction, reconstruct, and confirm the result is bit-for-bit identical
   to the original label array.
2. NOISE AVERAGING: perturb a fraction of pixels per patch independently
   (different noise per overlapping patch), and verify reconstruction
   recovers the true label with higher accuracy than the MEAN accuracy of
   the individual noisy patches (not "any single patch" / not the max --
   that would be a stronger and unverified claim; this test only measures
   improvement over the mean).
3. COVERAGE FAILURE: deliberately omit one patch's prediction and confirm
   reconstruct_prediction raises rather than silently producing a
   zero-filled or garbage region.
4. INVALID-PROBABILITY REJECTION: pass an array that doesn't sum to 1.0
   along the class axis (e.g. raw un-normalized scores) and confirm
   reconstruct_prediction raises ValueError rather than silently averaging
   nonsense and returning a plausible-looking wrong answer.
"""

import csv
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.common import load_config
from src.eval.reconstruct import (
    get_patch_coords_for_stem, reconstruct_prediction, one_hot_from_labels,
)


def check(name, condition, failures, detail=""):
    if not condition:
        failures.append(f"[FAIL] {name}" + (f" -- {detail}" if detail else ""))


def load_rows(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def find_one_tin_image_per_type(manifest_rows, patch_rows):
    """Deterministically picks one genuine TiN-containing image per
    available processing type, rather than an arbitrary first image that
    might not contain TiN at all."""
    tin_stems = sorted({r["source_stem"] for r in patch_rows
                         if r["tin_present"] in ("True", "true", "1")})
    manifest_by_stem = {r["stem"]: r for r in manifest_rows}
    picked = {}
    for stem in tin_stems:
        t = manifest_by_stem[stem]["true_type"]
        if t not in picked:
            picked[t] = stem
    return picked  # {"I": stem, "II": stem, "III": stem} -- only types that have any TiN-containing image


def run(config: dict) -> dict:
    failures = []
    out_dir = config["paths"]["output_dir"]
    data_root = config["paths"]["data_root"]
    num_classes = 5

    manifest_rows = load_rows(os.path.join(out_dir, "manifest.csv"))
    patch_rows = load_rows(os.path.join(out_dir, "patch_index.csv"))
    if not manifest_rows or not patch_rows:
        return {"check": "verify_reconstruct", "skipped": "manifest.csv/patch_index.csv not found"}

    manifest_by_stem = {r["stem"]: r for r in manifest_rows}
    tin_image_per_type = find_one_tin_image_per_type(manifest_rows, patch_rows)
    check("test setup: found at least one TiN-containing image",
          len(tin_image_per_type) > 0, failures)

    per_type_results = {}

    for true_type, stem in tin_image_per_type.items():
        row = manifest_by_stem[stem]
        label = np.array(Image.open(os.path.join(data_root, row["label_path"])))
        H, W = label.shape
        coords = get_patch_coords_for_stem(patch_rows, stem)
        check(f"type {true_type}: patches found for {stem}", len(coords) > 0, failures)

        # --- Test 1: exact recovery from perfect predictions, on a REAL TiN image ---
        patch_probs_perfect = {}
        for (y0, x0, ps) in coords:
            patch_label = label[y0:y0 + ps, x0:x0 + ps]
            patch_probs_perfect[(y0, x0)] = one_hot_from_labels(patch_label, num_classes)

        reconstructed = reconstruct_prediction(patch_probs_perfect, coords, (H, W), num_classes)
        n_diff = int((reconstructed != label).sum())
        check(f"type {true_type} ({stem}): exact recovery, TiN-containing image",
              n_diff == 0, failures, f"{n_diff} pixels differ out of {label.size}")

        # specifically confirm TiN pixels themselves are recovered exactly,
        # not just "overall" pixel count (a large majority-class match could
        # mask a small TiN region being wrong)
        tin_mask = (label == config["rare_class"]["id"])
        tin_diff = int((reconstructed[tin_mask] != label[tin_mask]).sum())
        check(f"type {true_type} ({stem}): TiN pixels specifically recovered exactly",
              tin_diff == 0, failures, f"{tin_diff} TiN pixels wrong out of {int(tin_mask.sum())}")

        per_type_results[true_type] = {"stem": stem, "pixels_differing": n_diff, "tin_pixels_differing": tin_diff}

    # --- Test 2: noise averaging (run once, on the first available TiN image) ---
    first_type = sorted(tin_image_per_type.keys())[0]
    stem = tin_image_per_type[first_type]
    row = manifest_by_stem[stem]
    label = np.array(Image.open(os.path.join(data_root, row["label_path"])))
    H, W = label.shape
    coords = get_patch_coords_for_stem(patch_rows, stem)

    rng = np.random.RandomState(42)
    noise_rate = 0.15
    patch_probs_noisy = {}
    single_patch_accuracies = []
    for (y0, x0, ps) in coords:
        patch_label = label[y0:y0 + ps, x0:x0 + ps]
        noisy_label = patch_label.copy()
        mask = rng.rand(ps, ps) < noise_rate
        random_classes = rng.randint(0, num_classes, size=(ps, ps))
        noisy_label[mask] = random_classes[mask]
        patch_probs_noisy[(y0, x0)] = one_hot_from_labels(noisy_label, num_classes)
        single_patch_accuracies.append(float((noisy_label == patch_label).mean()))

    reconstructed_noisy = reconstruct_prediction(patch_probs_noisy, coords, (H, W), num_classes)
    reconstructed_accuracy = float((reconstructed_noisy == label).mean())
    mean_single_patch_accuracy = float(np.mean(single_patch_accuracies))

    check("noise averaging: reconstructed accuracy beats MEAN single-patch accuracy "
          "(this test does NOT claim it beats every individual patch, only the mean)",
          reconstructed_accuracy > mean_single_patch_accuracy, failures,
          f"reconstructed={reconstructed_accuracy:.4f}, mean_single_patch={mean_single_patch_accuracy:.4f}")

    # --- Test 3: missing patch prediction must raise ---
    incomplete_probs = dict(patch_probs_noisy)
    removed_key = coords[0][:2]
    del incomplete_probs[removed_key]
    raised = False
    try:
        reconstruct_prediction(incomplete_probs, coords, (H, W), num_classes)
    except KeyError:
        raised = True
    check("missing patch prediction raises KeyError", raised, failures)

    # --- Test 4: invalid (non-probability) input must raise -- specifically
    #     isolate the SUM-check by scaling DOWN (0.5x keeps every value
    #     within [0,1] individually, so the range check does NOT fire first;
    #     only the sum-to-1 check catches this case) ---
    bad_probs_full = dict(patch_probs_noisy)
    bad_key = coords[0][:2]
    bad_probs_full[bad_key] = bad_probs_full[bad_key] * 0.5  # in-range but sums to 0.5, not 1.0
    raised_invalid = False
    try:
        reconstruct_prediction(bad_probs_full, coords, (H, W), num_classes)
    except ValueError as e:
        raised_invalid = "does not sum to 1.0" in str(e)
    check("probabilities that don't sum to 1.0 (but stay in-range) raise ValueError, "
          "not silently averaged", raised_invalid, failures)

    # --- Test 5: NaN/Inf rejected ---
    nan_probs_full = dict(patch_probs_noisy)
    nan_key = coords[0][:2]
    corrupted = nan_probs_full[nan_key].copy()
    corrupted[0, 0, 0] = np.nan
    nan_probs_full[nan_key] = corrupted
    raised_nan = False
    try:
        reconstruct_prediction(nan_probs_full, coords, (H, W), num_classes)
    except ValueError as e:
        raised_nan = "NaN/Inf" in str(e)
    check("NaN in probability array raises ValueError", raised_nan, failures)

    # --- Test 6: out-of-[0,1]-range values rejected even if they happen to sum to 1 ---
    range_probs_full = dict(patch_probs_noisy)
    range_key = coords[1][:2]
    corrupted2 = range_probs_full[range_key].copy()
    corrupted2[0] += 1.5  # push class 0 above 1.0
    corrupted2[1] -= 1.5  # compensate so it still sums to ~1.0 elementwise, but now has a negative value
    range_probs_full[range_key] = corrupted2
    raised_range = False
    try:
        reconstruct_prediction(range_probs_full, coords, (H, W), num_classes)
    except ValueError as e:
        raised_range = "outside [0,1]" in str(e)
    check("values outside [0,1] rejected even when they still sum to ~1.0", raised_range, failures)

    passed = len(failures) == 0
    return {
        "check": "verify_reconstruct",
        "passed": passed,
        "failures": failures,
        "tin_images_tested_per_type": {t: v["stem"] for t, v in per_type_results.items()},
        "per_type_results": per_type_results,
        "noise_test": {
            "reconstructed_accuracy": round(reconstructed_accuracy, 4),
            "mean_single_patch_accuracy": round(mean_single_patch_accuracy, 4),
        },
        "conclusion": (
            f"VERIFIED on genuine TiN-containing images across "
            f"{len(per_type_results)} type(s) ({sorted(per_type_results.keys())}): "
            f"exact recovery (0 differing pixels, TiN pixels specifically "
            f"checked, not just overall count), overlap-averaging beats mean "
            f"single-patch accuracy under noise "
            f"({reconstructed_accuracy:.4f} vs {mean_single_patch_accuracy:.4f}), "
            f"missing-patch coverage failures raise loudly, and non-probability "
            f"inputs (raw logits/scores) are rejected rather than silently "
            f"averaged into a meaningless result."
            if passed else
            f"VERIFICATION FAILED: {len(failures)} failure(s) in reconstruct.py."
        ),
    }


if __name__ == "__main__":
    cfg = load_config("configs/config.yaml")
    result = run(cfg)
    import json
    print(json.dumps(result, indent=2, default=str))
    sys.exit(0 if result.get("passed", False) else 1)
