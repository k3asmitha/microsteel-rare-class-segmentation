"""
validate_manifest.py
======================
Run this every time manifest.csv is regenerated, before anyone trains on it.
Checks are derived from config.yaml (expected_type_class_membership,
expected_counts_per_paper), not from separately hardcoded numbers.
"""

import csv
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.common import load_config


def load_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def validate(config: dict) -> dict:
    out_dir = config["paths"]["output_dir"]
    manifest_path = os.path.join(out_dir, "manifest.csv")
    excluded_path = os.path.join(out_dir, "excluded_labels.csv")
    codebook_path = os.path.join(out_dir, "class_codebook.json")

    rows = load_rows(manifest_path)
    with open(codebook_path) as f:
        codebook = json.load(f)
    class_names = list(codebook["class_id_to_name"].values())
    allowed_by_type = config["expected_type_class_membership"]

    failures = []
    warnings = []
    seen_stems = set()

    for r in rows:
        stem = r["stem"]

        if stem in seen_stems:
            failures.append(f"[FAIL] {stem}: duplicate stem in manifest")
        seen_stems.add(stem)

        total = int(r["total_pixels"])
        px_sum = sum(int(r[f"px_{c}"]) for c in class_names)
        if px_sum != total:
            failures.append(f"[FAIL] {stem}: pixel sum {px_sum} != total {total}")

        if int(r["tin_pixel_count"]) != int(r.get(f"px_{config['rare_class']['name']}", -1)):
            failures.append(f"[FAIL] {stem}: tin_pixel_count inconsistent with px_{config['rare_class']['name']}")

        tin_flag = r["tin_present"] in ("True", "true", "1")
        if tin_flag != (int(r["tin_pixel_count"]) > 0):
            failures.append(f"[FAIL] {stem}: tin_present flag inconsistent with pixel count")

        allowed = set(allowed_by_type.get(r["true_type"], []))
        for c in class_names:
            if c not in allowed and int(r[f"px_{c}"]) > 0:
                failures.append(
                    f"[FAIL] {stem}: illegal class {c} found ({r[f'px_{c}']} px) "
                    f"for type {r['true_type']} (allowed: {sorted(allowed)})"
                )

        ow, oh = int(r["orig_width"]), int(r["orig_height"])
        cw, ch = int(r["crop_target_width"]), int(r["crop_target_height"])
        if ow != cw:
            failures.append(f"[FAIL] {stem}: crop changed width ({ow} -> {cw})")
        if ch > oh:
            failures.append(f"[FAIL] {stem}: crop height {ch} exceeds original {oh}")

        if r["split"] not in ("train", "val", "test"):
            failures.append(f"[FAIL] {stem}: invalid split assignment '{r['split']}'")

    # cross-check counts against paper's expected values (warning, not failure --
    # paper's numbers could legitimately be superseded by a corrected dataset)
    if os.path.exists(excluded_path):
        excluded_rows = load_rows(excluded_path)
        expected_total = config["expected_counts_per_paper"]["total_labels"]
        actual_total = len(rows) + len(excluded_rows)
        if actual_total != expected_total:
            warnings.append(
                f"[WARN] usable ({len(rows)}) + excluded ({len(excluded_rows)}) "
                f"= {actual_total}, but config expects {expected_total} total labels."
            )

    per_type_counts = {}
    for r in rows:
        per_type_counts[r["true_type"]] = per_type_counts.get(r["true_type"], 0) + 1
    for t, exp_n in config["expected_counts_per_paper"].items():
        if t == "total_labels":
            continue
        actual_n = per_type_counts.get(t, 0)
        if actual_n != exp_n:
            warnings.append(
                f"[WARN] Type {t}: manifest has {actual_n} usable images, "
                f"paper states {exp_n}. This is EXPECTED if orphan labels "
                f"were excluded -- verify against excluded_labels.csv, "
                f"don't just suppress this warning."
            )

    split_counts = {}
    for r in rows:
        split_counts[r["split"]] = split_counts.get(r["split"], 0) + 1

    passed = len(failures) == 0
    return {
        "check": "validate_manifest",
        "n_rows_checked": len(rows),
        "passed": passed,
        "failures": failures,
        "warnings": warnings,
        "per_type_counts": per_type_counts,
        "split_counts": split_counts,
        "conclusion": (
            f"VALIDATION PASSED: all {len(rows)} rows satisfy every invariant "
            f"(pixel accounting, class legality per type, crop consistency, "
            f"split validity)." if passed else
            f"VALIDATION FAILED: {len(failures)} hard failure(s) -- "
            f"manifest must NOT be used for training until fixed."
        ),
    }


if __name__ == "__main__":
    cfg = load_config("configs/config.yaml")
    result = validate(cfg)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["passed"] else 1)
