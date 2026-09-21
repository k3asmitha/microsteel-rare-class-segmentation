"""
validate_patch_index.py
=========================
Checks patch_index.csv for internal consistency AND cross-checks its total
patch count against tin_speck_analysis's fragmentation simulation for the
same (patch_size, stride) -- these were computed by two different code
paths (simulate_fragmentation's grid vs patch_index_builder's grid), so if
they disagree, the grid.py refactor didn't actually unify them and the 0%
fragmentation guarantee doesn't describe what's really being trained on.
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
    patch_size = config["patching"]["patch_size"]
    stride = config["patching"]["stride"]

    patch_index_path = os.path.join(out_dir, "patch_index.csv")
    manifest_path = os.path.join(out_dir, "manifest.csv")
    audit_findings_path = os.path.join(out_dir, "audit_findings.json")
    codebook_path = os.path.join(out_dir, "class_codebook.json")

    patches = load_rows(patch_index_path)
    manifest_rows = load_rows(manifest_path)
    with open(codebook_path) as f:
        codebook = json.load(f)
    class_names = list(codebook["class_id_to_name"].values())

    failures = []
    warnings = []

    manifest_stems = {r["stem"]: r for r in manifest_rows}
    seen_patch_ids = set()
    stems_covered = set()

    for p in patches:
        pid = p["patch_id"]
        if pid in seen_patch_ids:
            failures.append(f"[FAIL] duplicate patch_id {pid}")
        seen_patch_ids.add(pid)

        stem = p["source_stem"]
        stems_covered.add(stem)
        if stem not in manifest_stems:
            failures.append(f"[FAIL] patch {pid}: source_stem {stem} not in manifest")
            continue

        src = manifest_stems[stem]

        # bbox within bounds
        y0, x0, ps = int(p["y0"]), int(p["x0"]), int(p["patch_size"])
        H, W = int(src["crop_target_height"]), int(src["crop_target_width"])
        if y0 < 0 or x0 < 0 or y0 + ps > H or x0 + ps > W:
            failures.append(f"[FAIL] patch {pid}: bbox out of bounds "
                             f"(y0={y0},x0={x0},size={ps}, image={H}x{W})")

        # pixel accounting
        total = int(p["total_pixels"])
        if total != ps * ps:
            failures.append(f"[FAIL] patch {pid}: total_pixels {total} != patch_size^2 {ps*ps}")
        px_sum = sum(int(p[f"px_{c}"]) for c in class_names)
        if px_sum != total:
            failures.append(f"[FAIL] patch {pid}: class pixel sum {px_sum} != total {total}")

        # tin flag consistency
        tin_flag = p["tin_present"] in ("True", "true", "1")
        if tin_flag != (int(p["tin_pixel_count"]) > 0):
            failures.append(f"[FAIL] patch {pid}: tin_present inconsistent with pixel count")

        # inherited fields must match source image
        if p["true_type"] != src["true_type"]:
            failures.append(f"[FAIL] patch {pid}: true_type mismatch with source image")
        if p["split"] != src["split"]:
            failures.append(f"[FAIL] patch {pid}: split mismatch with source image")

    # every manifest image should be covered by at least one patch
    missing_coverage = set(manifest_stems.keys()) - stems_covered
    if missing_coverage:
        failures.append(f"[FAIL] {len(missing_coverage)} manifest image(s) have "
                         f"ZERO patches: {sorted(missing_coverage)[:10]}")

    # --- cross-check total patch count against the fragmentation simulation ---
    cross_check_result = None
    if os.path.exists(audit_findings_path):
        with open(audit_findings_path) as f:
            findings = json.load(f)
        sim = findings.get("tin_speck_analysis", {}).get("fragmentation_simulation", [])
        matching = [s for s in sim if s["patch_size"] == patch_size and s["stride"] == stride]
        if matching:
            expected_total = matching[0]["total_patches_full_dataset"]
            actual_total = len(patches)
            cross_check_result = {
                "expected_from_simulation": expected_total,
                "actual_from_patch_index": actual_total,
                "match": expected_total == actual_total,
            }
            if expected_total != actual_total:
                failures.append(
                    f"[FAIL] patch count MISMATCH between fragmentation simulation "
                    f"({expected_total}) and actual patch_index ({actual_total}) for "
                    f"patch_size={patch_size}, stride={stride}. The two grid "
                    f"implementations have drifted -- this must be fixed, not "
                    f"suppressed, before trusting the 0% fragmentation result."
                )
        else:
            warnings.append(
                f"[WARN] no matching simulation entry for patch_size={patch_size}, "
                f"stride={stride} in audit_findings.json -- cannot cross-check."
            )

    passed = len(failures) == 0
    if passed and cross_check_result:
        cross_check_note = (f"Patch count also MATCHES the fragmentation "
                             f"simulation's independent prediction "
                             f"({cross_check_result['expected_from_simulation']}) -- "
                             f"the two separately-computed grids agree.")
    elif passed:
        cross_check_note = ("Cross-check against the fragmentation simulation "
                             "could NOT be performed (no matching entry found in "
                             "audit_findings.json) -- see warnings. Structural "
                             "checks passed, but the grid-agreement guarantee is unverified.")
    else:
        cross_check_note = ""

    return {
        "check": "validate_patch_index",
        "n_patches_checked": len(patches),
        "n_source_images_covered": len(stems_covered),
        "n_source_images_in_manifest": len(manifest_stems),
        "cross_check_vs_fragmentation_simulation": cross_check_result,
        "passed": passed,
        "failures": failures[:50],
        "n_failures_total": len(failures),
        "warnings": warnings,
        "conclusion": (
            f"VALIDATION PASSED: {len(patches)} patches checked, all structurally "
            f"consistent with manifest. {cross_check_note}"
            if passed else
            f"VALIDATION FAILED: {len(failures)} hard failure(s). Do not train "
            f"on this patch_index.csv until fixed."
        ),
    }


if __name__ == "__main__":
    cfg = load_config("configs/config.yaml")
    result = validate(cfg)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["passed"] else 1)
