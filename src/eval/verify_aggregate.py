"""
verify_aggregate.py
=====================
Hand-constructed scenario testing aggregate.py against the exact failure
mode flagged in review: a false-positive-only case (IoU=0.0/F1=0.0, not
None) must be INCLUDED in the mean, not filtered out alongside the truly
undefined absent_from_both cases.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.common import load_config
from src.eval.aggregate import aggregate_iou_or_f1, aggregate_hd95


def check(name, condition, failures, detail=""):
    if not condition:
        failures.append(f"[FAIL] {name}" + (f" -- {detail}" if detail else ""))


def run(config=None) -> dict:
    failures = []

    # 5 images' worth of class-1 IoU values:
    #   img1: both_present, IoU=0.8
    #   img2: both_present, IoU=0.6
    #   img3: missed_entirely -> IoU=0.0 (per metrics.py contract)
    #   img4: false_positive_only -> IoU=0.0 (per metrics.py contract)
    #   img5: absent_from_both -> None
    per_image = [
        {1: 0.8}, {1: 0.6}, {1: 0.0}, {1: 0.0}, {1: None},
    ]
    agg = aggregate_iou_or_f1(per_image, num_classes=2)

    # mean must be (0.8+0.6+0.0+0.0)/4 = 0.35, over n_included=4 (NOT
    # filtered down to just the 2 both_present cases, which would wrongly
    # give mean=0.7 and hide the two zero-score failures)
    expected_mean = (0.8 + 0.6 + 0.0 + 0.0) / 4
    check("aggregate IoU mean includes false-positive-only zero",
          abs(agg[1]["mean"] - expected_mean) < 1e-9, failures,
          f"expected {expected_mean}, got {agg[1]['mean']}")
    check("aggregate IoU n_included is 4 (not 2)",
          agg[1]["n_included"] == 4, failures, f"got {agg[1]['n_included']}")
    check("aggregate IoU n_excluded_absent_from_both is 1",
          agg[1]["n_excluded_absent_from_both"] == 1, failures, f"got {agg[1]['n_excluded_absent_from_both']}")

    # If someone "fixes" this by filtering to GT-present-only, mean would
    # wrongly become 0.7 (hiding the false-positive-only failure) -- assert
    # the mean is NOT that wrong value, as an explicit anti-regression check
    wrong_mean_if_filtered_to_gt_present = (0.8 + 0.6 + 0.0) / 3  # includes missed_entirely (GT present) but drops false_positive_only
    check("aggregate IoU mean is NOT the 'filtered to GT-present' wrong value",
          abs(agg[1]["mean"] - wrong_mean_if_filtered_to_gt_present) > 1e-9, failures)

    # --- HD95 aggregation: computed values averaged separately from counts ---
    per_image_hd = [{1: 4.2}, {1: 5.8}, {1: None}, {1: None}, {1: None}]
    per_image_reasons = [
        {1: "both_present"}, {1: "both_present"},
        {1: "missed_entirely"}, {1: "false_positive_only"}, {1: "absent_from_both"},
    ]
    hd_agg = aggregate_hd95(per_image_hd, per_image_reasons, num_classes=2)
    check("aggregate HD95 mean_computed uses only both_present values",
          abs(hd_agg[1]["mean_computed"] - 5.0) < 1e-9, failures, f"got {hd_agg[1]['mean_computed']}")
    check("aggregate HD95 n_computed == 2", hd_agg[1]["n_computed"] == 2, failures)
    check("aggregate HD95 n_missed_entirely == 1", hd_agg[1]["n_missed_entirely"] == 1, failures)
    check("aggregate HD95 n_false_positive_only == 1", hd_agg[1]["n_false_positive_only"] == 1, failures)
    check("aggregate HD95 n_absent_from_both == 1", hd_agg[1]["n_absent_from_both"] == 1, failures)

    passed = len(failures) == 0
    return {
        "check": "verify_aggregate",
        "passed": passed,
        "failures": failures,
        "conclusion": (
            "VERIFIED: aggregate.py correctly includes false-positive-only "
            "zeros in IoU/F1 means (does not silently filter to GT-present "
            "only), and HD95 correctly separates computed-value means from "
            "missed/false-positive/absent counts."
            if passed else
            f"VERIFICATION FAILED: {len(failures)} failure(s) in aggregate.py."
        ),
    }


if __name__ == "__main__":
    cfg = load_config("configs/config.yaml")
    result = run(cfg)
    import json
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["passed"] else 1)
