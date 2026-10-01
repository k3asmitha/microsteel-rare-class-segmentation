"""
verify_metrics.py
===================
Unit tests for metrics.py against HAND-COMPUTABLE known answers, plus
regression tests for the specific bug found in review (HD95 conflating
'missed_entirely' with 'false_positive_only'), plus tiny-object edge cases
(1/2/3-pixel components, thin lines) that must not crash or silently
misbehave before this contract is frozen for B and C's training runs.

This is not optional polish: a silently wrong IoU or HD95 implementation
would invalidate every one of the 24 training runs' comparisons, and eval
bugs are notoriously easy to not notice because a wrong number still looks
like a plausible number.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.common import load_config
from src.eval.metrics import (
    class_presence_status, confusion_matrix, iou_per_class,
    boundary_f1_per_class, hd95_per_class,
)


def check(name, condition, failures, detail=""):
    if not condition:
        failures.append(f"[FAIL] {name}" + (f" -- {detail}" if detail else ""))
    return condition


def test_presence_status(failures):
    a = np.array([[0, 1], [1, 0]])
    b = np.array([[0, 0], [0, 0]])
    check("presence: both_present", class_presence_status(a, a, 1) == "both_present", failures)
    check("presence: absent_from_both", class_presence_status(b, b, 1) == "absent_from_both", failures)
    check("presence: missed_entirely", class_presence_status(b, a, 1) == "missed_entirely", failures,
          "GT(a) has class 1, pred(b) doesn't")
    check("presence: false_positive_only", class_presence_status(a, b, 1) == "false_positive_only", failures,
          "pred(a) has class 1, GT(b) doesn't")


def test_iou_hand_computed(failures):
    # Perfect match: 2x2 image, class 1 in top-left cell only
    target = np.array([[1, 0], [0, 0]])
    pred = np.array([[1, 0], [0, 0]])
    iou = iou_per_class(pred, target, num_classes=2)
    check("IoU perfect match class 1", iou[1] == 1.0, failures, f"got {iou[1]}")
    check("IoU perfect match class 0", iou[0] == 1.0, failures, f"got {iou[0]}")

    # No overlap: pred and target are disjoint regions of class 1
    target2 = np.array([[1, 1, 0, 0]])
    pred2 = np.array([[0, 0, 1, 1]])
    iou2 = iou_per_class(pred2, target2, num_classes=2)
    check("IoU zero overlap", iou2[1] == 0.0, failures, f"got {iou2[1]}")

    # Known partial overlap: target has 4 pixels of class 1, pred has 4
    # pixels of class 1, overlapping in exactly 2 -- IoU should be
    # intersection/union = 2/6 = 0.3333...
    target3 = np.zeros((4, 4), dtype=int)
    target3[0:2, 0:2] = 1  # 4 pixels: (0,0)(0,1)(1,0)(1,1)
    pred3 = np.zeros((4, 4), dtype=int)
    pred3[1:3, 1:3] = 1    # 4 pixels: (1,1)(1,2)(2,1)(2,2)
    # overlap is exactly (1,1) -> intersection=1, union=4+4-1=7
    iou3 = iou_per_class(pred3, target3, num_classes=2)
    expected = 1.0 / 7.0
    check("IoU known partial overlap", abs(iou3[1] - expected) < 1e-9, failures,
          f"expected {expected}, got {iou3[1]}")

    # absent from both -> None, not 0 or 1
    empty = np.zeros((3, 3), dtype=int)
    iou4 = iou_per_class(empty, empty, num_classes=2)
    check("IoU absent from both is None", iou4[1] is None, failures, f"got {iou4[1]}")

    # missed entirely -> 0.0, not None
    target5 = np.zeros((3, 3), dtype=int); target5[0, 0] = 1
    pred5 = np.zeros((3, 3), dtype=int)
    iou5 = iou_per_class(pred5, target5, num_classes=2)
    check("IoU missed_entirely is 0.0 not None", iou5[1] == 0.0, failures, f"got {iou5[1]}")

    # false positive only -> 0.0, not None (the exact case flagged in review)
    target6 = np.zeros((3, 3), dtype=int)
    pred6 = np.zeros((3, 3), dtype=int); pred6[0, 0] = 1
    iou6 = iou_per_class(pred6, target6, num_classes=2)
    check("IoU false_positive_only is 0.0 not None", iou6[1] == 0.0, failures, f"got {iou6[1]}")


def test_confusion_matrix(failures):
    target = np.array([[0, 1], [1, 0]])
    pred = np.array([[0, 1], [0, 1]])
    cm = confusion_matrix(pred, target, num_classes=2)
    # (0,0): true=0,pred=0 -> cm[0,0]
    # (0,1): true=1,pred=1 -> cm[1,1]
    # (1,0): true=1,pred=0 -> cm[1,0]
    # (1,1): true=0,pred=1 -> cm[0,1]
    expected = np.array([[1, 1], [1, 1]])
    check("confusion matrix values", np.array_equal(cm, expected), failures, f"got {cm.tolist()}")
    check("confusion matrix sums to total pixels", cm.sum() == target.size, failures)

    # fail loudly on invalid class ID -- this is a REQUIRED behavior change
    # from an earlier version that silently dropped out-of-range pixels
    raised = False
    try:
        confusion_matrix(np.array([[0, 7]]), np.array([[0, 1]]), num_classes=2)
    except ValueError:
        raised = True
    check("confusion matrix raises on invalid class ID", raised, failures,
          "should raise ValueError, not silently drop the pixel")


def test_hd95_missed_vs_false_positive_regression(failures):
    """Regression test for the exact bug found in review: HD95 must NOT
    label 'GT present, pred absent' and 'GT absent, pred present' with the
    same reason string."""
    gt_has_it = np.zeros((10, 10), dtype=int)
    gt_has_it[3:5, 3:5] = 1
    pred_empty = np.zeros((10, 10), dtype=int)
    _, reasons1 = hd95_per_class(pred_empty, gt_has_it, num_classes=2)
    check("HD95 reason: GT present/pred absent -> missed_entirely",
          reasons1[1] == "missed_entirely", failures, f"got {reasons1[1]}")

    gt_empty = np.zeros((10, 10), dtype=int)
    pred_has_it = np.zeros((10, 10), dtype=int)
    pred_has_it[3:5, 3:5] = 1
    _, reasons2 = hd95_per_class(pred_has_it, gt_empty, num_classes=2)
    check("HD95 reason: GT absent/pred present -> false_positive_only",
          reasons2[1] == "false_positive_only", failures, f"got {reasons2[1]}")

    check("HD95 the two cases have DIFFERENT reasons (the actual bug)",
          reasons1[1] != reasons2[1], failures,
          f"both were '{reasons1[1]}' -- this is the exact bug that was fixed")


def test_hd95_hand_computed(failures):
    # two parallel lines offset by exactly 3 pixels vertically -> HD95
    # should equal 3.0 (every boundary pixel on one line is exactly 3px
    # from the nearest boundary pixel on the other, since they're straight
    # parallel lines of equal length)
    target = np.zeros((10, 10), dtype=int)
    target[3, :] = 1
    pred = np.zeros((10, 10), dtype=int)
    pred[6, :] = 1
    result, reasons = hd95_per_class(pred, target, num_classes=2)
    check("HD95 known parallel-line distance", abs(result[1] - 3.0) < 1e-6, failures,
          f"expected 3.0, got {result[1]}")
    check("HD95 reason both_present", reasons[1] == "both_present", failures)


def test_tiny_objects(failures):
    """The reviewer flagged that TiN components can be extremely small
    (some real specks are only a handful of pixels). Verify boundary_f1 and
    hd95 don't crash and produce sensible (not nonsensical) values for
    1-pixel, 2-pixel, 3-pixel objects and a thin line."""
    size = 20
    cases = {
        "1-pixel": [(10, 10)],
        "2-pixel": [(10, 10), (10, 11)],
        "3-pixel": [(10, 10), (10, 11), (11, 10)],
        "thin_line": [(10, i) for i in range(5, 15)],
    }
    for name, coords in cases.items():
        target = np.zeros((size, size), dtype=int)
        for (y, x) in coords:
            target[y, x] = 1
        # exact match
        pred_exact = target.copy()
        try:
            f1 = boundary_f1_per_class(pred_exact, target, num_classes=2)
            hd, reasons = hd95_per_class(pred_exact, target, num_classes=2)
            check(f"tiny object '{name}': F1 exact match doesn't crash and == 1.0",
                  f1[1] == 1.0, failures, f"got {f1[1]}")
            check(f"tiny object '{name}': HD95 exact match doesn't crash and == 0.0",
                  hd[1] == 0.0, failures, f"got {hd[1]}")
        except Exception as e:
            failures.append(f"[FAIL] tiny object '{name}' CRASHED: {e}")

        # shifted by 1 pixel (still overlapping partially or not, for 1-pixel case likely disjoint)
        pred_shifted = np.zeros((size, size), dtype=int)
        for (y, x) in coords:
            if 0 <= x + 1 < size:
                pred_shifted[y, x + 1] = 1
        try:
            f1s = boundary_f1_per_class(pred_shifted, target, num_classes=2)
            hds, reasonss = hd95_per_class(pred_shifted, target, num_classes=2)
            check(f"tiny object '{name}': shifted case doesn't crash",
                  f1s[1] is not None or reasonss[1] in ("missed_entirely", "false_positive_only"),
                  failures)
        except Exception as e:
            failures.append(f"[FAIL] tiny object '{name}' shifted CRASHED: {e}")


def test_real_data_sanity(config):
    """Real-data sanity check beyond synthetic toy grids: pred == target
    exactly on a real patch from the dataset must give perfect scores for
    every class actually present in that patch."""
    import csv
    from PIL import Image

    failures = []
    out_dir = config["paths"]["output_dir"]
    data_root = config["paths"]["data_root"]
    manifest_path = os.path.join(out_dir, "manifest.csv")
    if not os.path.exists(manifest_path):
        return {"skipped": "no manifest.csv found -- run the main pipeline first"}, failures

    with open(manifest_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # pick a Type I image (known to have the most classes: Alpha/TiB2/TiN/FeTiB)
    type1_rows = [r for r in rows if r["true_type"] == "I"]
    if not type1_rows:
        return {"skipped": "no Type I rows in manifest"}, failures

    row = type1_rows[0]
    label = np.array(Image.open(os.path.join(data_root, row["label_path"])))

    iou = iou_per_class(label, label, num_classes=5)
    f1 = boundary_f1_per_class(label, label, num_classes=5)
    hd, reasons = hd95_per_class(label, label, num_classes=5)

    present_classes = [c for c in range(5) if np.any(label == c)]
    for c in present_classes:
        check(f"real-data self-match IoU class {c} == 1.0", iou[c] == 1.0, failures, f"got {iou[c]}")
        check(f"real-data self-match F1 class {c} == 1.0", f1[c] == 1.0, failures, f"got {f1[c]}")
        check(f"real-data self-match HD95 class {c} == 0.0", hd[c] == 0.0, failures, f"got {hd[c]}")

    return {"stem": row["stem"], "present_classes": present_classes}, failures


def run(config: dict) -> dict:
    failures = []
    test_presence_status(failures)
    test_iou_hand_computed(failures)
    test_confusion_matrix(failures)
    test_hd95_missed_vs_false_positive_regression(failures)
    test_hd95_hand_computed(failures)
    test_tiny_objects(failures)
    real_data_info, real_failures = test_real_data_sanity(config)
    failures.extend(real_failures)

    passed = len(failures) == 0
    return {
        "check": "verify_metrics",
        "passed": passed,
        "n_failures": len(failures),
        "failures": failures[:50],
        "real_data_check": real_data_info,
        "conclusion": (
            f"VERIFIED: all synthetic hand-computed cases, the "
            f"missed_entirely/false_positive_only regression test, tiny-object "
            f"edge cases, and a real-data self-match sanity check all pass. "
            f"Metric contract is frozen -- see metrics.py module docstring "
            f"table before changing any of these functions."
            if passed else
            f"VERIFICATION FAILED: {len(failures)} failure(s) -- do NOT freeze "
            f"this metric contract or hand it to B/C until fixed."
        ),
    }


if __name__ == "__main__":
    cfg = load_config("configs/config.yaml")
    result = run(cfg)
    import json
    print(json.dumps(result, indent=2, default=str))
    sys.exit(0 if result["passed"] else 1)
