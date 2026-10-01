"""
metrics.py
===========
Core segmentation metrics: confusion matrix, per-class IoU, per-class
Boundary F1, per-class 95th-percentile Hausdorff Distance (HD95).

FROZEN METRIC CONTRACT (do not change without updating every caller and
re-running verify_metrics.py -- B and C will train against this contract,
so it must not shift mid-experiment):

    Situation                    | IoU  | Boundary F1 | HD95
    ------------------------------|------|-------------|------------------------
    GT absent,  Pred absent       | None | None        | None (absent_from_both)
    GT present, Pred present      | calc | calc        | calc
    GT present, Pred absent       | 0.0  | 0.0         | None (missed_entirely)
    GT absent,  Pred present      | 0.0  | 0.0         | None (false_positive_only)

Every metric function uses the SAME class_presence_status() helper for its
absent/present branching, specifically to prevent the three metrics from
silently disagreeing about what "undefined" means for the same pixel data
(IoU calling something undefined while HD95 calls it "missed" is exactly
the inconsistency that makes a 24-run comparison untrustworthy).

WHY "GT absent, Pred present" must NEVER collapse into "GT present, Pred
absent" (both previously mislabeled "missed_entirely" in an earlier version
of this file -- caught in review, verified by class_presence_status tests
in verify_metrics.py): these are opposite failure modes. A model that
hallucinates TiN everywhere and a model that never predicts TiN at all are
both bad, but for completely different reasons, and averaging their
statistics together (or discarding both as "undefined") hides which
failure mode is actually occurring.

AGGREGATION CONTRACT (binding on aggregate.py, stated here so it isn't
re-litigated per-caller): for IoU and Boundary F1, "GT absent, Pred absent"
is excluded from any mean (truly undefined). "GT absent, Pred present"
(IoU=0.0, F1=0.0) MUST be included in aggregate statistics -- it is real
false-positive signal, not something to filter out alongside the
GT-present subset. For HD95, "computed" values are averaged separately
from counts of missed_entirely / false_positive_only / absent_from_both --
never silently averaged together, since a missing HD95 is not a "0" or an
outlier to smooth over, it is a distinct failure category that must be
reported as a count, not folded into a mean.

BOUNDARY TOLERANCE: boundary_f1's boundary_width parameter is a SQUARE
morphological dilation radius (structuring element is a
(2w+1)x(2w+1) block of ones), not a Euclidean distance tolerance. A pixel
diagonally 2 pixels away in both x and y counts as "within tolerance" at
boundary_width=2 even though its Euclidean distance is ~2.83px.

USAGE NOTE FOR WHOEVER BUILDS THE FULL EVALUATOR (not yet built): these
functions are resolution-agnostic and operate on whatever 2D arrays they're
given. But because patches overlap 50% (see patch_index.csv), calling these
functions independently on raw overlapping training patches and averaging
the results would double/triple-count pixels in overlap regions and distort
every statistic. The evaluator MUST reconstruct full images via
sliding-window inference (e.g. averaging logits in overlap regions) BEFORE
calling these functions -- evaluate on reconstructed full images, never on
raw overlapping patches directly.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt, binary_dilation
from skimage.segmentation import find_boundaries


def class_presence_status(pred: np.ndarray, target: np.ndarray, class_id: int) -> str:
    """The ONE shared source of truth for absent/present branching, used by
    every metric below. Returns one of:
      'both_present'        -- class appears in both -> compute the metric
      'missed_entirely'      -- GT has it, prediction has none of it
      'false_positive_only'  -- prediction has it, GT has none of it
      'absent_from_both'     -- neither has it -> undefined, exclude from means
    """
    pred_present = bool(np.any(pred == class_id))
    target_present = bool(np.any(target == class_id))
    if target_present and pred_present:
        return "both_present"
    if target_present and not pred_present:
        return "missed_entirely"
    if not target_present and pred_present:
        return "false_positive_only"
    return "absent_from_both"


def confusion_matrix(pred: np.ndarray, target: np.ndarray, num_classes: int) -> np.ndarray:
    """Rows = true class, columns = predicted class. Standard convention --
    do not transpose without updating every caller.

    Fails loudly (ValueError) on any class ID outside [0, num_classes) in
    EITHER array. This dataset's valid class set is fixed and known (see
    class_codebook.json) -- an out-of-range class ID means something
    upstream is broken (a bad prediction head, a corrupted label), and
    silently dropping those pixels would produce a plausible-looking but
    wrong confusion matrix instead of surfacing the real problem.
    """
    assert pred.shape == target.shape, f"shape mismatch: {pred.shape} vs {target.shape}"

    if np.any((target < 0) | (target >= num_classes)):
        bad = np.unique(target[(target < 0) | (target >= num_classes)])
        raise ValueError(f"target contains invalid class ID(s) outside [0,{num_classes}): {bad.tolist()}")
    if np.any((pred < 0) | (pred >= num_classes)):
        bad = np.unique(pred[(pred < 0) | (pred >= num_classes)])
        raise ValueError(f"pred contains invalid class ID(s) outside [0,{num_classes}): {bad.tolist()}")

    flat_t = target.ravel()
    flat_p = pred.ravel()
    idx = flat_t * num_classes + flat_p
    counts = np.bincount(idx, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes)


def iou_per_class(pred: np.ndarray, target: np.ndarray, num_classes: int) -> dict:
    """Returns {class_id: iou_float_or_None} per the frozen metric contract.
    None ONLY for 'absent_from_both'; 'missed_entirely' and
    'false_positive_only' both correctly yield 0.0 (real, meaningful
    failures), never None."""
    result = {}
    for c in range(num_classes):
        status = class_presence_status(pred, target, c)
        if status == "absent_from_both":
            result[c] = None
            continue
        pred_c = (pred == c)
        target_c = (target == c)
        intersection = np.logical_and(pred_c, target_c).sum()
        union = np.logical_or(pred_c, target_c).sum()
        result[c] = float(intersection) / float(union)
    return result


def boundary_f1_per_class(pred: np.ndarray, target: np.ndarray, num_classes: int,
                           boundary_width: int = 2) -> dict:
    """Returns {class_id: f1_float_or_None} per the frozen metric contract.
    boundary_width is a SQUARE morphological dilation radius (see module
    docstring), not a Euclidean tolerance."""
    struct = np.ones((2 * boundary_width + 1, 2 * boundary_width + 1))
    result = {}

    for c in range(num_classes):
        status = class_presence_status(pred, target, c)
        if status == "absent_from_both":
            result[c] = None
            continue
        if status in ("missed_entirely", "false_positive_only"):
            result[c] = 0.0
            continue

        # both_present -- compute real boundary-tolerant precision/recall
        pred_c = (pred == c)
        target_c = (target == c)
        pred_boundary = find_boundaries(pred_c, mode="thick")
        target_boundary = find_boundaries(target_c, mode="thick")

        # a single-pixel (or otherwise boundary-less under 'thick') region
        # has no interior/exterior distinction -- fall back to treating the
        # whole mask as its own boundary rather than silently producing an
        # empty boundary for a class that IS present
        if not pred_boundary.any():
            pred_boundary = pred_c
        if not target_boundary.any():
            target_boundary = target_c

        pred_boundary_dilated = binary_dilation(pred_boundary, structure=struct)
        target_boundary_dilated = binary_dilation(target_boundary, structure=struct)

        precision = float(np.logical_and(pred_boundary, target_boundary_dilated).sum()) / float(pred_boundary.sum())
        recall = float(np.logical_and(target_boundary, pred_boundary_dilated).sum()) / float(target_boundary.sum())

        result[c] = 0.0 if (precision + recall == 0) else (2 * precision * recall / (precision + recall))

    return result


def hd95_per_class(pred: np.ndarray, target: np.ndarray, num_classes: int):
    """Returns (result, reasons) where result = {class_id: hd95_float_or_None}
    and reasons = {class_id: status_string}, status_string being exactly the
    four class_presence_status() values. 'missed_entirely' and
    'false_positive_only' are DISTINCT reasons -- conflating them was a real
    bug in an earlier version of this file (both a model that never
    predicts TiN and a model that hallucinates TiN everywhere used to be
    labeled 'missed_entirely'; verified fixed in verify_metrics.py)."""
    result = {}
    reasons = {}

    for c in range(num_classes):
        status = class_presence_status(pred, target, c)
        reasons[c] = status
        if status != "both_present":
            result[c] = None
            continue

        pred_c = (pred == c)
        target_c = (target == c)
        pred_boundary = find_boundaries(pred_c, mode="thick")
        target_boundary = find_boundaries(target_c, mode="thick")

        if not pred_boundary.any():
            pred_boundary = pred_c
        if not target_boundary.any():
            target_boundary = target_c

        dt_target = distance_transform_edt(~target_boundary)
        dt_pred = distance_transform_edt(~pred_boundary)

        d_pred_to_target = dt_target[pred_boundary]
        d_target_to_pred = dt_pred[target_boundary]

        result[c] = max(
            float(np.percentile(d_pred_to_target, 95)),
            float(np.percentile(d_target_to_pred, 95)),
        )

    return result, reasons
