"""
aggregate.py
==============
Aggregates per-image metric dicts (from metrics.py) into per-class summary
statistics, following the AGGREGATION CONTRACT frozen in metrics.py's
module docstring:

  - IoU / Boundary F1: mean over every non-None value. 'missed_entirely'
    and 'false_positive_only' are ALREADY 0.0 (not None) per metrics.py's
    contract, so a plain mean-of-non-None values correctly includes both
    failure modes. Only 'absent_from_both' (None) is excluded. This
    module does NOT filter to "GT-present only" -- doing so would silently
    discard real false-positive signal (a model that hallucinates TiN
    everywhere would look fine if false-positive-only cases were dropped).
  - HD95: mean ONLY over 'computed' (both_present) values. Counts of
    missed_entirely / false_positive_only / absent_from_both are reported
    as separate counts, NEVER folded into the mean -- a missing HD95 is a
    distinct failure category, not a value to smooth over.

Every summary reports its sample count (n) alongside any mean, because a
mean over 3 images looks identical to a mean over 300 without it, and TiN
is absent from most individual images/patches.
"""

import numpy as np


def aggregate_iou_or_f1(per_image_dicts: list, num_classes: int) -> dict:
    """per_image_dicts: list of {class_id: float_or_None}, one per image,
    all from the same grouping (e.g. same true_type + split).
    Returns {class_id: {mean, n_included, n_excluded_absent_from_both}}."""
    result = {}
    for c in range(num_classes):
        values = [d[c] for d in per_image_dicts if c in d]
        included = [v for v in values if v is not None]
        n_excluded = len(values) - len(included)
        result[c] = {
            "mean": float(np.mean(included)) if included else None,
            "n_included": len(included),
            "n_excluded_absent_from_both": n_excluded,
            "n_total_images": len(values),
        }
    return result


def aggregate_hd95(per_image_results: list, per_image_reasons: list, num_classes: int) -> dict:
    """per_image_results: list of {class_id: float_or_None}
    per_image_reasons: list of {class_id: status_string}, same order/length
    Returns {class_id: {mean_computed, n_computed, n_missed_entirely,
                          n_false_positive_only, n_absent_from_both}}."""
    assert len(per_image_results) == len(per_image_reasons)
    result = {}
    for c in range(num_classes):
        computed_values = []
        counts = {"missed_entirely": 0, "false_positive_only": 0, "absent_from_both": 0}
        for res, reasons in zip(per_image_results, per_image_reasons):
            if c not in reasons:
                continue
            status = reasons[c]
            if status == "both_present":
                computed_values.append(res[c])
            else:
                counts[status] += 1

        result[c] = {
            "mean_computed": float(np.mean(computed_values)) if computed_values else None,
            "n_computed": len(computed_values),
            "n_missed_entirely": counts["missed_entirely"],
            "n_false_positive_only": counts["false_positive_only"],
            "n_absent_from_both": counts["absent_from_both"],
        }
    return result


def format_summary_table(iou_agg: dict, f1_agg: dict, hd95_agg: dict, class_names: dict) -> str:
    """Human-readable table for a report -- one row per class, with n
    always visible next to any mean so nobody reads a rare-class score
    without knowing how few images it's based on."""
    lines = []
    header = f"{'Class':<10}{'IoU (n)':<20}{'BoundaryF1 (n)':<20}{'HD95 (n_computed)':<22}{'missed':<8}{'false_pos':<10}"
    lines.append(header)
    for c in sorted(class_names.keys()):
        name = class_names[c]
        iou_s = f"{iou_agg[c]['mean']:.3f} ({iou_agg[c]['n_included']})" if iou_agg[c]['mean'] is not None else f"n/a ({iou_agg[c]['n_included']})"
        f1_s = f"{f1_agg[c]['mean']:.3f} ({f1_agg[c]['n_included']})" if f1_agg[c]['mean'] is not None else f"n/a ({f1_agg[c]['n_included']})"
        hd_s = f"{hd95_agg[c]['mean_computed']:.2f} ({hd95_agg[c]['n_computed']})" if hd95_agg[c]['mean_computed'] is not None else f"n/a ({hd95_agg[c]['n_computed']})"
        lines.append(f"{name:<10}{iou_s:<20}{f1_s:<20}{hd_s:<22}"
                      f"{hd95_agg[c]['n_missed_entirely']:<8}{hd95_agg[c]['n_false_positive_only']:<10}")
    return "\n".join(lines)
