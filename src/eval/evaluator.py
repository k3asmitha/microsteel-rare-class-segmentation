"""
evaluator.py
=============
Ties together reconstruct.py -> metrics.py -> aggregate.py into the actual
evaluation loop B and C's trained models will plug into.

PRODUCTION INTERFACE (mandatory, enforced by construction, not just
documented): predict_fn(image_patch: np.ndarray) -> probability array of
shape (num_classes, ps, ps). It receives ONLY the image patch. There is no
code path in evaluate_image/evaluate_dataset that ever passes a label to
predict_fn -- this is deliberate: an interface that *could* pass the label
is an interface that *will* eventually leak it, whether by a copy-pasted
debug wrapper or a well-meaning "just in case" parameter. The one caller
that legitimately needs label access (verifying the reconstruct/metrics
chain without a real trained model) uses a SEPARATE function,
debug_evaluate_dataset_from_ground_truth, that never touches predict_fn at
all -- see below.

RESULTS: evaluate_dataset returns BOTH per_image_results (list of every
image's individual iou/f1/hd95 dicts, with stem/true_type/split/
architecture/configuration identity attached) AND the aggregated summary.
Per-image results are what let you later answer "which Type II image
failed", "is TiN performance dominated by one lucky/unlucky image", "what
should go in the failure-analysis section" -- an aggregate mean alone
cannot answer any of these, and recomputing per-image results after the
fact means rerunning inference.
"""

import csv
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.common import load_config
from src.eval.reconstruct import get_patch_coords_for_stem, reconstruct_prediction, one_hot_from_labels
from src.eval.metrics import iou_per_class, boundary_f1_per_class, hd95_per_class
from src.eval.aggregate import aggregate_iou_or_f1, aggregate_hd95, format_summary_table


def load_rows(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Production predict_fn dummies -- image-only, same interface a real model uses
# ---------------------------------------------------------------------------

def make_predict_majority_class(num_classes, majority_class_id=0):
    """The only dummy that legitimately fits the production interface --
    it doesn't need the label, so it's a fair test of predict_fn(image_patch)."""
    def predict_fn(image_patch):
        ps = image_patch.shape[0]
        probs = np.zeros((num_classes, ps, ps), dtype=np.float64)
        probs[majority_class_id] = 1.0
        return probs
    return predict_fn


# ---------------------------------------------------------------------------
# Production evaluation path -- predict_fn NEVER receives a label
# ---------------------------------------------------------------------------

def _load_image_and_label(row, data_root):
    label = np.array(Image.open(os.path.join(data_root, row["label_path"])))
    H, W = label.shape
    img = Image.open(os.path.join(data_root, row["image_path"])).convert("L")
    img = img.crop((0, 0, W, H))
    return np.array(img), label, H, W


def evaluate_image(stem: str, manifest_by_stem: dict, patch_rows: list,
                    data_root: str, num_classes: int, predict_fn,
                    architecture: str = None, configuration: str = None) -> dict:
    """Runs predict_fn(image_patch) -- IMAGE ONLY -- on every patch of this
    image, reconstructs the full prediction, and computes all three metrics
    against the true label. architecture/configuration are optional
    passthrough identity fields for once B/C have real trained models."""
    row = manifest_by_stem[stem]
    img_arr, label, H, W = _load_image_and_label(row, data_root)
    coords = get_patch_coords_for_stem(patch_rows, stem)

    patch_probs = {}
    for (y0, x0, ps) in coords:
        img_patch = img_arr[y0:y0 + ps, x0:x0 + ps]
        patch_probs[(y0, x0)] = predict_fn(img_patch)  # NO label passed -- ever

    pred = reconstruct_prediction(patch_probs, coords, (H, W), num_classes)
    return _compute_result(stem, row, pred, label, num_classes, architecture, configuration)


def _compute_result(stem, row, pred, label, num_classes, architecture, configuration) -> dict:
    iou = iou_per_class(pred, label, num_classes=num_classes)
    f1 = boundary_f1_per_class(pred, label, num_classes=num_classes)
    hd95, hd95_reasons = hd95_per_class(pred, label, num_classes=num_classes)
    return {"stem": stem, "true_type": row["true_type"], "split": row["split"],
            "architecture": architecture, "configuration": configuration,
            "iou": iou, "f1": f1, "hd95": hd95, "hd95_reasons": hd95_reasons}


def evaluate_dataset(config: dict, predict_fn, split: str = None,
                      image_stems: list = None,
                      architecture: str = None, configuration: str = None) -> dict:
    """Runs evaluate_image over a set of stems and aggregates per-class
    results. Returns BOTH per_image_results and the aggregates.

    SPLIT ENFORCEMENT (required, not optional): at least one of `split` or
    `image_stems` MUST be given -- no "evaluate on everything" default,
    since that previously made it trivial to silently mix train/val/test.
    """
    out_dir = config["paths"]["output_dir"]
    data_root = config["paths"]["data_root"]
    codebook_path = os.path.join(out_dir, "class_codebook.json")

    manifest_rows = load_rows(os.path.join(out_dir, "manifest.csv"))
    patch_rows = load_rows(os.path.join(out_dir, "patch_index.csv"))
    manifest_by_stem = {r["stem"]: r for r in manifest_rows}

    with open(codebook_path, encoding="utf-8") as f:
        codebook = json.load(f)
    id_to_name = {int(k): v for k, v in codebook["class_id_to_name"].items()}
    num_classes = len(id_to_name)

    if split is None and image_stems is None:
        raise ValueError(
            "evaluate_dataset requires at least one of `split` or "
            "`image_stems` -- there is no 'evaluate on everything' default. "
            "Pass split='test' (etc.) to evaluate a whole split, or pass an "
            "explicit image_stems list."
        )

    if image_stems is None:
        stems = [s for s, r in manifest_by_stem.items() if r["split"] == split]
    else:
        stems = list(image_stems)
        if split is not None:
            mismatched = [s for s in stems if manifest_by_stem[s]["split"] != split]
            if mismatched:
                raise ValueError(
                    f"image_stems contains {len(mismatched)} stem(s) NOT in "
                    f"split='{split}' (e.g. {mismatched[:3]}) -- refusing to "
                    f"evaluate a mix of splits under one split label."
                )

    per_image_results = []
    for stem in stems:
        result = evaluate_image(stem, manifest_by_stem, patch_rows, data_root, num_classes,
                                 predict_fn, architecture=architecture, configuration=configuration)
        per_image_results.append(result)

    iou_agg = aggregate_iou_or_f1([r["iou"] for r in per_image_results], num_classes)
    f1_agg = aggregate_iou_or_f1([r["f1"] for r in per_image_results], num_classes)
    hd95_agg = aggregate_hd95([r["hd95"] for r in per_image_results],
                               [r["hd95_reasons"] for r in per_image_results], num_classes)

    table = format_summary_table(iou_agg, f1_agg, hd95_agg, id_to_name)

    return {
        "n_images": len(stems),
        "split": split,
        "architecture": architecture,
        "configuration": configuration,
        "per_image_results": per_image_results,
        "iou_agg": iou_agg,
        "f1_agg": f1_agg,
        "hd95_agg": hd95_agg,
        "summary_table": table,
    }


def save_evaluation_results(result: dict, output_root: str) -> str:
    """Writes per-image results to
    outputs/results/<type>/<architecture>/<configuration>/per_image.csv
    plus a summary.json with the aggregates, one folder per (type,
    architecture, configuration) combination present in per_image_results.
    Returns the list of directories written."""
    written = []
    by_type = {}
    for r in result["per_image_results"]:
        by_type.setdefault(r["true_type"], []).append(r)

    for true_type, rows in by_type.items():
        arch = rows[0]["architecture"] or "unspecified_architecture"
        config_name = rows[0]["configuration"] or "unspecified_configuration"
        out_dir = os.path.join(output_root, "results", f"Type_{true_type}", arch, config_name)
        os.makedirs(out_dir, exist_ok=True)

        csv_path = os.path.join(out_dir, "per_image.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["stem", "true_type", "split", "class_id", "iou", "f1",
                              "hd95", "hd95_reason"])
            for r in rows:
                for c in r["iou"]:
                    writer.writerow([r["stem"], r["true_type"], r["split"], c,
                                      r["iou"].get(c), r["f1"].get(c),
                                      r["hd95"].get(c), r["hd95_reasons"].get(c)])

        summary_path = os.path.join(out_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({
                "n_images": len(rows),
                "iou_agg": {c: v for c, v in result["iou_agg"].items()},
                "f1_agg": {c: v for c, v in result["f1_agg"].items()},
                "hd95_agg": {c: v for c, v in result["hd95_agg"].items()},
            }, f, indent=2, default=str)

        written.append(out_dir)
    return written


# ---------------------------------------------------------------------------
# TEST-ONLY ground-truth check -- deliberately NOT part of the production
# predict_fn interface. Never call this to evaluate a real model; it exists
# solely to prove the reconstruct -> metrics -> aggregate wiring is correct,
# by feeding the true label back in as a "perfect prediction". Because it
# builds patch_probs directly from labels instead of calling any predict_fn,
# it is architecturally impossible for this path to be mistaken for real
# model evaluation code -- there's no predict_fn parameter to plug a real
# model into.
# ---------------------------------------------------------------------------

def debug_evaluate_dataset_from_ground_truth(config: dict, split: str = None,
                                              image_stems: list = None) -> dict:
    out_dir = config["paths"]["output_dir"]
    data_root = config["paths"]["data_root"]
    codebook_path = os.path.join(out_dir, "class_codebook.json")

    manifest_rows = load_rows(os.path.join(out_dir, "manifest.csv"))
    patch_rows = load_rows(os.path.join(out_dir, "patch_index.csv"))
    manifest_by_stem = {r["stem"]: r for r in manifest_rows}

    with open(codebook_path, encoding="utf-8") as f:
        codebook = json.load(f)
    id_to_name = {int(k): v for k, v in codebook["class_id_to_name"].items()}
    num_classes = len(id_to_name)

    if split is None and image_stems is None:
        raise ValueError("requires at least one of `split` or `image_stems`")
    if image_stems is None:
        stems = [s for s, r in manifest_by_stem.items() if r["split"] == split]
    else:
        stems = list(image_stems)
        if split is not None:
            mismatched = [s for s in stems if manifest_by_stem[s]["split"] != split]
            if mismatched:
                raise ValueError(f"image_stems contains stem(s) NOT in split='{split}': {mismatched[:3]}")

    per_image_results = []
    for stem in stems:
        row = manifest_by_stem[stem]
        img_arr, label, H, W = _load_image_and_label(row, data_root)
        coords = get_patch_coords_for_stem(patch_rows, stem)
        patch_probs = {}
        for (y0, x0, ps) in coords:
            lbl_patch = label[y0:y0 + ps, x0:x0 + ps]
            patch_probs[(y0, x0)] = one_hot_from_labels(lbl_patch, num_classes)
        pred = reconstruct_prediction(patch_probs, coords, (H, W), num_classes)
        per_image_results.append(_compute_result(stem, row, pred, label, num_classes, None, "debug_ground_truth"))

    iou_agg = aggregate_iou_or_f1([r["iou"] for r in per_image_results], num_classes)
    f1_agg = aggregate_iou_or_f1([r["f1"] for r in per_image_results], num_classes)
    hd95_agg = aggregate_hd95([r["hd95"] for r in per_image_results],
                               [r["hd95_reasons"] for r in per_image_results], num_classes)
    table = format_summary_table(iou_agg, f1_agg, hd95_agg, id_to_name)

    return {"n_images": len(stems), "per_image_results": per_image_results,
            "iou_agg": iou_agg, "f1_agg": f1_agg, "hd95_agg": hd95_agg, "summary_table": table}


if __name__ == "__main__":
    cfg = load_config("configs/config.yaml")
    manifest_rows = load_rows(os.path.join(cfg["paths"]["output_dir"], "manifest.csv"))
    type1_stems = [r["stem"] for r in manifest_rows if r["true_type"] == "I"][:5]

    print("=== Smoke test: debug_evaluate_dataset_from_ground_truth on 5 Type I images ===")
    result_gt = debug_evaluate_dataset_from_ground_truth(cfg, image_stems=type1_stems)
    print(result_gt["summary_table"])

    print("\n=== Smoke test: predict_majority_class (always Alpha, image-only interface) ===")
    result_maj = evaluate_dataset(cfg, make_predict_majority_class(5, majority_class_id=0), image_stems=type1_stems)
    print(result_maj["summary_table"])
