"""
build_manifest.py
===================
Builds manifest.csv: the single source of truth every downstream component
(patcher, sampler, eval, B's and C's training loops) reads from.

Deliberately does NOT re-derive facts already established by src/audit/*.
It imports and reuses:
  - a01/a02 pairing logic (via common.build_image_map / list_label_stems)
    to determine usable pairs and excluded orphans
  - a04's decoded codebook (class_id -> phase name), so the manifest's class
    columns use whatever the codebook audit actually found, not a hardcoded
    guess re-typed here
  - a05's split normalization, so split assignment handles the same typos
    the split audit already found
"""

import csv
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.common import (
    true_type_from_stem, build_image_map, list_label_stems,
    label_path, coloured_label_path, normalize_split_entry, load_config,
)
from src.audit.a04_codebook_audit import run as run_codebook_audit


def load_split_membership(data_root: str) -> dict:
    splits_dir = os.path.join(data_root, "splits")
    stem_to_split = {}
    for split_name in ["train", "val", "test"]:
        path = os.path.join(splits_dir, f"{split_name}.txt")
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                stem = normalize_split_entry(line)
                stem_to_split[stem] = split_name
    return stem_to_split


def build_manifest(config: dict):
    data_root = config["paths"]["data_root"]
    out_dir = config["paths"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # Reuse the codebook audit's decoded mapping instead of hardcoding it.
    codebook_result = run_codebook_audit(config)
    if not codebook_result["codebook_is_consistent"]:
        raise RuntimeError(
            "Codebook audit found an INCONSISTENT integer<->RGB mapping. "
            "Refusing to build a manifest on top of an unsafe codebook. "
            "See a04_codebook_audit output for details."
        )
    decoded_codebook = {int(k): v for k, v in codebook_result["decoded_codebook"].items()}
    class_names = list(decoded_codebook.values())
    rare_class_id = config["rare_class"]["id"]
    rare_class_name = config["rare_class"]["name"]

    img_map, _ = build_image_map(data_root)
    label_stems = list_label_stems(data_root)
    stem_to_split = load_split_membership(data_root)

    rows = []
    excluded = []

    for stem in label_stems:
        true_type = true_type_from_stem(stem)

        if stem not in img_map:
            excluded.append({"stem": stem, "true_type": true_type,
                              "reason": "no_matching_image_file"})
            continue

        img_path = img_map[stem]
        lbl_path = label_path(data_root, stem)
        clbl_path = coloured_label_path(data_root, stem)

        with Image.open(img_path) as im:
            orig_w, orig_h = im.size
            orig_mode = im.mode
        with Image.open(lbl_path) as lb:
            lbl_w, lbl_h = lb.size
            lbl_arr = np.array(lb)

        if orig_w != lbl_w:
            raise ValueError(
                f"{stem}: width mismatch image={orig_w} label={lbl_w} -- "
                f"the audited pattern is height-only cropping, this breaks that assumption."
            )
        crop_note = (f"crop_bottom_{orig_h - lbl_h}px" if orig_h != lbl_h
                     else "no_crop_needed")

        vals, counts = np.unique(lbl_arr, return_counts=True)
        px_counts = {int(v): int(c) for v, c in zip(vals, counts)}
        total_px = int(lbl_arr.size)
        tin_px = px_counts.get(rare_class_id, 0)

        split = stem_to_split.get(stem, "UNASSIGNED")

        row = {
            "stem": stem,
            "true_type": true_type,
            "split": split,
            "image_path": os.path.relpath(img_path, data_root),
            "label_path": os.path.relpath(lbl_path, data_root),
            "coloured_label_path": os.path.relpath(clbl_path, data_root),
            "orig_width": orig_w,
            "orig_height": orig_h,
            "crop_target_width": lbl_w,
            "crop_target_height": lbl_h,
            "crop_note": crop_note,
            "orig_color_mode": orig_mode,
            "tin_present": tin_px > 0,
            "tin_pixel_count": tin_px,
            "tin_pixel_pct": round(100.0 * tin_px / total_px, 6),
            "total_pixels": total_px,
        }
        for cid, cname in decoded_codebook.items():
            row[f"px_{cname}"] = px_counts.get(cid, 0)
            row[f"pct_{cname}"] = round(100.0 * px_counts.get(cid, 0) / total_px, 6)

        rows.append(row)

    # --- write outputs ---
    manifest_path = os.path.join(out_dir, "manifest.csv")
    fieldnames = list(rows[0].keys()) if rows else []
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    excluded_path = os.path.join(out_dir, "excluded_labels.csv")
    with open(excluded_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["stem", "true_type", "reason"])
        writer.writeheader()
        writer.writerows(excluded)

    codebook_path = os.path.join(out_dir, "class_codebook.json")
    with open(codebook_path, "w") as f:
        json.dump({
            "class_id_to_name": decoded_codebook,
            "rare_class_id": rare_class_id,
            "rare_class_name": rare_class_name,
            "type_class_membership": config["expected_type_class_membership"],
            "decode_evidence": codebook_result["decode_evidence"],
        }, f, indent=2)

    summary = {
        "check": "build_manifest",
        "usable_pairs": len(rows),
        "excluded_count": len(excluded),
        "excluded": excluded,
        "manifest_path": manifest_path,
        "excluded_path": excluded_path,
        "codebook_path": codebook_path,
    }
    return rows, excluded, summary


if __name__ == "__main__":
    cfg = load_config("configs/config.yaml")
    rows, excluded, summary = build_manifest(cfg)
    print(json.dumps(summary, indent=2, default=str))
