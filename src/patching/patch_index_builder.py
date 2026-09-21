"""
patch_index_builder.py
========================
Builds outputs/patch_index.csv: one row per patch (not per image). This is
what the weighted sampler and training dataloader actually read from.

Patches are NOT extracted to disk as separate image files -- only their
coordinates and per-class pixel stats are indexed here. The actual pixel
crop happens at training time (see dataset.py), from the manifest's
image_path/label_path. This avoids inflating disk usage ~44x (avg patches
per image) for a dataset this size.

Uses compute_patch_origins from grid.py -- the SAME function used by
tin_speck_analysis.py's fragmentation simulation. This is deliberate: if
these were two separate implementations, the "0% fragmentation" guarantee
measured during patch-size selection would not actually describe the
patches produced here.
"""

import csv
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.common import load_config
from src.patching.grid import compute_patch_origins


def build_patch_index(config: dict):
    out_dir = config["paths"]["output_dir"]
    data_root = config["paths"]["data_root"]
    patch_size = config["patching"]["patch_size"]
    stride = config["patching"]["stride"]
    rare_name = config["rare_class"]["name"]

    manifest_path = os.path.join(out_dir, "manifest.csv")
    codebook_path = os.path.join(out_dir, "class_codebook.json")
    with open(manifest_path) as f:
        image_rows = list(csv.DictReader(f))
    with open(codebook_path) as f:
        codebook = json.load(f)
    class_names = list(codebook["class_id_to_name"].values())

    patch_rows = []
    patch_id = 0

    for img_row in image_rows:
        H = int(img_row["crop_target_height"])
        W = int(img_row["crop_target_width"])
        label_arr = np.array(Image.open(os.path.join(data_root, img_row["label_path"])))
        assert label_arr.shape == (H, W), (
            f"{img_row['stem']}: label array shape {label_arr.shape} != "
            f"manifest crop_target ({H},{W}) -- manifest and raw file have drifted."
        )

        origins = compute_patch_origins(H, W, patch_size, stride)

        for (y0, x0) in origins:
            patch_lbl = label_arr[y0:y0 + patch_size, x0:x0 + patch_size]
            total_px = patch_lbl.size
            vals, counts = np.unique(patch_lbl, return_counts=True)
            px_counts = {int(v): int(c) for v, c in zip(vals, counts)}

            id_to_name = {int(k): v for k, v in codebook["class_id_to_name"].items()}
            row = {
                "patch_id": patch_id,
                "source_stem": img_row["stem"],
                "true_type": img_row["true_type"],
                "split": img_row["split"],
                "y0": y0,
                "x0": x0,
                "patch_size": patch_size,
            }
            tin_px = 0
            for cid, cname in id_to_name.items():
                px = px_counts.get(cid, 0)
                row[f"px_{cname}"] = px
                row[f"pct_{cname}"] = round(100.0 * px / total_px, 6)
                if cname == rare_name:
                    tin_px = px
            row["tin_present"] = tin_px > 0
            row["tin_pixel_count"] = tin_px
            row["tin_pixel_pct"] = round(100.0 * tin_px / total_px, 6)
            row["total_pixels"] = total_px

            patch_rows.append(row)
            patch_id += 1

    patch_index_path = os.path.join(out_dir, "patch_index.csv")
    fieldnames = list(patch_rows[0].keys()) if patch_rows else []
    with open(patch_index_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(patch_rows)

    per_split_counts = {}
    per_split_tin_counts = {}
    for r in patch_rows:
        per_split_counts[r["split"]] = per_split_counts.get(r["split"], 0) + 1
        if r["tin_present"]:
            per_split_tin_counts[r["split"]] = per_split_tin_counts.get(r["split"], 0) + 1

    summary = {
        "check": "build_patch_index",
        "patch_size": patch_size,
        "stride": stride,
        "total_patches": len(patch_rows),
        "n_source_images": len(image_rows),
        "avg_patches_per_image": round(len(patch_rows) / len(image_rows), 2) if image_rows else 0,
        "per_split_patch_counts": per_split_counts,
        "per_split_tin_patch_counts": per_split_tin_counts,
        "patch_index_path": patch_index_path,
        "conclusion": (
            f"Built {len(patch_rows)} patches from {len(image_rows)} images "
            f"({patch_size}px, stride {stride}). Per-split: {per_split_counts}, "
            f"of which TiN-containing: {per_split_tin_counts}."
        ),
    }
    return patch_rows, summary


if __name__ == "__main__":
    cfg = load_config("configs/config.yaml")
    rows, summary = build_patch_index(cfg)
    print(json.dumps(summary, indent=2, default=str))
