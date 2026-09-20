"""
a05_split_audit.py
====================
QUESTIONS:
  1. Are the official splits/*.txt files internally consistent (do the
     per-class files train_c1/c2/c3.txt union up to train.txt, etc.)?
  2. Do any entries have typos/extension issues that would break a naive
     string-match loader?
  3. Is the rare class (TiN) reasonably represented across train/val/test,
     or does the official split leave a set with too little/no signal?

METHOD: Parse every split file directly, normalize extensions, and diff
aggregate vs per-class union. Compute real TiN pixel share per split using
the manifest-independent raw label files (so this audit doesn't depend on
the manifest already being correct).
"""

import os
from collections import defaultdict

import numpy as np
from PIL import Image

from src.common import (
    true_type_from_stem, build_image_map, list_label_stems,
    label_path, normalize_split_entry,
)


def read_raw_split_file(path):
    """Returns list of RAW (un-normalized) lines, to detect typos."""
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def run(config: dict) -> dict:
    data_root = config["paths"]["data_root"]
    rare_id = config["rare_class"]["id"]
    splits_dir = os.path.join(data_root, "splits")

    img_map, _ = build_image_map(data_root)
    label_stems = set(list_label_stems(data_root))
    usable_stems = label_stems & set(img_map.keys())

    findings_typos = []
    aggregate_consistency = {}

    for base, parts in [
        ("train", ["train_c1.txt", "train_c2.txt", "train_c3.txt"]),
        ("val", ["val_c1.txt", "val_c2.txt", "val_c3.txt"]),
        ("test", ["test_c1.txt", "test_c2.txt", "test_c3.txt"]),
    ]:
        agg_raw = read_raw_split_file(os.path.join(splits_dir, f"{base}.txt"))
        agg_norm = set(normalize_split_entry(l) for l in agg_raw)

        union_raw = []
        for p in parts:
            union_raw.extend(read_raw_split_file(os.path.join(splits_dir, p)))
        union_norm = set(normalize_split_entry(l) for l in union_raw)

        # detect typo'd raw entries: normalized form matches, but raw string
        # doesn't look like a clean "<stem>.png"
        typo_entries = [l for l in union_raw if l != f"{normalize_split_entry(l)}.png"]
        if typo_entries:
            findings_typos.extend([{"split_group": base, "raw_entry": t,
                                     "normalized_to": normalize_split_entry(t)}
                                    for t in typo_entries])

        aggregate_consistency[base] = {
            "aggregate_count": len(agg_norm),
            "per_class_union_count": len(union_norm),
            "matches": agg_norm == union_norm,
            "in_agg_not_in_union": sorted(agg_norm - union_norm),
            "in_union_not_in_agg": sorted(union_norm - agg_norm),
        }

    # --- TiN stratification per split, using normalized stems, excluding
    #     any stem without a real image (orphans) so this doesn't crash ---
    tin_stats = {}
    for split_name in ["train", "val", "test"]:
        raw_lines = read_raw_split_file(os.path.join(splits_dir, f"{split_name}.txt"))
        stems = [normalize_split_entry(l) for l in raw_lines]
        usable = [s for s in stems if s in usable_stems]
        dropped = [s for s in stems if s not in usable_stems]

        tin_px = 0
        total_px = 0
        n_with_tin = 0
        per_type_n = defaultdict(int)
        for stem in usable:
            arr = np.array(Image.open(label_path(data_root, stem)))
            n_tin = int((arr == rare_id).sum())
            if n_tin > 0:
                n_with_tin += 1
            tin_px += n_tin
            total_px += arr.size
            per_type_n[true_type_from_stem(stem)] += 1

        tin_stats[split_name] = {
            "n_usable_images": len(usable),
            "n_dropped_no_image": len(dropped),
            "dropped_stems": dropped,
            "n_images_containing_tin": n_with_tin,
            "tin_pixel_pct": round(100.0 * tin_px / total_px, 4) if total_px else None,
            "per_type_image_count": dict(per_type_n),
        }

    all_consistent = all(v["matches"] for v in aggregate_consistency.values())

    findings = {
        "check": "split_audit",
        "question": "Are split files internally consistent, typo-free, and TiN-stratified?",
        "aggregate_vs_per_class_consistency": aggregate_consistency,
        "all_splits_internally_consistent": all_consistent,
        "typo_entries_found": findings_typos,
        "tin_stratification_per_split": tin_stats,
        "conclusion": (
            (f"Split files are internally consistent. "
             if all_consistent else
             f"INCONSISTENCY found between aggregate and per-class split files -- see details. ")
            + (f"{len(findings_typos)} typo'd entries found (e.g. double "
               f"extensions) -- split parsing MUST normalize extensions, "
               f"not string-match literally. "
               if findings_typos else "No typo'd entries found. ")
            + "TiN pixel share across splits: " +
            ", ".join(f"{s}={tin_stats[s]['tin_pixel_pct']}%" for s in ["train", "val", "test"])
            + ". " + (
                "Reasonably balanced -- official splits are usable as-is "
                "(after excluding orphan stems with no image)."
                if max(v["tin_pixel_pct"] for v in tin_stats.values() if v["tin_pixel_pct"]) /
                   max(min(v["tin_pixel_pct"] for v in tin_stats.values() if v["tin_pixel_pct"]), 1e-9) < 3
                else "Split TiN shares differ by >3x -- consider rebuilding "
                     "a custom stratified split instead of trusting the official one."
            )
        ),
    }
    return findings


if __name__ == "__main__":
    import sys, json
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from src.common import load_config
    cfg = load_config("configs/config.yaml")
    result = run(cfg)
    print(json.dumps(result, indent=2, default=str))
