"""
a02_pairing_audit.py
======================
QUESTION: Does every label file have a matching source image, and does every
image have a matching label? Are there duplicate stems across folders?

METHOD: Build the full stem sets for images (via true filename-based mapping,
not folder-based) and labels, then set-compare.
"""

import os

from src.common import true_type_from_stem, build_image_map, list_label_stems


def run(config: dict) -> dict:
    data_root = config["paths"]["data_root"]

    img_map, dupes = build_image_map(data_root)
    label_stems = list_label_stems(data_root)

    img_stems = set(img_map.keys())
    lbl_stems = set(label_stems)

    labels_without_image = sorted(lbl_stems - img_stems)
    images_without_label = sorted(img_stems - lbl_stems)
    usable_stems = sorted(lbl_stems & img_stems)

    orphan_details = [
        {"stem": s, "true_type": true_type_from_stem(s)}
        for s in labels_without_image
    ]

    findings = {
        "check": "pairing_audit",
        "question": "Does every label have a matching image and vice versa?",
        "total_label_files": len(lbl_stems),
        "total_image_files": len(img_stems),
        "usable_pairs": len(usable_stems),
        "labels_without_image_count": len(labels_without_image),
        "labels_without_image": orphan_details,
        "images_without_label_count": len(images_without_label),
        "images_without_label": images_without_label,
        "duplicate_stems_across_folders": dupes,
        "conclusion": (
            f"{len(usable_stems)} usable image-label pairs found "
            f"(out of {len(lbl_stems)} labels). "
            f"{len(labels_without_image)} label(s) have NO source image and "
            f"must be excluded from training -- they cannot be used no "
            f"matter what the split files say."
            if labels_without_image else
            f"All {len(lbl_stems)} labels have a matching image. No exclusions needed."
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
