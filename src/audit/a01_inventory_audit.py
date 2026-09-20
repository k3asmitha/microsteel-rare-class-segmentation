"""
a01_inventory_audit.py
=======================
QUESTION: Can folder names (images_class1/2/3) be trusted as the processing
type label, or must true type be parsed from the filename?

METHOD: For every image in every images_class* folder, parse the true type
from its filename prefix (Class_I_/Class_II_/Class_III_) and compare against
what the folder name would imply. Report any mismatch.

This is checked FIRST because every other audit and the manifest itself
depends on knowing which check is authoritative (folder vs filename).
"""

import os
from collections import defaultdict

from src.common import true_type_from_stem, build_image_map


FOLDER_IMPLIES_TYPE = {
    "images_class1": "I",
    "images_class2": "II",
    "images_class3": "III",
}


def run(config: dict) -> dict:
    data_root = config["paths"]["data_root"]

    per_folder_true_types = defaultdict(lambda: defaultdict(int))
    mismatches = []

    for folder_name, implied_type in FOLDER_IMPLIES_TYPE.items():
        folder_path = os.path.join(data_root, folder_name)
        if not os.path.isdir(folder_path):
            continue
        for fname in os.listdir(folder_path):
            if not fname.lower().endswith((".jpg", ".jpeg")):
                continue
            stem = os.path.splitext(fname)[0]
            true_type = true_type_from_stem(stem)
            per_folder_true_types[folder_name][true_type] += 1
            if true_type != implied_type:
                mismatches.append({
                    "stem": stem,
                    "folder": folder_name,
                    "folder_implies_type": implied_type,
                    "true_type_from_filename": true_type,
                })

    folder_trustworthy = len(mismatches) == 0

    findings = {
        "check": "inventory_audit",
        "question": "Can folder name be trusted as processing type?",
        "folder_trustworthy": folder_trustworthy,
        "per_folder_type_breakdown": {
            f: dict(counts) for f, counts in per_folder_true_types.items()
        },
        "mismatch_count": len(mismatches),
        "mismatches_sample": mismatches[:20],
        "conclusion": (
            "Folder names are RELIABLE -- true type always matches folder."
            if folder_trustworthy else
            "Folder names are UNRELIABLE. Downstream code must parse true "
            "type from the filename prefix, never from the containing folder."
        ),
    }
    return findings


if __name__ == "__main__":
    import sys, json
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from src.common import load_config
    cfg = load_config("configs/config.yaml")
    result = run(cfg)
    print(json.dumps(result, indent=2))
