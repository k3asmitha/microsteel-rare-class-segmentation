"""
common.py
=========
Shared low-level utilities used by every audit/manifest/analysis script.
Exists so that "how do we find the true type of an image" or "how do we map
a stem to its image file" is implemented ONCE, not copy-pasted and drifting
across five scripts.
"""

import glob
import os
import re

import yaml


def load_config(config_path="configs/config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def true_type_from_stem(stem: str) -> str:
    """Parse true processing type from the FILENAME PREFIX.
    Do not use folder names -- images_class2/ is a verified mixed bag of
    real Class_II and Class_III images (see audit/inventory_audit.py)."""
    m = re.match(r"^Class_(I|II|III)_", stem)
    return m.group(1) if m else "UNKNOWN"


def build_image_map(data_root: str) -> dict:
    """stem -> image_path, scanning ALL images_class* folders regardless of
    which folder a file happens to sit in."""
    img_map = {}
    dupes = []
    folders = sorted(glob.glob(os.path.join(data_root, "images_class*")))
    for folder in folders:
        for ext in ("*.jpg", "*.jpeg", "*.JPG"):
            for p in glob.glob(os.path.join(folder, ext)):
                stem = os.path.splitext(os.path.basename(p))[0]
                if stem in img_map:
                    dupes.append((stem, img_map[stem], p))
                img_map[stem] = p
    return img_map, dupes


def normalize_split_entry(raw: str) -> str:
    """Strip trailing .png/.jpg extensions repeatedly -- fixes the verified
    '.jpg.png' double-extension typo in splits/val_c1.txt without assuming
    there's only ever one extension to strip."""
    s = raw.strip()
    while True:
        new_s = re.sub(r"\.(png|jpg|jpeg)$", "", s, flags=re.IGNORECASE)
        if new_s == s:
            break
        s = new_s
    return s


def list_label_stems(data_root: str) -> list:
    label_dir = os.path.join(data_root, "labels", "labels")
    return sorted(os.path.splitext(f)[0] for f in os.listdir(label_dir) if f.endswith(".png"))


def label_path(data_root: str, stem: str) -> str:
    return os.path.join(data_root, "labels", "labels", f"{stem}.png")


def coloured_label_path(data_root: str, stem: str) -> str:
    return os.path.join(data_root, "labels", "coloured_labels", f"{stem}.png")
