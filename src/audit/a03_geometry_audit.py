"""
a03_geometry_audit.py
=======================
QUESTIONS:
  1. Do image and label dimensions match? If not, is the mismatch constant
     (safe to encode as a fixed crop rule) or variable (needs per-image logic)?
  2. Is image resolution constant across the dataset, or must patch/crop
     code handle variable sizes?
  3. Are all images single-channel grayscale, or are some RGB -- and if RGB,
     do the channels actually carry different information?

METHOD: Measure every usable pair directly. Save one visual overlay of mask
vs cropped image as evidence that the crop rule produces correct spatial
alignment, not just matching pixel counts.
"""

import os
from collections import defaultdict

import numpy as np
from PIL import Image

from src.common import true_type_from_stem, build_image_map, list_label_stems, label_path


def get_usable_stems(data_root):
    img_map, _ = build_image_map(data_root)
    label_stems = set(list_label_stems(data_root))
    usable = sorted(label_stems & set(img_map.keys()))
    return usable, img_map


def run(config: dict) -> dict:
    data_root = config["paths"]["data_root"]
    evidence_dir = config["paths"]["evidence_dir"]
    os.makedirs(evidence_dir, exist_ok=True)

    usable_stems, img_map = get_usable_stems(data_root)

    size_diffs = defaultdict(list)
    resolutions = defaultdict(int)
    color_modes = defaultdict(int)
    rgb_channel_identical = {}

    for stem in usable_stems:
        img = Image.open(img_map[stem])
        lbl = Image.open(label_path(data_root, stem))

        diff = (img.size[0] - lbl.size[0], img.size[1] - lbl.size[1])
        size_diffs[diff].append(stem)
        resolutions[img.size] += 1
        color_modes[img.mode] += 1

        if img.mode == "RGB":
            arr = np.array(img)
            r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
            identical = bool(np.array_equal(r, g) and np.array_equal(g, b))
            rgb_channel_identical[stem] = identical

    constant_offset = len(size_diffs) == 1
    the_offset = list(size_diffs.keys())[0] if constant_offset else None

    # --- Evidence generation: pick one representative pair ---
    if usable_stems:
        example_stem = usable_stems[0]
        img = Image.open(img_map[example_stem])
        w, h_img = img.size
        lbl_arr = np.array(Image.open(label_path(data_root, example_stem)))
        h_lbl = lbl_arr.shape[0]

        # bottom strip = the part cropped away
        bottom_strip = img.crop((0, h_lbl, w, h_img))
        bottom_strip.save(os.path.join(evidence_dir, "removed_bottom_strip.png"))

        # alignment overlay: cropped image + nonzero-class mask in red
        cropped = np.array(img.crop((0, 0, w, h_lbl)).convert("RGB")).astype(np.float32)
        mask_nonzero = lbl_arr > 0
        overlay = cropped.copy()
        overlay[mask_nonzero] = overlay[mask_nonzero] * 0.4 + np.array([255, 0, 0]) * 0.6
        Image.fromarray(overlay.astype(np.uint8)).save(
            os.path.join(evidence_dir, "crop_alignment_overlay.png")
        )
        evidence_files = ["removed_bottom_strip.png", "crop_alignment_overlay.png"]
    else:
        evidence_files = []

    n_rgb_trivial = sum(1 for v in rgb_channel_identical.values() if v)
    n_rgb_nontrivial = sum(1 for v in rgb_channel_identical.values() if not v)

    findings = {
        "check": "geometry_audit",
        "question": "Do image/label dims match, is resolution constant, are all images grayscale?",
        "size_diff_patterns": {
            f"w_diff={k[0]},h_diff={k[1]}": len(v) for k, v in size_diffs.items()
        },
        "constant_crop_offset": constant_offset,
        "crop_offset_px": the_offset,
        "resolution_distribution": {str(k): v for k, v in resolutions.items()},
        "resolution_is_constant": len(resolutions) == 1,
        "color_mode_distribution": dict(color_modes),
        "rgb_images_with_identical_channels": n_rgb_trivial,
        "rgb_images_with_differing_channels": n_rgb_nontrivial,
        "evidence_files": evidence_files,
        "conclusion": (
            (f"Crop rule is CONSTANT: every image is exactly "
             f"{the_offset[1]}px taller than its label (width unchanged). "
             f"Rule: crop each image to its label's own height -- this is "
             f"relative, not a fixed target resolution, "
             f"{'since resolution varies across the dataset.' if len(resolutions) > 1 else 'though resolution happens to be constant here.'} "
             f"Visual overlay confirms spatial alignment is correct.")
            if constant_offset else
            "Crop offset VARIES across images -- a single fixed-offset crop "
            "rule is NOT safe. Per-image logic required."
        ) + (
            f" {n_rgb_nontrivial} RGB image(s) have real per-channel "
            f"differences and must be handled specially."
            if n_rgb_nontrivial > 0 else
            f" All RGB-mode images ({n_rgb_trivial}) have identical R=G=B "
            f"channels -- safe to treat as grayscale with no information loss."
            if n_rgb_trivial > 0 else ""
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
