"""
dataset.py
===========
Core patch extraction: given a manifest row + patch bbox, return the actual
(image_patch, label_patch) numpy arrays. This is deliberately
framework-agnostic (returns numpy, not torch tensors) so it's testable here
without a torch install, and reusable regardless of which architecture
(DeepLabV3+ or Attention U-Net) consumes it.

extract_patch() applies the crop rule from the manifest (image is cropped
to the label's height BEFORE slicing out the patch) -- this is the same
crop_target_height/width already verified in a03_geometry_audit and used by
patch_index_builder.py. Getting this wrong here would silently misalign
every patch even though the manifest itself is correct.
"""

import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def extract_patch(data_root: str, manifest_row: dict, y0: int, x0: int, patch_size: int):
    """Returns (image_patch, label_patch) as numpy arrays, both
    (patch_size, patch_size), correctly cropped and aligned."""
    img_path = os.path.join(data_root, manifest_row["image_path"])
    lbl_path = os.path.join(data_root, manifest_row["label_path"])

    crop_h = int(manifest_row["crop_target_height"])
    crop_w = int(manifest_row["crop_target_width"])

    img = Image.open(img_path).convert("L")  # force single-channel; verified
    # safe in a03_geometry_audit (all RGB images have R==G==B exactly)
    img = img.crop((0, 0, crop_w, crop_h))   # apply the verified crop rule
    img_arr = np.array(img)

    lbl_arr = np.array(Image.open(lbl_path))

    assert img_arr.shape == (crop_h, crop_w), (
        f"{manifest_row['stem']}: cropped image shape {img_arr.shape} != "
        f"expected ({crop_h},{crop_w})"
    )
    assert lbl_arr.shape == (crop_h, crop_w), (
        f"{manifest_row['stem']}: label shape {lbl_arr.shape} != "
        f"expected ({crop_h},{crop_w})"
    )

    img_patch = img_arr[y0:y0 + patch_size, x0:x0 + patch_size]
    lbl_patch = lbl_arr[y0:y0 + patch_size, x0:x0 + patch_size]

    assert img_patch.shape == (patch_size, patch_size), (
        f"{manifest_row['stem']}: image patch shape {img_patch.shape} at "
        f"(y0={y0},x0={x0}) != ({patch_size},{patch_size}) -- bbox out of bounds?"
    )
    assert lbl_patch.shape == (patch_size, patch_size)

    return img_patch, lbl_patch


# ---------------------------------------------------------------------------
# Optional torch Dataset wrapper. Guarded import so this module (and the
# framework-agnostic extract_patch above) stays importable/testable in
# environments without torch installed.
# ---------------------------------------------------------------------------
try:
    import torch
    from torch.utils.data import Dataset
    _TORCH_AVAILABLE = True
except ImportError:
    Dataset = object
    _TORCH_AVAILABLE = False


class MicroSteelPatchDataset(Dataset):
    """torch Dataset over patch_index.csv rows for a given split.

    Requires torch -- raises a clear error at construction time if it's not
    installed, rather than failing confusingly deep inside __getitem__.
    """

    def __init__(self, data_root: str, manifest_rows: list, patch_rows: list,
                 augment=None, num_classes: int = 5):
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "MicroSteelPatchDataset requires torch, which is not installed "
                "in this environment. extract_patch() above works without it "
                "if you just need to test patch extraction logic."
            )
        self.data_root = data_root
        self.manifest_by_stem = {r["stem"]: r for r in manifest_rows}
        self.patch_rows = patch_rows
        self.augment = augment
        self.num_classes = num_classes

    def __len__(self):
        return len(self.patch_rows)

    def __getitem__(self, idx):
        p = self.patch_rows[idx]
        src = self.manifest_by_stem[p["source_stem"]]
        patch_size = int(p["patch_size"])
        img_patch, lbl_patch = extract_patch(
            self.data_root, src, int(p["y0"]), int(p["x0"]), patch_size
        )

        if self.augment is not None:
            img_patch, lbl_patch = self.augment(img_patch, lbl_patch)

        img_tensor = torch.from_numpy(img_patch.astype(np.float32) / 255.0).unsqueeze(0)
        lbl_tensor = torch.from_numpy(lbl_patch.astype(np.int64))
        return img_tensor, lbl_tensor
