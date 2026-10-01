"""
dataset_torch.py
===================
torch Dataset over patch_index.csv rows for one (true_type, split)
combination, built on top of the ALREADY-VERIFIED extract_patch and
Augmentor from src/patching/ -- this file does not reimplement patch
extraction or augmentation logic, only wraps it for torch's DataLoader.
"""

import csv
import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.patching.dataset import extract_patch
from src.patching.augmentation import Augmentor


def load_rows(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


class MicroSteelTorchDataset(Dataset):
    def __init__(self, data_root: str, manifest_rows: list, patch_rows: list,
                 true_type: str, split: str, augment: bool = False, seed: int = 0):
        self.data_root = data_root
        self.manifest_by_stem = {r["stem"]: r for r in manifest_rows}
        self.patch_rows = [r for r in patch_rows
                            if r["true_type"] == true_type and r["split"] == split]
        if not self.patch_rows:
            raise ValueError(f"No patches found for true_type={true_type}, split={split}")
        self.augmentor = Augmentor(seed=seed) if augment else None

    def __len__(self):
        return len(self.patch_rows)

    def __getitem__(self, idx):
        p = self.patch_rows[idx]
        src = self.manifest_by_stem[p["source_stem"]]
        patch_size = int(p["patch_size"])
        img_patch, lbl_patch = extract_patch(self.data_root, src, int(p["y0"]), int(p["x0"]), patch_size)

        if self.augmentor is not None:
            img_patch, lbl_patch = self.augmentor(img_patch, lbl_patch)

        img_tensor = torch.from_numpy(img_patch.astype(np.float32) / 255.0).unsqueeze(0)
        lbl_tensor = torch.from_numpy(lbl_patch.astype(np.int64))
        return img_tensor, lbl_tensor

    def tin_pixel_counts(self):
        """Returns list of tin_pixel_count aligned with self.patch_rows --
        used by the weighted sampler, which needs this WITHOUT loading
        every image (already precomputed in patch_index.csv)."""
        return [int(r["tin_pixel_count"]) for r in self.patch_rows]
