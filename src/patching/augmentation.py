"""
augmentation.py
=================
Deterministic, seedable augmentations for grayscale microstructure patches.
Only geometric + intensity transforms that make physical sense for SEM
micrographs: flips and 90-degree rotations (a grain structure has no
canonical orientation, so these are label-preserving), and brightness/
contrast jitter (SEM brightness/contrast varies by instrument settings, not
by phase identity, so jittering it doesn't change ground truth).

Deliberately excludes anything that could distort phase boundary geometry
(elastic deformation, aggressive rotation+crop) without first checking
whether that's appropriate for a rare-class boundary-sensitive task -- do
not add these without re-verifying they don't hurt Boundary F1 on TiN.
"""

import numpy as np


class Augmentor:
    def __init__(self, seed: int = 0, brightness_jitter: float = 0.1,
                 contrast_jitter: float = 0.1, flip_prob: float = 0.5,
                 rotate_prob: float = 0.5):
        self.rng = np.random.RandomState(seed)
        self.brightness_jitter = brightness_jitter
        self.contrast_jitter = contrast_jitter
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob

    def __call__(self, img_patch: np.ndarray, lbl_patch: np.ndarray):
        img = img_patch.copy()
        lbl = lbl_patch.copy()

        # horizontal flip
        if self.rng.rand() < self.flip_prob:
            img = np.fliplr(img)
            lbl = np.fliplr(lbl)
        # vertical flip
        if self.rng.rand() < self.flip_prob:
            img = np.flipud(img)
            lbl = np.flipud(lbl)
        # 90-degree rotation (0/1/2/3 * 90)
        if self.rng.rand() < self.rotate_prob:
            k = self.rng.randint(1, 4)
            img = np.rot90(img, k)
            lbl = np.rot90(lbl, k)

        # brightness/contrast jitter -- IMAGE ONLY, never touches the label
        img = img.astype(np.float32)
        brightness_delta = self.rng.uniform(-self.brightness_jitter, self.brightness_jitter) * 255
        contrast_scale = 1.0 + self.rng.uniform(-self.contrast_jitter, self.contrast_jitter)
        img = (img - 127.5) * contrast_scale + 127.5 + brightness_delta
        img = np.clip(img, 0, 255).astype(np.uint8)

        return np.ascontiguousarray(img), np.ascontiguousarray(lbl)
