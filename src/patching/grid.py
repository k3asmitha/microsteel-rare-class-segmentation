"""
grid.py
========
ONE implementation of "given an image of size (H, W), what are the patch
origins for a given patch_size/stride". Used by BOTH the speck-fragmentation
simulation (src/analysis/tin_speck_analysis.py) and the real patch indexer
(patch_index_builder.py). These must never be two separate implementations
-- if they drift, the "0% fragmentation" guarantee measured by the analysis
stage stops meaning anything for the patches actually produced.
"""


def compute_patch_origins(H: int, W: int, patch_size: int, stride: int):
    """Returns list of (y0, x0) top-left origins covering the full image.
    Always includes a final row/column anchored at (H-patch_size, W-patch_size)
    if the regular stride grid would leave a gap at the edge, so no image
    content is ever left untiled -- this matters for small images where
    patch_size doesn't evenly divide (H, W)."""
    if H < patch_size or W < patch_size:
        raise ValueError(
            f"Image ({H}x{W}) is smaller than patch_size {patch_size} -- "
            f"cannot tile. Handle undersized images upstream (pad or skip)."
        )

    ys = list(range(0, H - patch_size + 1, stride))
    if not ys or ys[-1] + patch_size < H:
        ys.append(H - patch_size)

    xs = list(range(0, W - patch_size + 1, stride))
    if not xs or xs[-1] + patch_size < W:
        xs.append(W - patch_size)

    return [(y0, x0) for y0 in ys for x0 in xs]
