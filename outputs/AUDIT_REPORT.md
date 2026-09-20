# MicroSteel Pipeline — Audit Report

Generated: 2026-09-20T06:44:12.350086Z

This report is generated directly from `outputs/audit_findings.json`, which is itself the return value of each stage in `run_all.py` run against the raw dataset. Nothing below is hand-typed after the fact.

## 1. Inventory Audit — can folder names be trusted?

**Question:** Can folder name be trusted as processing type?

**Conclusion:** Folder names are UNRELIABLE. Downstream code must parse true type from the filename prefix, never from the containing folder.

## 2. Pairing Audit — does every label have an image?

**Question:** Does every label have a matching image and vice versa?

**Conclusion:** 79 usable image-label pairs found (out of 82 labels). 3 label(s) have NO source image and must be excluded from training -- they cannot be used no matter what the split files say.

## 3. Geometry Audit — crop offset, resolution, color mode

**Question:** Do image/label dims match, is resolution constant, are all images grayscale?

**Conclusion:** Crop rule is CONSTANT: every image is exactly 65px taller than its label (width unchanged). Rule: crop each image to its label's own height -- this is relative, not a fixed target resolution, since resolution varies across the dataset. Visual overlay confirms spatial alignment is correct. All RGB-mode images (5) have identical R=G=B channels -- safe to treat as grayscale with no information loss.

## 4. Codebook Audit — decoding integer mask values

**Question:** What phase does each integer code represent, and is it consistent?

**Conclusion:** Integer<->RGB mapping is 100% consistent across all 79 usable images (zero exceptions). Decoded mapping: {4: 'Fe2B', 3: 'FeTiB', 0: 'Alpha', 1: 'TiB2', 2: 'TiN'}. Rare class 'TiN' decoded as integer code 2.

## 5. Split Audit — split file integrity & TiN stratification

**Question:** Are split files internally consistent, typo-free, and TiN-stratified?

**Conclusion:** Split files are internally consistent. 4 typo'd entries found (e.g. double extensions) -- split parsing MUST normalize extensions, not string-match literally. TiN pixel share across splits: train=0.144%, val=0.2241%, test=0.2303%. Reasonably balanced -- official splits are usable as-is (after excluding orphan stems with no image).

## 6. Manifest Build

Usable pairs: **79**, excluded: **3**
  - `Class_III_C104F2x10000_6` (III): no_matching_image_file
  - `Class_II_C28F3x10000` (II): no_matching_image_file
  - `Class_II_C30F3x10000_2` (II): no_matching_image_file

## 7. Manifest Validation

**Result:** VALIDATION PASSED: all 79 rows satisfy every invariant (pixel accounting, class legality per type, crop consistency, split validity).

Warnings:
  - [WARN] Type II: manifest has 15 usable images, paper states 17. This is EXPECTED if orphan labels were excluded -- verify against excluded_labels.csv, don't just suppress this warning.
  - [WARN] Type III: manifest has 35 usable images, paper states 36. This is EXPECTED if orphan labels were excluded -- verify against excluded_labels.csv, don't just suppress this warning.

## 8. Patch Size Decision (TiN Speck Analysis)

**Conclusion:** TiN specks are tiny (median max-dim 9.0px, p99 23.0px, largest ever seen 39px). Speck SIZE was never the constraint -- boundary PLACEMENT was: non-overlapping tiling splits several percent of specks regardless of patch size. Recommendation: patch_size=256, stride=128 (50.0% overlap) achieves 100.0% speck containment at 44.3 patches/image on average -- large patch sizes with fewer patches/image were rejected because they leave too little sub-image granularity for patch-level oversampling to mean anything.

**Fragmentation simulation results:**

| Patch | Stride | Overlap % | Total patches | % specks intact |
|---|---|---|---|---|
| 128 | 128 | 0.0 | 4448 | 90.43 |
| 128 | 64 | 50.0 | 14770 | 100.0 |
| 256 | 256 | 0.0 | 1112 | 95.76 |
| 256 | 128 | 50.0 | 3500 | 100.0 |
| 512 | 512 | 0.0 | 334 | 99.5 |
| 512 | 256 | 50.0 | 666 | 100.0 |

**Final recommendation:** patch_size=256, stride=128 (50.0% overlap)

---
*End of generated report.*