# MicroSteel Pipeline — Audit Report

Generated: updated_after_type_specific_sampler_change

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

**Conclusion:** TiN specks are tiny (median max-dim 9.0px, p99 23.0px, largest ever seen 39px). Speck SIZE was never the constraint -- boundary PLACEMENT was: non-overlapping tiling splits several percent of specks regardless of patch size. Recommendation: patch_size=256, stride=128 (50.0% overlap) achieves 100.0% speck containment at 62.6 patches/image on average -- large patch sizes with fewer patches/image were rejected because they leave too little sub-image granularity for patch-level oversampling to mean anything.

**Fragmentation simulation results:**

| Patch | Stride | Overlap % | Total patches | % specks intact |
|---|---|---|---|---|
| 128 | 128 | 0.0 | 6288 | 90.43 |
| 128 | 64 | 50.0 | 20888 | 100.0 |
| 256 | 256 | 0.0 | 1572 | 95.76 |
| 256 | 128 | 50.0 | 4949 | 100.0 |
| 512 | 512 | 0.0 | 472 | 99.5 |
| 512 | 256 | 50.0 | 942 | 100.0 |

**Final recommendation:** patch_size=256, stride=128 (50.0% overlap)

## 9. Patch Index Build

**Conclusion:** Built 4949 patches from 79 images (256px, stride 128). Per-split: {'train': 3311, 'val': 756, 'test': 882}, of which TiN-containing: {'train': 1686, 'val': 428, 'test': 481}.

## 10. Patch Index Validation (incl. cross-check vs. speck analysis)

**Result:** VALIDATION PASSED: 4949 patches checked, all structurally consistent with manifest. Patch count also MATCHES the fragmentation simulation's independent prediction (4949) -- the two separately-computed grids agree.

Cross-check: fragmentation simulation predicted **4949** total patches; actual patch index built **4949** — MATCH.

## 11. Weighted Sampler (TiN oversampling)

**Conclusion:** RESOLVED: rejected binary-presence oversampling and the old pooled absolute target. Adopted pixel-content-weighted sampling with a type-specific target: 2x each type's own natural TiN pixel fraction, capped at 0.004. Each processing type is solved and simulated independently using the exact train patch population its experiment will sample from. All three type-specific simulations meet their target within 10% and pass the concentration guard: PASS. The cap is necessary because the pixel-content sampler has a hard dataset-specific feasibility ceiling around 0.004-0.005.

## 12. Dataset Extraction & Augmentation Verification

**Result:** VERIFIED: extract_patch() output matches patch_index.csv's precomputed pixel counts exactly across 200 sampled patches (bit-for-bit, not just shape). Augmentation preserves label validity (class set and per-class pixel counts unchanged) across 50 sampled patches.

## 13. Eval Metrics Verification (IoU, Boundary F1, HD95)

**Result:** VERIFIED: all synthetic hand-computed cases, the missed_entirely/false_positive_only regression test, tiny-object edge cases, and a real-data self-match sanity check all pass. Metric contract is frozen -- see metrics.py module docstring table before changing any of these functions.

Real-data self-match sanity check used `Class_I_C103F1x10000` (classes present: [0, 3]).

## 14. Eval Aggregation Verification

**Result:** VERIFIED: aggregate.py correctly includes false-positive-only zeros in IoU/F1 means (does not silently filter to GT-present only), and HD95 correctly separates computed-value means from missed/false-positive/absent counts.

## 15. Patch Reconstruction Verification

**Result:** VERIFIED on genuine TiN-containing images across 3 type(s) (['I', 'II', 'III']): exact recovery (0 differing pixels, TiN pixels specifically checked, not just overall count), overlap-averaging beats mean single-patch accuracy under noise (0.9700 vs 0.8800), missing-patch coverage failures raise loudly, and non-probability inputs (raw logits/scores) are rejected rather than silently averaged into a meaningless result.

## 16. Full Evaluator End-to-End Verification (TiN baseline)

**Result:** VERIFIED end-to-end on 6 real TiN-containing images balanced across 3 processing type(s) (['I', 'II', 'III']): the ground-truth path achieves perfect scores across every class present, proving reconstruct->metrics->aggregate wiring is correct. The production predict_fn interface is confirmed image-only (no code path can pass it a label). The majority-class predictor establishes a trivial baseline for interpreting TiN recovery (IoU=0.0, all cases correctly missed_entirely) -- not a pass/fail threshold, just a sanity floor. Split enforcement and per_image_results presence both verified.

Majority-class (always-Alpha) baseline on 6 real TiN-containing images:
```
Class     IoU (n)             BoundaryF1 (n)      HD95 (n_computed)     missed  false_pos 
Alpha     0.763 (6)           0.279 (6)           56.78 (6)             0       0         
TiB2      0.000 (6)           0.000 (6)           n/a (0)               6       0         
TiN       0.000 (6)           0.000 (6)           n/a (0)               6       0         
FeTiB     0.000 (2)           0.000 (2)           n/a (0)               2       0         
Fe2B      0.000 (2)           0.000 (2)           n/a (0)               2       0         
```

## 17. Class-Weighted CE — Weight Computation & Verification

**Result:** VERIFIED (including per-type weights): weights for a type are computed only from that type's own train pixels over only the classes present in that type, matching an independent recomputation from the raw px columns, reproducing the hand-checked Type II reference [0.6356, 1.3924, 10.7524], differing from the pooled weights, and the real training entry point is confirmed (via a spy on train_one_experiment) to pass true_type through to build_loss. VERIFIED: hand-computed inverse-frequency formula matches manual calculation exactly, inverse_sqrt_freq is confirmed to be exactly sqrt(inverse_freq), the beta-underflow degenerate case correctly raises instead of silently returning uniform weights, weighted CE demonstrably shifts batch loss toward the higher-weighted class in a real behavioral test (not just a weight-value check), and on real data every scheme correctly gives TiN (rarest) the highest weight and Alpha (most common) the lowest.

**Weights actually used in training — `inverse_sqrt_freq`, computed per processing type from that type's own train pixels (absent classes show the inert placeholder 1.0):**

- Type I: {'Alpha': 0.5906, 'TiB2': 4.4574, 'TiN': 44.1562, 'FeTiB': 0.9612, 'Fe2B': 1.0}
- Type II: {'Alpha': 0.6356, 'TiB2': 1.3924, 'TiN': 10.7524, 'FeTiB': 1.0, 'Fe2B': 1.0}
- Type III: {'Alpha': 0.5871, 'TiB2': 2.3693, 'TiN': 11.2758, 'Fe2B': 1.0466, 'FeTiB': 1.0}

**Pooled-across-types weights (informational scale reference only, NOT used for training):**

- `inverse_freq`: {'Alpha': 0.271, 'TiB2': 3.672, 'TiN': 138.936, 'FeTiB': 1.987, 'Fe2B': 1.919}
- `inverse_sqrt_freq`: {'Alpha': 0.52, 'TiB2': 1.916, 'TiN': 11.787, 'FeTiB': 1.41, 'Fe2B': 1.385}
- `effective_number`: {'Alpha': 0.327, 'TiB2': 0.337, 'TiN': 3.683, 'FeTiB': 0.327, 'Fe2B': 0.327}

---
*End of generated report.*