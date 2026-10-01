# MicroSteel Pipeline — Person A Infra

This is not a narrative of things that were run once and reported. Every
number in `outputs/AUDIT_REPORT.md` is regenerated from the raw dataset
every time you run `run_all.py`. If you change the dataset, rerun it, and
trust the new report — don't trust anything written in chat about it.

## Project structure

```
microsteel_project/
├── configs/
│   └── config.yaml          # ALL paths + parameters live here, nowhere else
├── src/
│   ├── common.py             # shared utilities (type parsing, image mapping) — single implementation
│   ├── audit/
│   │   ├── a01_inventory_audit.py   # can folder names be trusted?
│   │   ├── a02_pairing_audit.py     # does every label have a matching image?
│   │   ├── a03_geometry_audit.py    # image/label crop offset, resolution, color mode
│   │   ├── a04_codebook_audit.py    # decode integer mask codes -> phase names
│   │   └── a05_split_audit.py       # split file integrity, TiN stratification
│   ├── manifest/
│   │   ├── build_manifest.py        # builds outputs/manifest.csv from audit results
│   │   └── validate_manifest.py     # checks every row against invariants
│   ├── analysis/
│   │   └── tin_speck_analysis.py    # measures TiN speck sizes, decides patch_size/stride
│   ├── patching/
│   │   ├── grid.py                  # ONE canonical patch-grid function (shared with analysis stage)
│   │   ├── patch_index_builder.py   # builds outputs/patch_index.csv (per-patch stats)
│   │   ├── validate_patch_index.py  # validates + cross-checks vs. fragmentation simulation
│   │   ├── sampler.py               # TiN-oversampling weights + simulated verification
│   │   ├── dataset.py               # patch pixel extraction (+ optional torch Dataset)
│   │   ├── augmentation.py          # deterministic flip/rotate/brightness augmentations
│   │   └── verify_dataset.py        # cross-checks extraction against patch_index, tests augmentation
│   ├── eval/
│   │   ├── metrics.py               # confusion matrix, per-class IoU, Boundary F1, HD95 — FROZEN CONTRACT
│   │   ├── aggregate.py             # aggregates per-image metrics into per-class summaries
│   │   ├── reconstruct.py           # reconstructs full images from overlapping patch predictions (probability-averaging)
│   │   ├── evaluator.py             # full reconstruct->metrics->aggregate pipeline + dummy predictors
│   │   ├── verify_metrics.py        # hand-computed synthetic tests + tiny-object edge cases + real-data sanity check
│   │   ├── verify_aggregate.py      # verifies the false-positive-inclusion aggregation rule
│   │   ├── verify_reconstruct.py    # real-data exact-recovery test + noise-averaging test
│   │   └── verify_evaluator.py      # end-to-end smoke test on real TiN images, establishes the baseline any model must beat
│   └── loss/
│       ├── class_weights.py         # class-weighted CE weight computation (3 schemes), from real per-pixel counts
│       └── verify_class_weights.py  # hand-computed test, beta-underflow regression, behavioral weighted-CE check
├── run_all.py                 # THE entrypoint — runs all 17 fast infra stages in order
├── src/models/                 # DeepLabV3+, Attention U-Net, factory, verification (see below)
├── src/training/                # training loop, torch dataset/sampler, losses, run_experiment.py
├── src/inference/                # checkpoint loading -> frozen-evaluator predict_fn adapter
├── requirements.txt
├── evidence/                  # visual proof (crop alignment overlay, removed metadata strip)
└── outputs/                   # generated — manifest.csv, patch_index.csv, AUDIT_REPORT.md, etc.
```

## How to run it

```bash
cd microsteel_project
pip install -r requirements.txt

# Edit configs/config.yaml first: set paths.data_root to wherever you
# extracted microsteel_dataset.zip (must contain images_class*/, labels/, splits/)

python3 run_all.py
```

That's the only command you need. It runs all 17 stages, in order:
1. `a01` → `a02` → `a03` → `a04` → `a05` (raw-data audits)
2. `build_manifest.py` (only runs if all audits pass — a04 in particular
   will hard-fail the whole run if the integer↔RGB codebook turns out to be
   inconsistent, because building a manifest on top of that would be
   silently wrong)
3. `validate_manifest.py` (checks every row of the manifest just built)
4. `tin_speck_analysis.py` (measures speck geometry, decides patch size)
5. `patch_index_builder.py` (enumerates every patch position + per-class stats)
6. `validate_patch_index.py` (checks every patch row, cross-checks total
   count against stage 4's independent simulation)
7. `sampler.py` (computes TiN-oversampling weights, verifies achieved
   sampling distribution by actual simulated draws)
8. `verify_dataset.py` (confirms real pixel extraction matches what stage 5
   indexed, tests augmentation invariants)
9. `verify_metrics.py` (hand-computed IoU/Boundary F1/HD95 tests, the
   missed-vs-false-positive regression test, tiny-object edge cases, a
   real-data self-match sanity check)
10. `verify_aggregate.py` (confirms false-positive-only cases are correctly
    included in aggregated means, not silently filtered out)
11. `verify_reconstruct.py` (real-data exact-recovery test on genuine
    TiN-containing images across all 3 types, noise-averaging test,
    coverage-failure and invalid-probability rejection tests)
12. `verify_evaluator.py` (full end-to-end reconstruct→metrics→aggregate
    smoke test on real TiN images balanced across all 3 types, establishes
    the majority-class baseline, confirms the production `predict_fn`
    interface never receives a label, confirms split enforcement)
13. `verify_class_weights.py` (hand-computed class-weight test, the
    beta-underflow regression test, a behavioral check that weighted CE
    actually shifts batch loss toward the higher-weighted class)

Exit code is `0` only if every hard check passes. Non-zero means something
is broken and the manifest/patch index/eval module/class weights should not
be trusted or used for training.

Standalone, any single stage module can also be run directly, e.g.:
```bash
python3 -m src.patching.patch_index_builder
python3 -m src.patching.validate_patch_index
python3 -m src.patching.sampler
python3 -m src.patching.verify_dataset
python3 -m src.eval.verify_metrics
python3 -m src.eval.verify_aggregate
python3 -m src.eval.verify_reconstruct
python3 -m src.eval.verify_evaluator
python3 -m src.loss.verify_class_weights
```

## How to verify it worked

Don't take my word for it — check these yourself:

1. **Run it twice, diff the output.** `outputs/manifest.csv` and
   `outputs/AUDIT_REPORT.md` should be byte-identical (aside from the
   timestamp line) between runs, since everything is deterministically
   recomputed from the same raw files.
   ```bash
   python3 run_all.py && cp outputs/manifest.csv /tmp/run1.csv
   python3 run_all.py && diff /tmp/run1.csv outputs/manifest.csv
   ```
2. **Read `outputs/AUDIT_REPORT.md` top to bottom.** Each section states a
   question, then a conclusion computed from that run. Section 8's
   fragmentation table is the actual evidence behind the patch_size/stride
   choice in `configs/config.yaml` — check the table yourself rather than
   trusting the "recommendation" line.
3. **Look at the evidence images.** `evidence/removed_bottom_strip.png` is
   the SEM instrument metadata bar being cropped away — open it and confirm
   it's really an instrument bar, not image content. `evidence/crop_alignment_overlay.png`
   overlays the label mask on the cropped image in red — confirm the red
   region boundary actually tracks real grain edges, not misaligned by even
   a few pixels.
4. **Run `validate_manifest.py` standalone** any time you're suspicious of
   the manifest, without rerunning the whole audit:
   ```bash
   python3 -m src.manifest.validate_manifest
   ```
   Exit code 0 + "VALIDATION PASSED" in stdout is the bar. Anything else
   means don't train on this manifest yet.
5. **Run any single audit standalone** to inspect its raw findings as JSON,
   e.g.:
   ```bash
   python3 -m src.audit.a04_codebook_audit
   ```
   This prints the full decode evidence (which pixel percentage matched
   which expected phase name, and why) — useful if you ever add a new
   processing type and need to confirm the codebook still decodes cleanly.

## What each audit actually checks (and why it exists)

| Module | Question | Why it matters |
|---|---|---|
| `a01_inventory_audit` | Do folder names (`images_class1/2/3`) match the true type in each filename? | **They don't** — `images_class2/` mixes 15 real Class_II images with 10 Class_III images. Any code trusting folder names silently mislabels 10 images. |
| `a02_pairing_audit` | Does every label have a source image? | **3 don't.** These are hard-excluded — no amount of clever code recovers a missing source image. |
| `a03_geometry_audit` | Do image/label dimensions match? Is resolution constant? Are all images single-channel? | Every image is 65px taller than its label (SEM metadata bar) — verified visually, not just by pixel-count coincidence. Resolution is NOT constant (one outlier at 960×768), so crop logic must be relative, never a hardcoded target size. |
| `a04_codebook_audit` | What phase does each integer mask value represent? | Decoded via type-presence signature + pixel-percentage matching against the paper's stated values — not hardcoded from memory. Fails loudly if the integer↔RGB mapping is ever inconsistent, which would make a global codebook unsafe. |
| `a05_split_audit` | Are the official train/val/test splits internally consistent, typo-free, and TiN-stratified? | Found a `.jpg.png` double-extension typo in `val_c1.txt` (4 entries) that a literal string-match loader would silently drop. TiN pixel share across splits is within 2x, so the official splits are usable as-is once the typo and orphans are handled. |

## Known findings baked into `configs/config.yaml`

- **`patching.patch_size: 256`, `patching.stride: 128`** — not a default,
  a measured decision. TiN specks are tiny (p99 max-dimension 23px, largest
  ever seen 39px), so speck *size* was never the constraint; boundary
  *placement* was. Any patch size at 50% overlap achieves 0% fragmentation,
  so the choice among those came down to compute cost: 256/128 achieves
  identical (100%) speck containment to 128/64 at **4.2x fewer total
  patches**. See `outputs/AUDIT_REPORT.md` section 8 for the full
  fragmentation table across all six candidates that were actually tested.
- **Real per-type TiN rarity is uneven**: Type I ≈0.011%, Type II ≈0.344%,
  Type III ≈0.228% (from `a04_codebook_audit`, cross-checked in
  `manifest.csv`'s `pct_TiN` column). Type I's TiN is roughly 20–30x rarer
  than in the other two types — factor this into any claim that a method
  "generalizes across types," since a difference in outcome could just be
  this rarity gap, not the method or architecture.
- **Real usable counts**: Type I = 29, Type II = 15, Type III = 35 (79
  total), not the paper's stated 29/17/36 (82 total) — the 3-image gap is
  the orphan labels found by `a02_pairing_audit`.

## Next step

The manifest, patch pipeline, and eval module are all built and verified —
see "What's left before B and C can start training" further down in this
document for the actual current status.

## Bugs found and fixed WHILE building the patching stage (stages 8-12)

Documented here because they're exactly the kind of thing that looks fine
until you check the numbers, and they change conclusions stated earlier:

1. **Patch-recommendation logic picked the wrong optimization target twice.**
   First it minimized `patch_size` directly instead of total compute cost
   (picked 128/64 over the cheaper, equally-effective 256/128). After fixing
   that, a second run exposed that the ">=10 patches/image" granularity
   floor was an arbitrary guess that let 512/256 sneak through as "cheapest"
   — exactly the coarse-tiling outcome the floor was meant to prevent. Fixed
   by raising the floor to >=20/image and labeling it explicitly as an
   engineering judgment call, not a derived number.
2. **`simulate_fragmentation()` silently returned 0 patches for any image
   with zero TiN specks** (23 of 79 images), undercounting every "total
   patches" figure in the recommendation table by ~30%. This didn't change
   which config had 0% fragmentation, but it did affect the total-patch-count
   comparison used to pick among zero-fragmentation candidates. Fixed, then
   the whole table was regenerated and cross-checked against the real patch
   indexer (see #3).
3. **`validate_patch_index.py`'s cross-check against the fragmentation
   simulation ran before `audit_findings.json` existed** on a clean run —
   silently degrading to "can't compare" while the success message still
   claimed "MATCHES." Fixed the write ordering in `run_all.py` and the
   misleading message; the real cross-check (4949 == 4949, computed by two
   independently-implemented code paths) is what's now in the report.
4. **`build_image_map()` reported spurious cross-folder "duplicates"** on
   case-insensitive filesystems (macOS/Windows) because it globbed each
   folder with `("*.jpg","*.jpeg","*.JPG")` — the same physical file could
   match two of those patterns and get recorded as a duplicate of itself.
   Fixed by switching to a single `os.listdir()` pass with an explicit
   lowercase extension check, so it can't double-count regardless of the
   underlying filesystem. (Found by a second reviewer checking the report
   output, not by this pipeline's own tests — worth having someone else
   read the audit output, not just re-run it.)

## Sampler: type-specific TiN pixel-content oversampling — LOCKED

**Rejected:** binary "contains any TiN pixel" oversampling. With 50% patch
overlap, ~51% of train patches already contain at least one TiN pixel with
zero oversampling — some from a single overlap-sliver pixel, not real speck
content.

**Adopted:** sample proportional to each patch's actual TiN pixel *content*
(continuous, not a presence/absence flag), capped at the 90th percentile
among TiN-containing patches to prevent a few unusually large patches from
dominating every draw.

The target is now **type-specific**, because the three processing types have
different natural TiN prevalence. For each type, the target is:

`target = min(2.0 × that type's natural TiN pixel fraction, 0.004)`

The natural fraction is computed from the exact train patch pool that the
corresponding experiment will use. Therefore the sampling intervention is
not calibrated on a pooled population and is not accidentally a near-no-op
for one type or infeasible for another.

The resulting targets are approximately:

| Type | Natural TiN pixel fraction | Target | Effective multiplier |
|---|---:|---:|---:|
| I | 0.00013 | 0.00027 | 2.0x |
| II | 0.00283 | 0.00400 | 1.4x (0.004 cap) |
| III | 0.00195 | 0.00390 | 2.0x |

The 0.004 cap is an explicit feasibility/diversity ceiling discovered from
the patch-content distribution. The sampler solves against true uncapped
TiN pixel content while using capped content only to shape the sampling
weights. Every type is independently simulated over 50,000 draws and checked
for both target attainment and concentration before the audit passes.

This replaces the previous pooled `target_tin_pixel_fraction=0.003`, which
was infeasible for Type I and only a 1.06x intervention for Type II.


## Eval module (metrics.py + aggregate.py) — FROZEN CONTRACT

Per-class IoU, Boundary F1, and HD95 (95th-percentile Hausdorff), plus a
confusion matrix. This contract must not change once B and C start
training — all 24 runs need to be compared under the same metric
definitions. The frozen contract:

| Situation | IoU | Boundary F1 | HD95 |
|---|---|---|---|
| GT absent, Pred absent | None | None | None (`absent_from_both`) |
| GT present, Pred present | computed | computed | computed |
| GT present, Pred absent | 0.0 | 0.0 | None (`missed_entirely`) |
| GT absent, Pred present | 0.0 | 0.0 | None (`false_positive_only`) |

All three metrics share one `class_presence_status()` helper for this
branching, specifically so they can't silently disagree about what
"undefined" means for the same pixel data.

**A real bug was found and fixed here via external review, not by this
pipeline's own tests** (worth noting: automated tests only catch what
they're written to check — a second reviewer reading the actual code
caught something the test suite at the time didn't cover): an earlier
version of `hd95_per_class` used a single `if not pred_c.any() or not
target_c.any()` check that labeled BOTH "model never predicts this class"
and "model hallucinates this class everywhere GT doesn't have it" as
`missed_entirely`. These are opposite failure modes and must never be
conflated. Fixed by splitting into `missed_entirely` and
`false_positive_only`, with a regression test in `verify_metrics.py` that
explicitly checks the two cases get different labels.

**Aggregation contract:** for IoU/Boundary F1, `false_positive_only` cases
(already 0.0, not None) MUST be included in any mean — filtering to
"GT-present only" would silently hide a model that hallucinates the rare
class everywhere. For HD95, `computed` values are averaged separately from
counts of `missed_entirely`/`false_positive_only`/`absent_from_both` —
never folded into one number. `verify_aggregate.py` has an explicit
anti-regression test asserting the mean is NOT the value you'd get from
the wrong "filter to GT-present" approach.

**Full evaluator (built and verified):** `reconstruct.py` reconstructs whole
images from overlapping patch predictions by averaging per-pixel
PROBABILITIES (never averaging hard class labels — categorical data can't
be averaged meaningfully) before taking argmax. Verified against real data,
not just synthetic grids: chopping a real label into its actual overlapping
patches, one-hot encoding each as a "perfect model," and reconstructing
recovers the original label bit-for-bit (0 differing pixels). A second test
adds independent random noise to each overlapping patch and confirms
reconstruction genuinely reduces error (97.8% accuracy vs 88.0% mean
single-patch accuracy) — proving overlap-averaging helps, not just that it
runs without crashing.

`evaluator.py` wires reconstruct → metrics → aggregate into the actual loop
B and C's trained models will plug into, with two dummy predictors that
make the whole pipeline testable right now, before any real model exists:
`predict_ground_truth` (must score perfectly — proves the wiring) and
`predict_majority_class` (always predicts Alpha — the trivial baseline any
real model's TiN performance must clearly beat to be worth reporting).
Verified end-to-end on 8 real TiN-containing images in
`verify_evaluator.py`: ground truth scores perfectly, majority-class
baseline correctly and completely fails on TiN (IoU=0.0 across all 8,
correctly labeled `missed_entirely`, zero false-positive miscounts).

**A second external review caught four more real gaps, verified and fixed:**
1. `reconstruct.py`'s docstring said probability arrays "or logits" were
   acceptable — false. Averaging raw logits before argmax is mathematically
   different from averaging post-softmax probabilities (softmax is
   nonlinear). Confirmed the code would silently accept an array summing to
   11.0 instead of 1.0 with zero complaint. Fixed by adding an explicit
   validation check (`probs.sum(axis=0) ≈ 1.0`) that raises `ValueError`
   with a clear message, and restricting the contract to probabilities only.
2. `verify_reconstruct.py` tested reconstruction on an arbitrary first
   Type I image that — as already noted once before — didn't happen to
   contain any TiN at all, missing the entire point of the pipeline. Fixed
   by deterministically selecting one genuine TiN-containing image per
   available processing type (I, II, III) and specifically checking that
   TiN pixels themselves (not just overall pixel count) are recovered
   exactly. Also fixed overclaimed wording ("beats any single patch") down
   to what the test actually measures ("beats the mean of individual
   patches" — a true but weaker claim than what was written).
3. `evaluate_dataset()` had no default-safety net: calling it with no
   arguments beyond a predict function would silently mix train, val, and
   test images into one "evaluation," which would be a real leakage bug the
   first time someone forgot to filter. Fixed: `evaluate_dataset` now
   requires at least one of `split` or `image_stems`, and if both are
   given, raises if any stem in `image_stems` doesn't actually belong to
   the declared split — verified in `verify_evaluator.py` with explicit
   regression tests for both the missing-argument and mismatched-split cases.
4. Added the full-class regression table the review specifically asked
   for: the ground-truth predictor is now checked for a perfect score on
   *every* class present in the tested images (Alpha, TiB2, TiN, FeTiB,
   Fe2B), not just TiN in isolation.

## Third external review — verified and fixed, plus one REJECTED claim

**Mandatory fixes, verified and implemented:**

1. **Label-leakage-permitting interface (confirmed real, fixed):** the
   evaluator previously called `predict_fn(image_patch, label_patch)` —
   even though a real model must never see the label, the interface itself
   allowed it. Fixed: production `predict_fn` now takes `image_patch` only,
   enforced by construction (there is no code path in `evaluate_image` that
   can pass a label to it). The one legitimate need for label access
   (verifying the reconstruct→metrics→aggregate chain without a trained
   model) now lives in `debug_evaluate_dataset_from_ground_truth`, a
   completely separate function with no `predict_fn` parameter at all —
   architecturally impossible to mistake for real model evaluation code.
   Regression-tested with a spy function that would raise `TypeError` if
   the evaluator ever tried to pass it a second argument.
2. **Aggregates-only, no per-image results (confirmed real, fixed):**
   `evaluate_dataset` now returns `per_image_results` (every image's
   individual iou/f1/hd95 plus stem/true_type/split/architecture/
   configuration identity) alongside the aggregates. Added
   `save_evaluation_results()`, writing `outputs/results/Type_X/
   <architecture>/<configuration>/per_image.csv` + `summary.json` — the
   structure needed to later answer "which image failed", "is this
   dominated by one image", "what goes in the failure-analysis section".
3. **Weak probability validation (confirmed real, fixed):** added
   `isfinite` and `[0,1]`-range checks in `reconstruct.py`, layered before
   the existing sum-to-1 check — catches NaN/Inf, negative values, and
   broken softmax output, not just wrong totals. All three checks have
   dedicated regression tests in `verify_reconstruct.py`.
4. **Overstated baseline wording (confirmed, fixed):** "any real model must
   beat this to be worth reporting" implied a pass/fail bar the
   majority-class dummy was never meant to be. Changed to "establishes a
   trivial baseline for interpreting TiN recovery" — accurate, no implied
   threshold.
5. **Unbalanced type coverage in `verify_evaluator.py` (confirmed, fixed):**
   `tin_containing_stems[:8]` didn't guarantee all three types were tested.
   Replaced with explicit per-type selection (2 images per type when
   available) — the rerun confirms Type I, II, and III are all genuinely
   exercised now, not just whichever types happened to sort first
   alphabetically.

**One claim REJECTED after checking the source document:** the review
asserted the "locked" plan used class-weighted cross-entropy and that
Tversky/focal loss was an unauthorized substitution. This directly
contradicts the actual project overview document provided at the start of
this project (which states, twice: *"implement and tune Tversky/focal loss
per type"* and *"Owns loss-weight tuning (Tversky/focal)"* under Person B's
ownership). No change made — an external review's assertion about what a
plan "was" doesn't override the actual source document, and should be
checked against it before acting, the same as every other claim in this
project.

## Fourth review — packaging and cross-platform issues, verified

1. **Stale README section (confirmed real):** a leftover "## Next step"
   section from the very first version of this README still said the
   patching code "hasn't been built yet," contradicting every later
   section that correctly describes it as done. Removed.
2. **Machine-specific config path (not a bug, but made impossible to miss):**
   `configs/config.yaml`'s `data_root` is specific to the sandbox this
   project was built in and was never going to match your machine — this
   was already documented in the README's "How to run it" section, but
   `config.yaml` itself now has an unmissable inline warning at the
   `data_root` line telling you to edit it, for anyone who opens the config
   directly without reading the README first.
3. **Windows encoding risk (confirmed real, fixed everywhere, not just where
   flagged):** file writes using `open(path, "w")` without an explicit
   encoding default to the OS's preferred encoding on write, which on
   Windows is typically cp1252, not UTF-8. `AUDIT_REPORT.md` contains
   characters outside cp1252 (⚠️, em-dashes), so writing it without
   `encoding="utf-8"` would raise `UnicodeEncodeError` on Windows. The
   review flagged 2 instances in `run_all.py`; a full codebase grep found
   **8 total** write locations missing this across 5 files, plus a
   symmetric risk on the **19 read locations** that load those same files
   back. All 27 locations now explicitly specify `encoding="utf-8"`, fixing
   the whole bug class rather than only the two lines pointed out. Full
   pipeline reverified afterward (all 16 stages pass, every standalone
   module spot-checked individually) since this was a mechanical
   find-and-replace across 12 files and needed re-verification, not just
   trust that the pattern was applied correctly everywhere.

## PLAN CHANGE: Tversky/focal → class-weighted cross-entropy

**This is an explicit, team-decided change to the original project document,
recorded here rather than silently applied.** The original project overview
specified Tversky/focal loss for the weighted-loss ablation condition (twice:
in the pipeline section and in Person B's ownership description). The team
has since decided to use plain class-weighted cross-entropy instead. This
replaces that part of the plan going forward.

`src/loss/class_weights.py` computes the actual weight tensor from real
per-pixel class counts in the train split — three schemes, all verified in
`verify_class_weights.py`:

| Scheme | Real TiN/Alpha ratio | Note |
|---|---|---|
| `inverse_freq` | **513.5x** | Classic "balanced" weighting. This large a per-pixel weight differential is a real stability risk — watch for loss spikes/NaN if used. |
| `inverse_sqrt_freq` | **22.7x** | Gentler correction. **Current default.** |
| `effective_number` (Cui et al.) | 11.3x–394x, beta-dependent | See critical finding below. |

**A real, silent-failure-mode bug was found and fixed while building this:**
the `effective_number` scheme's commonly-cited paper beta values (0.99,
0.999, 0.9999) are calibrated for class-level sample counts (tens to
thousands), not per-pixel counts in the tens of millions this dataset has.
Verified directly: at beta=0.999, `beta^n` underflows to exactly `0.0` in
float64 for Alpha's 47.6M pixels, degenerating the whole formula to
identical weights for every class — silently, with no error. `beta` must be
pushed to ~0.999999+ to get real differentiation at this dataset's scale.
`compute_class_weights` now raises `ValueError` if this degenerate case is
detected, rather than silently returning useless uniform weights.

**Verified behaviorally, not just by weight value:** `verify_class_weights.py`
constructs a toy batch where an "easy" Alpha prediction and a "wrong" TiN
prediction are mixed, and confirms that weighting TiN higher actually
increases the batch's mean loss relative to unweighted — proving the
weights shift training signal the way they're supposed to, not just that
the numbers look plausible in isolation. A pure-numpy reference
implementation of PyTorch's documented `nn.CrossEntropyLoss(weight=...)`
formula is provided in `weighted_ce_loss_numpy` for this verification (torch
itself isn't installed in this environment, but the formula is simple
elementwise arithmetic — identical regardless of framework — so this is not
a placeholder, it verifies the same mathematics B's real torch training
loop will execute).

**For Person B:** call `compute_class_weights(manifest_rows, class_names,
scheme, split="train")` to get a `{class_name: weight}` dict, convert to a
tensor aligned with your class-ID ordering, and pass to
`nn.CrossEntropyLoss(weight=...)`. Don't hardcode weight values — recompute
from the manifest if the dataset or splits ever change.

## Fifth review — class weights were pooled across types (REAL BUG, fixed)

**Confirmed and reproduced.** `train.py` passed *every* manifest row to
`build_loss`, and `compute_class_weights` had no `true_type` filter, so a
model trained only on Type II received class weights computed from Type I
+ II + III pixels. It was also wrong in a second way: `K` in the formula
counted FeTiB and Fe2B, which don't exist in Type II at all.

| `inverse_sqrt_freq` | Alpha | TiB2 | TiN |
|---|---|---|---|
| Pooled (old, wrong for Type II) | 0.5202 | 1.9162 | 11.7871 |
| **Type II only, K=3 (correct)** | **0.6356** | **1.3924** | **10.7524** |

The experiment design is unchanged (B/P/L/PL, same inverse-sqrt rule for
every type) — only the population the weights are computed from changed.
Per-type weights now in use:

| Type | Alpha | TiB2 | TiN | FeTiB | Fe2B |
|---|---|---|---|---|---|
| I | 0.5906 | 4.4574 | **44.1562** | 0.9612 | – |
| II | 0.6356 | 1.3924 | 10.7524 | – | – |
| III | 0.5871 | 2.3693 | 11.2758 | – | 1.0466 |

Classes absent from a type get an inert placeholder weight of 1.0 (no
target pixel ever has that class, and PyTorch's weighted-mean CE only sums
the weights of target classes).

**What changed:** `compute_class_weights` takes `true_type` +
`type_class_membership` and fails loudly if the declared membership and the
real pixel counts disagree; `build_loss` now *requires* `true_type` (no
default, so the mistake can't silently come back); `train.py` passes it.

**How it's guarded:** `verify_class_weights.py` recomputes the weights
independently from the raw `px_<Class>` columns (not by calling the function
under test), checks the Type II reference values, checks they differ from the
pooled ones, and spies on the *real* `train_one_experiment` to confirm
`true_type` reaches `build_loss`. Both ways of reintroducing the bug (the
function ignoring `true_type`; `train.py` passing the wrong type) were
mutation-tested and each is caught.

**Note for Type I:** TiN's inverse-sqrt weight is 44.2 vs Alpha's 0.59
(~75x) — much steeper than Type II (~17x). Watch Type I loss curves for
instability specifically.

## Sampler target decision — RESOLVED

The sampler target is no longer calibrated on the pooled train patch pool.
It is derived independently for each processing type as:

`target = min(2.0 × natural TiN pixel fraction for that type, 0.004)`

This avoids an infeasible Type-I target and avoids a near-no-op Type-II
intervention while respecting the empirically observed pixel-content
feasibility ceiling.


## Models, training, and inference — Member A's training-pipeline deliverable

Built on top of everything above: model wrappers for both architectures, a
common training loop, inference connected to the frozen evaluator, and
Person A's 8 Type II experiment runs are ready to launch.

### What's in `src/models/`, `src/training/`, `src/inference/`

| File | What it does |
|---|---|
| `src/models/deeplabv3plus.py` | DeepLabV3+ (EfficientNet-b0 encoder) via `segmentation-models-pytorch`. Single-channel input. |
| `src/models/attention_unet.py` | Attention U-Net (Oktay et al. 2018) built from scratch — `smp` doesn't provide this architecture. |
| `src/models/build_model.py` | Factory: architecture name → (model, predict_probs adapter). |
| `src/models/verify_models.py` | Forward-pass shape check, gradient-flow check (every parameter gets a real gradient), and probability-contract check for both architectures. |
| `src/training/dataset_torch.py` | torch `Dataset` wrapping the already-verified `extract_patch`/`Augmentor` — no patch logic reimplemented. |
| `src/training/sampler_torch.py` | Wraps the already-verified pixel-weighted sampler into a torch `WeightedRandomSampler`. |
| `src/training/losses.py` | Builds CE or class-weighted CE using the already-verified `class_weights.py`. |
| `src/training/train.py` | The common training loop: optimizer (Adam), LR schedule (`ReduceLROnPlateau`), checkpointing on best val loss, early stopping. |
| `src/training/run_experiment.py` | **The entrypoint.** Train → load best checkpoint → evaluate on test split via the frozen evaluator → save results. |
| `src/training/verify_training_pipeline.py` | Ultra-fast (~20s) end-to-end smoke test — run this FIRST on any new machine. |
| `src/inference/predict.py` | Loads a checkpoint, builds a `predict_fn(image_patch)` matching the frozen evaluator's image-only interface exactly. |

### A real gotcha found and fixed while building this

**BatchNorm requires batch size ≥2 in training mode** (both architectures
use it). The `DataLoader` uses `drop_last=True` specifically so a leftover
batch of size 1 at the end of an epoch never reaches the model and crashes
training — found by `verify_models.py` failing on a batch-size-1 gradient
test, not discovered mid-training on your machine.

**A random-init artifact, investigated and confirmed harmless:** with
`encoder_weights=None` (used only for testing in this network-restricted
sandbox), a whole MBConv block's Squeeze-Excitation gate can saturate to
exactly 0 for certain random seeds, zeroing that block's gradient
transiently. Verified across 4 seeds that it's a *different* block (or
absent) each time, not a structural defect — and it won't occur at all with
real pretrained weights (`encoder_weights="imagenet"`, the default for
actual training). Documented in `verify_models.py` rather than silently
ignored or wrongly hard-failed.

## How to run training — step by step

**1. Install the extra dependencies** (if not already):
```bash
pip install -r requirements.txt
```

**2. Verify your machine's setup first** (~20 seconds, no real training):
```bash
python3 -m src.models.verify_models
python3 -m src.training.verify_training_pipeline
```
Both must print `"passed": true` and exit 0. If either fails, fix that
before spending real time on training — a multi-hour training run failing
at the very last evaluation step because of a broken install wastes far
more time than catching it now.

**3. Realistic CPU timing (measured, not estimated):** on the machine this
was built on, both architectures take **~1.8 seconds per batch** at
`batch_size=4`. For Type II (567 train + 189 val patches), one full epoch
(~188 batches) is **~5–6 minutes**. With `early_stopping_patience=8` and up
to 30 epochs, expect **anywhere from ~45 minutes to ~3 hours per run**,
depending how quickly validation loss plateaus. **There is no GPU
anywhere in this project — plan around this timing, don't expect it to be
faster.**

**4. Run your first real experiment — get ONE working before the rest:**
```bash
python3 -m src.training.run_experiment \
  --architecture DeepLabV3Plus --configuration baseline \
  --true-type II --run-id II-DL-B
```
This trains, checkpoints, evaluates on the Type II test split, and writes:
- `outputs/training_runs/II-DL-B/best_checkpoint.pt`
- `outputs/training_runs/II-DL-B/training_history.csv` (per-epoch train/val loss, LR)
- `outputs/training_runs/II-DL-B/run_summary.json`
- `outputs/results/Type_II/DeepLabV3Plus/baseline/per_image.csv` + `summary.json`

**5. Once that looks right, run the remaining 7:**
```bash
python3 -m src.training.run_experiment --architecture DeepLabV3Plus --configuration patch   --true-type II --run-id II-DL-P
python3 -m src.training.run_experiment --architecture DeepLabV3Plus --configuration loss     --true-type II --run-id II-DL-L
python3 -m src.training.run_experiment --architecture DeepLabV3Plus --configuration both      --true-type II --run-id II-DL-PL
python3 -m src.training.run_experiment --architecture AttentionUNet --configuration baseline  --true-type II --run-id II-AUN-B
python3 -m src.training.run_experiment --architecture AttentionUNet --configuration patch     --true-type II --run-id II-AUN-P
python3 -m src.training.run_experiment --architecture AttentionUNet --configuration loss      --true-type II --run-id II-AUN-L
python3 -m src.training.run_experiment --architecture AttentionUNet --configuration both      --true-type II --run-id II-AUN-PL
```
Run these one at a time, not in parallel — CPU training already uses all
available cores per run; running several simultaneously just makes all of
them slower with no time saved.

## How to verify a completed run's output is sensible

Send me these three things for any run and I can tell you if something
looks wrong:

1. **`training_history.csv`** — train_loss and val_loss should both
   generally trend down over epochs (some noise is normal). If val_loss
   is flat from epoch 1 or explodes to NaN/huge numbers, something's
   wrong — likely worth checking the loss scheme (`inverse_freq`'s 513.5x
   ratio is a plausible culprit for instability; `inverse_sqrt_freq`, the
   current default, is much less likely to).
2. **`run_summary.json`** — sanity-check `n_epochs_run` (did it run the
   expected number, or stop suspiciously early/late?) and `best_val_loss`.
3. **`outputs/results/Type_II/<architecture>/<configuration>/per_image.csv`
   and `summary.json`** — the actual IoU/Boundary-F1/HD95 numbers. A few
   things to sanity check yourself before sending them to me:
   - **Alpha IoU should be reasonably high** (it's ~74% of pixels — if a
     trained model's Alpha IoU is near 0, something is badly broken, not
     just "the model needs more training").
   - **TiN's `n_included`/`n_computed` counts** tell you how many test
     images actually contained TiN pixels at all — a TiN IoU of 0.0 with
     `n_computed=0` in HD95 (meaning TiN was `missed_entirely` every time)
     is a real, expected possibility for an early/undertrained model, not
     necessarily a bug — this is exactly the majority-class baseline
     behavior `verify_evaluator.py` already established as the sanity
     floor.
   - Compare `baseline` vs `patch`/`loss`/`both` for the SAME architecture
     — if TiN IoU doesn't improve at all across configs, that's the actual
     scientific finding to discuss, not necessarily a bug to fix.

## What's left after Type II

- **SAM-2 pre-annotation benchmark** — Person A's independent add-on, not started.
- **Types I and III** — Person B and C's ownership, using the exact same
  `run_experiment.py` with `--true-type I` or `--true-type III` once their
  own architecture integration is confirmed working the same way Type II's is.
- **Final cross-type comparison compilation** — once all 24 runs across all
  three people are done, per `outputs/results/Type_X/<arch>/<config>/summary.json`.



## Experiment Execution

The core ablation evaluates two segmentation architectures:

- DeepLabV3+
- Attention U-Net

under four configurations:

1. baseline: uniform patch sampling + standard multiclass CE
2. patch: TiN-aware patch sampling + standard multiclass CE
3. loss: uniform sampling + type-specific class-weighted CE
4. both: TiN-aware patch sampling + type-specific class-weighted CE

Experiments are run separately for each MicroSteel processing type.

Run identifiers follow:

- `II-DL-B`
- `II-DL-P`
- `II-DL-L`
- `II-DL-PL`
- `II-AUN-B`
- `II-AUN-P`
- `II-AUN-L`
- `II-AUN-PL`

Training automatically uses CUDA when available and falls back to CPU otherwise.
## Experiment Results

Training checkpoints and experiment outputs are maintained separately from
the source repository.

Each experiment archive contains:

```text
<run-id>/
├── training_run/
│   ├── best_checkpoint.pt
│   ├── training_history.csv
│   └── run_summary.json
└── results/
    ├── summary.json
    └── per_image.csv