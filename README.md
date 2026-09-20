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
│   ├── common.py            # shared utilities (type parsing, image mapping) — single implementation
│   ├── audit/
│   │   ├── a01_inventory_audit.py   # can folder names be trusted?
│   │   ├── a02_pairing_audit.py     # does every label have a matching image?
│   │   ├── a03_geometry_audit.py    # image/label crop offset, resolution, color mode
│   │   ├── a04_codebook_audit.py    # decode integer mask codes -> phase names
│   │   └── a05_split_audit.py       # split file integrity, TiN stratification
│   ├── manifest/
│   │   ├── build_manifest.py        # builds outputs/manifest.csv from audit results
│   │   └── validate_manifest.py     # checks every row against invariants
│   └── analysis/
│       └── tin_speck_analysis.py    # measures TiN speck sizes, decides patch_size/stride
├── run_all.py                # THE entrypoint — runs everything in order
├── requirements.txt
├── evidence/                 # visual proof (crop alignment overlay, removed metadata strip)
└── outputs/                  # generated — manifest.csv, AUDIT_REPORT.md, etc.
```

## How to run it

```bash
cd microsteel_project
pip install -r requirements.txt

# Edit configs/config.yaml first: set paths.data_root to wherever you
# extracted microsteel_dataset.zip (must contain images_class*/, labels/, splits/)

python3 run_all.py
```

That's the only command you need. It runs, in order:
1. `a01` → `a02` → `a03` → `a04` → `a05` (raw-data audits)
2. `build_manifest.py` (only runs if all audits pass — a04 in particular
   will hard-fail the whole run if the integer↔RGB codebook turns out to be
   inconsistent, because building a manifest on top of that would be
   silently wrong)
3. `validate_manifest.py` (checks every row of the manifest just built)
4. `tin_speck_analysis.py` (measures speck geometry, decides patch size)

Exit code is `0` only if every hard check passes. Non-zero means something
is broken and the manifest should not be trusted or used for training.

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

The manifest and patch-size decision here are the foundation. The next
deliverable on the critical path is the actual patch-tiling + TiN-tagging +
weighted-sampler code that reads `outputs/manifest.csv` and produces
training-ready patches using `patch_size=256, stride=128` from
`configs/config.yaml`. That hasn't been built yet — this repo only
establishes and verifies the *inputs* to it.
