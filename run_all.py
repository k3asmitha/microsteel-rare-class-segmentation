"""
run_all.py
===========
Single entrypoint that runs the ENTIRE audit -> manifest -> analysis
pipeline in order, exactly as it must run (later stages depend on earlier
ones being correct). Produces:

    outputs/audit_findings.json   -- machine-readable, one entry per stage
    outputs/AUDIT_REPORT.md       -- human-readable report generated FROM
                                      the JSON above (not hand-written)
    outputs/manifest.csv          -- the canonical manifest
    outputs/excluded_labels.csv
    outputs/class_codebook.json
    evidence/*.png                -- visual proof for the geometry audit

Exit code is 0 only if every hard check passes. Run this, then read
AUDIT_REPORT.md top to bottom -- every conclusion in it was just computed
from the raw dataset on this run, not copy-pasted from a prior one.
"""

import json
import os
import sys
import traceback
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.common import load_config
from src.audit import a01_inventory_audit, a02_pairing_audit, a03_geometry_audit
from src.audit import a04_codebook_audit, a05_split_audit
from src.manifest.build_manifest import build_manifest
from src.manifest.validate_manifest import validate as validate_manifest
from src.analysis.tin_speck_analysis import run as run_speck_analysis
from src.patching.patch_index_builder import build_patch_index
from src.patching.validate_patch_index import validate as validate_patch_index
from src.patching.sampler import run as run_sampler
from src.patching.verify_dataset import run as run_verify_dataset


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    cfg = load_config("configs/config.yaml")
    os.makedirs(cfg["paths"]["output_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["evidence_dir"], exist_ok=True)

    all_findings = {"run_timestamp": datetime.utcnow().isoformat() + "Z"}
    hard_failure = False

    stages = [
        ("a01_inventory_audit", lambda: a01_inventory_audit.run(cfg)),
        ("a02_pairing_audit", lambda: a02_pairing_audit.run(cfg)),
        ("a03_geometry_audit", lambda: a03_geometry_audit.run(cfg)),
        ("a04_codebook_audit", lambda: a04_codebook_audit.run(cfg)),
        ("a05_split_audit", lambda: a05_split_audit.run(cfg)),
    ]

    for name, fn in stages:
        section(f"RUNNING: {name}")
        try:
            result = fn()
            all_findings[name] = result
            print(result.get("conclusion", "(no conclusion field)"))
            if result.get("codebook_is_consistent") is False:
                hard_failure = True
        except Exception as e:
            all_findings[name] = {"error": str(e), "traceback": traceback.format_exc()}
            print(f"STAGE ERRORED: {e}")
            hard_failure = True

    if hard_failure:
        print("\nHard failure in audit stage(s) -- stopping before manifest build.")
        _write_outputs(cfg, all_findings)
        return 1

    section("RUNNING: build_manifest")
    try:
        rows, excluded, manifest_summary = build_manifest(cfg)
        all_findings["build_manifest"] = manifest_summary
        print(f"Built manifest with {len(rows)} usable rows, {len(excluded)} excluded.")
    except Exception as e:
        all_findings["build_manifest"] = {"error": str(e), "traceback": traceback.format_exc()}
        print(f"MANIFEST BUILD ERRORED: {e}")
        _write_outputs(cfg, all_findings)
        return 1

    section("RUNNING: validate_manifest")
    val_result = validate_manifest(cfg)
    all_findings["validate_manifest"] = val_result
    print(val_result["conclusion"])
    if not val_result["passed"]:
        hard_failure = True

    if hard_failure:
        _write_outputs(cfg, all_findings)
        return 1

    section("RUNNING: tin_speck_analysis")
    speck_result = run_speck_analysis(cfg)
    all_findings["tin_speck_analysis"] = speck_result
    print(speck_result["conclusion"])
    # Write intermediate results NOW -- validate_patch_index needs this file
    # on disk to cross-check against the fragmentation simulation, and it
    # would otherwise not exist yet on a clean run (this file is normally
    # only finalized at the end of main()).
    _write_outputs(cfg, all_findings)

    section("RUNNING: build_patch_index")
    try:
        patch_rows, patch_summary = build_patch_index(cfg)
        all_findings["build_patch_index"] = patch_summary
        print(patch_summary["conclusion"])
    except Exception as e:
        all_findings["build_patch_index"] = {"error": str(e), "traceback": traceback.format_exc()}
        print(f"PATCH INDEX BUILD ERRORED: {e}")
        _write_outputs(cfg, all_findings)
        return 1

    section("RUNNING: validate_patch_index")
    patch_val_result = validate_patch_index(cfg)
    all_findings["validate_patch_index"] = patch_val_result
    print(patch_val_result["conclusion"])
    if not patch_val_result["passed"]:
        hard_failure = True

    section("RUNNING: sampler")
    sampler_result = run_sampler(cfg)
    all_findings["sampler"] = sampler_result
    print(sampler_result["conclusion"])
    if not sampler_result.get("passed", True):
        hard_failure = True

    section("RUNNING: verify_dataset")
    verify_result = run_verify_dataset(cfg)
    all_findings["verify_dataset"] = verify_result
    print(verify_result["conclusion"])
    if not verify_result["passed"]:
        hard_failure = True

    _write_outputs(cfg, all_findings)
    if hard_failure:
        return 1
    section("DONE")
    print("All stages passed. See outputs/AUDIT_REPORT.md for the full report.")
    return 0


def _write_outputs(cfg, all_findings):
    out_dir = cfg["paths"]["output_dir"]
    json_path = os.path.join(out_dir, "audit_findings.json")
    with open(json_path, "w",encoding="utf-8") as f:
        json.dump(all_findings, f, indent=2, default=str)

    report = _render_report(all_findings)
    report_path = os.path.join(out_dir, "AUDIT_REPORT.md")
    with open(report_path, "w",encoding="utf-8") as f:
        f.write(report)

    print(f"\nWrote {json_path}")
    print(f"Wrote {report_path}")


def _render_report(f: dict) -> str:
    lines = []
    lines.append("# MicroSteel Pipeline — Audit Report")
    lines.append(f"\nGenerated: {f.get('run_timestamp', 'unknown')}")
    lines.append("\nThis report is generated directly from `outputs/audit_findings.json`, "
                  "which is itself the return value of each stage in `run_all.py` run "
                  "against the raw dataset. Nothing below is hand-typed after the fact.")

    def stage_section(key, title):
        if key not in f:
            return
        r = f[key]
        lines.append(f"\n## {title}")
        if "error" in r:
            lines.append(f"\n**STAGE ERRORED:** `{r['error']}`")
            return
        lines.append(f"\n**Question:** {r.get('question', '(n/a)')}")
        lines.append(f"\n**Conclusion:** {r.get('conclusion', '(n/a)')}")

    stage_section("a01_inventory_audit", "1. Inventory Audit — can folder names be trusted?")
    stage_section("a02_pairing_audit", "2. Pairing Audit — does every label have an image?")
    stage_section("a03_geometry_audit", "3. Geometry Audit — crop offset, resolution, color mode")
    stage_section("a04_codebook_audit", "4. Codebook Audit — decoding integer mask values")
    stage_section("a05_split_audit", "5. Split Audit — split file integrity & TiN stratification")

    if "build_manifest" in f:
        r = f["build_manifest"]
        lines.append("\n## 6. Manifest Build")
        if "error" in r:
            lines.append(f"\n**STAGE ERRORED:** `{r['error']}`")
        else:
            lines.append(f"\nUsable pairs: **{r.get('usable_pairs')}**, "
                          f"excluded: **{r.get('excluded_count')}**")
            for e in r.get("excluded", []):
                lines.append(f"  - `{e['stem']}` ({e['true_type']}): {e['reason']}")

    if "validate_manifest" in f:
        r = f["validate_manifest"]
        lines.append("\n## 7. Manifest Validation")
        lines.append(f"\n**Result:** {r.get('conclusion')}")
        if r.get("warnings"):
            lines.append("\nWarnings:")
            for w in r["warnings"]:
                lines.append(f"  - {w}")

    if "tin_speck_analysis" in f:
        r = f["tin_speck_analysis"]
        lines.append("\n## 8. Patch Size Decision (TiN Speck Analysis)")
        lines.append(f"\n**Conclusion:** {r.get('conclusion')}")
        lines.append("\n**Fragmentation simulation results:**\n")
        lines.append("| Patch | Stride | Overlap % | Total patches | % specks intact |")
        lines.append("|---|---|---|---|---|")
        for s in r.get("fragmentation_simulation", []):
            lines.append(f"| {s['patch_size']} | {s['stride']} | {s['overlap_pct']} | "
                          f"{s['total_patches_full_dataset']} | {s['pct_specks_fully_intact']} |")
        rec = r.get("recommendation")
        if rec:
            lines.append(f"\n**Final recommendation:** patch_size={rec['patch_size']}, "
                          f"stride={rec['stride']} ({rec['overlap_pct']}% overlap)")

    if "build_patch_index" in f:
        r = f["build_patch_index"]
        lines.append("\n## 9. Patch Index Build")
        if "error" in r:
            lines.append(f"\n**STAGE ERRORED:** `{r['error']}`")
        else:
            lines.append(f"\n**Conclusion:** {r.get('conclusion')}")

    if "validate_patch_index" in f:
        r = f["validate_patch_index"]
        lines.append("\n## 10. Patch Index Validation (incl. cross-check vs. speck analysis)")
        lines.append(f"\n**Result:** {r.get('conclusion')}")
        cc = r.get("cross_check_vs_fragmentation_simulation")
        if cc:
            lines.append(f"\nCross-check: fragmentation simulation predicted "
                          f"**{cc['expected_from_simulation']}** total patches; "
                          f"actual patch index built **{cc['actual_from_patch_index']}** — "
                          f"{'MATCH' if cc['match'] else 'MISMATCH (see failures)'}.")

    if "sampler" in f:
        r = f["sampler"]
        lines.append("\n## 11. Weighted Sampler (TiN oversampling)")
        lines.append(f"\n**Conclusion:** {r.get('conclusion')}")
        if r.get("warning"):
            lines.append(f"\n> ⚠️ {r['warning']}")

    if "verify_dataset" in f:
        r = f["verify_dataset"]
        lines.append("\n## 12. Dataset Extraction & Augmentation Verification")
        lines.append(f"\n**Result:** {r.get('conclusion')}")

    lines.append("\n---\n*End of generated report.*")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
