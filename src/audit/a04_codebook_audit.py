"""
a04_codebook_audit.py
=======================
QUESTION: What phase does each integer mask value (0,1,2,3,4...) represent,
and is that mapping consistent across every image, or does it drift?

METHOD (fully data-driven, no hardcoded "label 2 = TiN" assumption):
  1. For every usable image, cross-reference each integer mask value against
     its RGB color in the coloured_labels/ visualization. If one integer
     ever maps to more than one RGB color (or vice versa), the codebook is
     UNSAFE to treat as global and this audit fails loudly.
  2. Compute, per integer code, which processing TYPES it appears in at all
     (its "presence signature") and its global pixel share.
  3. Match each integer code to an expected phase name from config by:
       a) presence signature must match the phase's expected type membership
          (e.g. FeTiB is expected only in Type I -- an integer appearing in
          II or III cannot be FeTiB no matter its pixel share)
       b) among codes whose signature matches multiple candidate phases
          (Alpha/TiB2/TiN all expected in all 3 types), disambiguate by
          closest global pixel-share to the expected percentage in config.
  4. Report the final decoded codebook plus the evidence for each mapping,
     so the decision is auditable, not asserted.
"""

import os
from collections import defaultdict

import numpy as np
from PIL import Image

from src.common import true_type_from_stem, build_image_map, list_label_stems, label_path, coloured_label_path


def get_usable_stems(data_root):
    img_map, _ = build_image_map(data_root)
    label_stems = set(list_label_stems(data_root))
    return sorted(label_stems & set(img_map.keys()))


def run(config: dict) -> dict:
    data_root = config["paths"]["data_root"]
    expected_membership = config["expected_type_class_membership"]
    expected_pct = config["expected_pixel_pct"]

    usable_stems = get_usable_stems(data_root)

    int_to_rgb = defaultdict(set)
    rgb_to_int = defaultdict(set)
    global_px = defaultdict(int)
    per_type_px = defaultdict(lambda: defaultdict(int))
    presence = defaultdict(set)  # int_code -> set of types it appears in

    for stem in usable_stems:
        t = true_type_from_stem(stem)
        lbl = np.array(Image.open(label_path(data_root, stem)))
        clbl = np.array(Image.open(coloured_label_path(data_root, stem)).convert("RGB"))

        vals, counts = np.unique(lbl, return_counts=True)
        for v, c in zip(vals, counts):
            v = int(v)
            global_px[v] += int(c)
            per_type_px[t][v] += int(c)
            presence[v].add(t)
            mask = (lbl == v)
            colors = set(map(tuple, np.unique(clbl[mask], axis=0)))
            int_to_rgb[v].update(colors)
            for col in colors:
                rgb_to_int[col].add(v)

    # --- Step 1: consistency hard-check ---
    inconsistent_int = {i: list(c) for i, c in int_to_rgb.items() if len(c) > 1}
    inconsistent_rgb = {str(c): list(i) for c, i in rgb_to_int.items() if len(i) > 1}
    codebook_is_consistent = not inconsistent_int and not inconsistent_rgb

    if not codebook_is_consistent:
        return {
            "check": "codebook_audit",
            "question": "What phase does each integer code represent?",
            "codebook_is_consistent": False,
            "inconsistent_int_to_rgb": inconsistent_int,
            "inconsistent_rgb_to_int": inconsistent_rgb,
            "conclusion": (
                "HARD FAILURE: integer<->RGB mapping is NOT globally consistent. "
                "Do not proceed to build a global class codebook -- treat "
                "class semantics as potentially per-image and investigate "
                "the inconsistent codes above before writing any training code."
            ),
        }

    # --- Step 2 & 3: match codes to phase names ---
    total_px = sum(global_px.values())
    global_pct = {i: 100.0 * px / total_px for i, px in global_px.items()}

    # invert expected_membership: phase_name -> set of types it's expected in
    phase_to_types = defaultdict(set)
    for type_name, phases in expected_membership.items():
        for phase in phases:
            phase_to_types[phase].add(type_name)

    decoded = {}
    decode_evidence = {}
    used_phases = set()

    # sort codes by how "unique" their signature is (fewest candidate phases first)
    # to resolve unambiguous matches before ambiguous ones
    code_signature = {i: presence[i] for i in presence}

    def candidate_phases_for(sig):
        return [p for p, types in phase_to_types.items() if types == sig and p not in used_phases]

    codes_sorted = sorted(code_signature, key=lambda i: len(candidate_phases_for(code_signature[i])))

    for code in codes_sorted:
        sig = code_signature[code]
        candidates = candidate_phases_for(sig)
        if len(candidates) == 0:
            decoded[code] = "UNMATCHED"
            decode_evidence[code] = {
                "presence_signature": sorted(sig),
                "global_pct": round(global_pct[code], 4),
                "reason": "No expected phase has this exact type-presence signature",
            }
            continue
        if len(candidates) == 1:
            decoded[code] = candidates[0]
            used_phases.add(candidates[0])
            decode_evidence[code] = {
                "presence_signature": sorted(sig),
                "global_pct": round(global_pct[code], 4),
                "matched_by": "unique presence signature",
            }
            continue
        # ambiguous: disambiguate by closest expected pixel percentage
        best = min(candidates, key=lambda p: abs(expected_pct.get(p, 0) - global_pct[code]))
        decoded[code] = best
        used_phases.add(best)
        decode_evidence[code] = {
            "presence_signature": sorted(sig),
            "global_pct": round(global_pct[code], 4),
            "matched_by": f"closest pixel-pct match among ambiguous candidates {candidates} "
                          f"(expected {best}={expected_pct.get(best)}%, observed {global_pct[code]:.4f}%)",
        }

    per_type_pct_report = {}
    for t, px_dict in per_type_px.items():
        tot = sum(px_dict.values())
        per_type_pct_report[t] = {
            decoded.get(code, f"code_{code}"): round(100.0 * px / tot, 4)
            for code, px in px_dict.items()
        }

    findings = {
        "check": "codebook_audit",
        "question": "What phase does each integer code represent, and is it consistent?",
        "codebook_is_consistent": True,
        "decoded_codebook": decoded,
        "decode_evidence": decode_evidence,
        "global_pixel_pct_by_code": {k: round(v, 4) for k, v in global_pct.items()},
        "per_type_pixel_pct_by_phase": per_type_pct_report,
        "rare_class_decoded_id": next((c for c, p in decoded.items()
                                        if p == config["rare_class"]["name"]), None),
        "conclusion": (
            f"Integer<->RGB mapping is 100% consistent across all "
            f"{len(usable_stems)} usable images (zero exceptions). "
            f"Decoded mapping: {decoded}. "
            f"Rare class '{config['rare_class']['name']}' decoded as integer "
            f"code {next((c for c, p in decoded.items() if p == config['rare_class']['name']), 'NOT FOUND')}."
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
