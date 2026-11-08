#!/usr/bin/env python3
"""Generate the review appendix tables from the locked score summaries."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence/rebuttal_followups"
OUTPUT = ROOT / "generated/rebuttal_numbers.tex"
SOURCES = {
    "protenix_msa32_full": ("minidefault.msa.evaluation.v1", 192, 8064),
    "protenix_msa256": ("minidefault.msa.evaluation.v1", 192, 4032),
    "protenix_msa512": ("minidefault.msa.evaluation.v1", 192, 4032),
    "openfold_full_msa": ("openfold.followup.v1", 96, 2016),
    "protenix_cross_seed": ("protenix.cross_seed.analysis.v1", 192, 13824),
    "openfold_cross_seed": ("openfold.followup.v1", 96, 2400),
    "protenix_frozen_map": ("protenix.same_parent_branch.v1", 192, 20736),
}
PAIR = "ca_lddt"
TARGET = "target_only_conditional_ci95"
CROSSED = "crossed_target_training_seed_ci95_exploratory"
CELLS = ("F_native", "F_rotated", "G_native", "G_rotated")
MSA = (
    ("Protenix, cap 32", "protenix_msa32_full", "shallow32_"),
    ("Protenix, cap 256", "protenix_msa256", "msa256_"),
    ("Protenix, cap 512", "protenix_msa512", "msa512_"),
    ("Protenix, full", "protenix_msa32_full", "full_"),
    ("OpenFold, full", "openfold_full_msa", ""),
)


def load_sources():
    lock = json.loads((EVIDENCE / "sources.lock.json").read_bytes())
    if set(lock) != set(SOURCES):
        raise ValueError("Review evidence lock does not match the expected studies")
    data = {}
    for key, (schema, targets, predictions) in SOURCES.items():
        raw = (EVIDENCE / f"{key}.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != lock[key]["snapshot_sha256"]:
            raise ValueError(f"Review evidence changed: {key}")
        source = json.loads(raw)
        identity = (source["schema"], source.get("targets", source.get("n_targets")), source["predictions"])
        if identity != (schema, targets, predictions) or source.get("baseline_only", False):
            raise ValueError(f"Unexpected evaluation coverage: {key}")
        data[key] = source
    return data


def require_close(value, expected, name):
    if not math.isclose(value, expected, rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"{name}: {value} does not match {expected}")


def point(value, signed=False):
    return f"{value:+.5f}" if signed else f"{value:.5f}"


def interval(result, method=TARGET):
    lo, hi = result[method]
    if not all(map(math.isfinite, (lo, hi))) or lo > hi:
        raise ValueError(f"Invalid interval: {result}")
    return rf"[{point(lo, True)},\,{point(hi, True)}]"


def stack(result, *, bold=False):
    value = point(result["mean"], True)
    if bold:
        value = rf"\mathbf{{{value}}}"
    return rf"\shortstack[r]{{\({value}\)\\\({interval(result)}\)}}"


def row(*cells):
    return " & ".join(cells) + r" \\[2pt]"


def msa_rows(data, metric):
    rows = []
    for name, key, prefix in MSA:
        stats = data[key]["metrics"][metric]
        baseline, factor, generic = (stats[prefix + arm]["mean"] for arm in ("baseline", "factor", "generic_plus"))
        gain = stats[prefix + "factor_gain"]
        head = stats[prefix + "factor_minus_generic_plus"]
        require_close(gain["mean"], factor - baseline, f"{name}: Factor gain")
        require_close(head["mean"], factor - generic, f"{name}: Factor minus G+")
        rows.append(row(
            name, *(point(x) for x in (baseline, factor, generic)),
            stack(gain, bold=metric == PAIR and prefix != "shallow32_"), stack(head),
        ))
    return rows


def cross_seed_rows(data):
    rows = []
    for label, key in (("Protenix/ESMC", "protenix_cross_seed"), ("OpenFold/ESM2", "openfold_cross_seed")):
        stats = data[key]["metrics"][PAIR]
        psi = stats["Psi"]
        cohort = [stats[k]["mean"] for k in CELLS]
        require_close(psi["mean"], cohort[0] - cohort[1] - cohort[2] + cohort[3], f"{key}: interaction")
        per_seed = list(zip(*(stats[k]["per_training_seed"] for k in (*CELLS, "Psi")), strict=True))
        if len(per_seed) != 3:
            raise ValueError(f"Expected three new training seeds: {key}")
        require_close(psi["mean"], mean(s[-1] for s in per_seed), f"{key}: seed mean")
        rows.append(row(label, "Mean", *(point(x) for x in cohort), stack(psi)))
        for i, (fn, fr, gn, gr, interaction) in enumerate(per_seed, 1):
            require_close(interaction, fn - fr - gn + gr, f"{key}: seed {i} interaction")
            rows.append(row("", str(i), *(point(x) for x in (fn, fr, gn, gr)), rf"\({point(interaction, True)}\)"))
        rows.append(r"\midrule")
    return rows[:-1]


def frozen_map_rows(data):
    stats = data["protenix_frozen_map"]["metrics"][PAIR]
    rotated, native, diff = (stats[k] for k in ("C_gain_rotated", "C_gain_native", "T_frozen"))
    for direction, gain in (("rotated", rotated), ("native", native)):
        require_close(gain["mean"], stats[f"C_only_{direction}"]["mean"] - stats[f"baseline_{direction}"]["mean"],
                      f"Map-only gain, {direction}")
    require_close(diff["mean"], rotated["mean"] - native["mean"], "Rotated minus native map gain")
    return [
        row(label, rf"\({point(result['mean'], True)}\)", rf"\({interval(result)}\)", rf"\({interval(result, CROSSED)}\)")
        for label, result in (
            ("Rotated: map-only $-$ parent", rotated),
            ("Native: map-only $-$ parent", native),
            ("Rotated gain $-$ Native gain", diff),
        )
    ]


def branch_rows(data):
    stats = data["protenix_frozen_map"]["metrics"][PAIR]
    rows = []
    for label, arm in (("Parent", "baseline"), ("Map only", "C_only"), ("Head only", "head_only"), ("Joint", "joint")):
        rows.append(row(label, *(point(stats[f"{arm}_{d}"]["mean"]) for d in ("native", "rotated"))))
    rows.append(r"\midrule")
    for label, contrast in (
        ("Head only $-$ parent", "head_gain"),
        ("Joint $-$ parent", "joint_gain"),
        ("Joint $-$ head only", "joint_minus_head"),
        ("Map only $-$ head only", "C_minus_head"),
    ):
        rows.append(row(label, *(stack(stats[f"{contrast}_{d}"]) for d in ("native", "rotated"))))
    return rows


def build(data):
    msas = data["protenix_msa32_full"]["metrics"][PAIR]
    full_minus_cap = msas["factor_full_minus_shallow_gain"]
    require_close(full_minus_cap["mean"], msas["full_factor_gain"]["mean"] - msas["shallow32_factor_gain"]["mean"],
                  "Full-minus-32 Factor gain")
    lo, hi = full_minus_cap[TARGET]
    depth = {"mean": -full_minus_cap["mean"], TARGET: (-hi, -lo)}
    inference = data["protenix_cross_seed"]["metrics"][PAIR]["Psi"]["per_inference_seed"]
    require_close(mean(inference), data["protenix_cross_seed"]["metrics"][PAIR]["Psi"]["mean"], "Protenix inference mean")
    if len(inference) != 3:
        raise ValueError("Expected three paired Protenix inference seeds")
    lines = ["% Generated by scripts/build_rebuttal_assets.py; do not edit."]
    commands = {
        "rebuttalMsaPairRows": msa_rows(data, PAIR),
        "rebuttalMsaResidueRows": msa_rows(data, "residue_ca_lddt"),
        "rebuttalMsaTmRows": msa_rows(data, "tm_score_fixed_full_length"),
        "rebuttalCrossSeedRows": cross_seed_rows(data),
        "rebuttalFrozenMapRows": frozen_map_rows(data),
        "rebuttalBranchRows": branch_rows(data),
    }
    for name, rows in commands.items():
        lines += [rf"\newcommand{{\{name}}}{{%", *rows, "}"]
    scalars = {
        "rebuttalMsaDepthGain": rf"{point(depth['mean'], True)}\;{interval(depth)}",
        "rebuttalProtenixInferenceMeans": r",\; ".join(point(x, True) for x in inference),
        "rebuttalProtenixSeedInterval": interval(data["protenix_cross_seed"]["metrics"][PAIR]["Psi"], CROSSED),
        "rebuttalOpenFoldSeedInterval": interval(data["openfold_cross_seed"]["metrics"][PAIR]["Psi"], CROSSED),
    }
    lines.extend(rf"\newcommand{{\{name}}}{{{value}}}" for name, value in scalars.items())
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify that the committed TeX is up to date")
    args = parser.parse_args()
    output = build(load_sources())
    if args.check:
        if OUTPUT.read_text() != output:
            raise SystemExit(f"Stale TeX: {OUTPUT.relative_to(ROOT)}")
    else:
        OUTPUT.write_text(output)
    print(f"Verified {len(SOURCES)} score summaries and {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
