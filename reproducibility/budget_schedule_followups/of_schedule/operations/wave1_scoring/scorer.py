"""Explicitly authorized single-rotation interim analysis of schedule wave one.

This does not replace the sealed three-rotation, 72-fit primary analysis.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

SEEDS = (20260923, 20260924, 20260925)
STRATEGIES = ("first", "all", "last")
ROTATION = 20261001
METRICS = ("ca_lddt", "residue_ca_lddt", "tm_score_fixed_full_length")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def name(strategy, kind, rotation, seed):
    direction = "native" if rotation is None else f"r{rotation}"
    return f"{strategy}_{kind}_{direction}_s{seed}"


def analyze(rows, ids):
    if len(ids) != 96 or len(set(ids)) != 96:
        raise ValueError("the locked Confirm96 target set is required")
    systems = {name(st, kind, rot, s) for st in STRATEGIES
               for kind in ("factor", "generic_plus")
               for rot in (None, ROTATION) for s in SEEDS} | {"query"}
    expected = {(system, tid) for system in systems for tid in ids}
    keys = [(r["system"], r["target_id"]) for r in rows]
    if len(set(keys)) != len(keys) or set(keys) != expected:
        raise ValueError("expected all 37 systems x 96 targets exactly once")
    lookup = dict(zip(keys, rows, strict=True))
    draw = np.random.default_rng(20260926).integers(96, size=(20000, 96))

    def summary(a):
        a = np.broadcast_to(a, (3, 96))
        if not np.isfinite(a).all():
            raise ValueError("nonfinite metric")
        v = a.mean(0)
        return dict(mean=float(v.mean()),
                    ci95=np.quantile(v[draw].mean(1), [.025, .975]).tolist(),
                    per_seed=a.mean(1).tolist(), per_target=v.tolist(),
                    positive_targets=int((v > 0).sum()),
                    conditional_on_fitted_models=True)

    result = {}
    for metric in METRICS:
        def vector(system):
            return np.array([lookup[system, tid][metric] for tid in ids])
        query = vector("query")
        conditions, effects = {}, {}
        for st in STRATEGIES:
            cells = {label: np.array([vector(name(st, kind, rot, s)) for s in SEEDS])
                     for label, kind, rot in (
                         ("factor_native", "factor", None),
                         ("factor_rotated", "factor", ROTATION),
                         ("gplus_native", "generic_plus", None),
                         ("gplus_rotated", "generic_plus", ROTATION))}
            df = cells["factor_native"] - cells["factor_rotated"]
            dg = cells["gplus_native"] - cells["gplus_rotated"]
            effects[st] = (df, dg, df - dg)
            conditions[st] = dict(
                absolute={k: summary(v) for k, v in cells.items()},
                gain_over_query={k: summary(v - query) for k, v in cells.items()},
                factor_rotation=summary(df), gplus_rotation=summary(dg),
                interaction=summary(df - dg),
                native_factor_minus_gplus=summary(cells["factor_native"] - cells["gplus_native"]))
        differences = {}
        for label, a, b in (("D_write_R1", "all", "first"),
                            ("D_timing_R1", "first", "last"),
                            ("D_last_R1", "all", "last")):
            f, g, p = [effects[a][k] - effects[b][k] for k in range(3)]
            assert np.max(np.abs(p - (f - g))) < 1e-12
            differences[label] = dict(interaction=summary(p),
                                     factor_contribution=summary(f),
                                     gplus_contribution=summary(g))
        result[metric] = dict(query=summary(query), conditions=conditions,
                             differences=differences)
    return result


def seal(root, out):
    """Bind reporting amendment and all identities before reading quality scores."""
    from engramfold.evaluation import structure, independent, post_validation
    lock = read(root / "execution_lock.json")
    wave = read(root / "wave1_complete.json")
    assert wave["complete"] and wave["failures"] == 0
    assert wave["execution_lock_sha256"] == sha(root / "execution_lock.json")
    runs = [r for r in lock["formal"] if r["wave"] == 1]
    assert len(runs) == 36
    assert {r["rotation"] for r in runs} == {None, ROTATION}
    targets = read(root / "reference_manifest.json")
    inputs = read(root / "inference_manifest.json")
    assert [(t["target_id"], t["sequence"]) for t in inputs] == [
        (t["target_id"], t["sequence"]) for t in targets]
    paths = [root / p for p in ("execution_lock.json", "wave1_complete.json",
             "operations/wave1_completion_audit.json", "reference_manifest.json",
             "inference_manifest.json", "data/TMscore", "data/alphafold_lddt.py")]
    paths += [Path(__file__), Path(structure.__file__), Path(independent.__file__),
              Path(post_validation.__file__)]
    for run in runs + [dict(name="query")]:
        p = root / "formal" / run["name"] / "complete.json"
        assert sha(p) == wave["receipts"][str(p.relative_to(root))]
        r = read(p)
        assert r["complete"] and r["execution_lock_sha256"] == sha(root / "execution_lock.json")
        assert [x["target_id"] for x in r["records"]] == [t["target_id"] for t in targets]
        paths.append(p)
    for t in targets:
        r = lock["references"][t["target_id"]]
        assert sha(r["path"]) == r["sha256"] == t["raw_mmcif_sha256"]
        paths.append(Path(r["path"]))
    obj = dict(utc=datetime.now(timezone.utc).isoformat(),
               reporting_authorization="用户先要求先报告，继而明确：没事，现在使用36足够了，汇报这个以及Protenix／ESMC完整四格的预算扩展的结果",
               amendment="Report completed single-R1 36-fit scope now; R2/R3 are not required for this report and will not be launched. Original 72-fit scientific lock preserved; no claim to its full primary result.",
               status="SINGLE_ROTATION_INTERIM_NOT_FULL72", rotation=ROTATION,
               seeds=SEEDS, targets=[t["target_id"] for t in targets], runs=runs,
               n_predictions=3552, full_matrix_primary_unavailable=True,
               bootstrap=dict(unit="target", repeats=20000, seed=20260926,
                              paired=True, stratified=False),
               primary_analogue="D_write_R1 = Psi_all_R1 - Psi_first_R1",
               secondary=["D_timing_R1", "D_last_R1", "row effects", "absolute and query gains", "residue-lDDT", "TM-score"],
               familywise_adjustment=False, panel="observed Confirm96-B",
               metric_contract="Full CIF CA with sealed reference mask; FP64 distances; full-sequence-length TM-score; failure zero retained.",
               files={str(p.resolve()): sha(p) for p in paths})
    path = out / "reporting_amendment.json"
    assert not path.exists(), "refuse to overwrite sealed reporting amendment"
    write(path, obj)
    print(json.dumps(dict(sealed=str(path), sha256=sha(path))))


def score(root, out, workers):
    from engramfold.evaluation.structure import read_atom_site_positions
    from engramfold.evaluation.independent import fixed_mask_metrics
    from engramfold.evaluation.post_validation import ca_pdb, parse_tm_score, load_reference_lddt
    contract = read(out / "reporting_amendment.json")
    for p, expected in contract["files"].items():
        assert sha(p) == expected, p
    lock = read(root / "execution_lock.json")
    targets = read(root / "reference_manifest.json")
    fn = load_reference_lddt(root / "data/alphafold_lddt.py")
    refs = {}
    for t in targets:
        tid = t["target_id"]
        _, atoms = read_atom_site_positions(Path(lock["references"][tid]["path"]),
                                            label_asym_id=t["source_label_asym_id"])
        ref = {i: v for (i, atom), v in atoms.items() if atom == "CA"}
        assert sorted(ref) == t["reference_ca_indices"]
        refs[tid] = ref
        folder = out / "coordinates" / tid
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "reference.pdb").write_text(ca_pdb(ref))

    def metric(job):
        system, t, rec = job
        tid = t["target_id"]
        base = dict(system=system, target_id=tid, status=rec["status"])
        if rec["status"] != "ok":
            return dict(base, **{m: 0.0 for m in METRICS}, error=rec.get("error"))
        path = root / "formal" / system / "predictions" / (tid + ".cif")
        assert sha(path) == rec["cif_sha256"]
        _, atoms = read_atom_site_positions(path, label_asym_id="A")
        pred = {i: v for (i, atom), v in atoms.items() if atom == "CA"}
        assert sorted(pred) == list(range(1, len(t["sequence"]) + 1))
        npz = path.with_suffix(".npz")
        assert sha(npz) == rec["npz_sha256"]
        with np.load(npz) as data:
            assert str(data["sequence"]) == t["sequence"]
            raw = data["coordinates"][:, 1].astype(np.float64)
        cif_npz_maxabs = float(np.max(np.abs(np.stack(list(pred.values())) - raw)))
        assert cif_npz_maxabs == 0.0, "17-digit CIF must preserve FP32 CA exactly"
        ref = refs[tid]
        ids = sorted(ref)
        rx, px = np.stack([ref[i] for i in ids]), np.stack([pred[i] for i in ids])
        mask = np.ones((1, len(ids), 1))
        pair = fixed_mask_metrics(ref, pred)["ca_lddt"]
        af = float(fn(px[None], rx[None], mask)[0])
        assert abs(pair - af) < 1e-6
        residue = float(fn(px[None], rx[None], mask, per_residue=True).mean())
        folder = out / "coordinates" / tid
        pp = folder / (system + ".pdb")
        pp.write_text(ca_pdb({i: pred[i] for i in ids}))
        tm = subprocess.run([str(root / "data/TMscore"), str(pp),
                             str(folder / "reference.pdb"), "-l", str(len(t["sequence"]))],
                            capture_output=True, text=True, check=True, timeout=90)
        (folder / (system + ".tm.txt")).write_text(tm.stdout + tm.stderr)
        return dict(base, ca_lddt=pair, residue_ca_lddt=residue,
                    tm_score_fixed_full_length=parse_tm_score(tm.stdout),
                    cif_sha256=rec["cif_sha256"], npz_sha256=rec["npz_sha256"],
                    cif_npz_ca_maxabs=cif_npz_maxabs, independent_lddt_abs_error=abs(pair - af))

    jobs = []
    for run in contract["runs"] + [dict(name="query")]:
        receipt = read(root / "formal" / run["name"] / "complete.json")
        jobs.extend((run["name"], t, r) for t, r in zip(targets, receipt["records"], strict=True))
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(metric, jobs):
            rows.append(row)
            if len(rows) % 96 == 0:
                write(out / "progress.json", dict(scored=len(rows), expected=len(jobs)))
                print(f"scored {len(rows)}/{len(jobs)}", flush=True)
    write(out / "metric_records.json", rows)
    result = dict(utc=datetime.now(timezone.utc).isoformat(),
                  status=contract["status"], rotation=ROTATION, seeds=SEEDS,
                  target_ids=contract["targets"], n_predictions=len(rows),
                  failures=sum(r["status"] != "ok" for r in rows),
                  execution_lock_sha256=sha(root / "execution_lock.json"),
                  reporting_amendment_sha256=sha(out / "reporting_amendment.json"),
                  statistics=analyze(rows, contract["targets"]))
    write(out / "analysis.json", result)
    write(out / "complete.json", dict(complete=True, utc=result["utc"],
          n_predictions=len(rows), failures=result["failures"],
          analysis_sha256=sha(out / "analysis.json"),
          metric_records_sha256=sha(out / "metric_records.json"),
          reporting_amendment_sha256=result["reporting_amendment_sha256"]))
    print("COMPLETE", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["seal", "score"])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    if a.action == "seal":
        seal(a.root, a.out)
    else:
        score(a.root, a.out, a.workers)


if __name__ == "__main__":
    main()
