"""Sealed Atlas scoring: pooled Dev8 selection and paired Native-only contrasts."""
from __future__ import annotations

import argparse
import subprocess
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from engramfold.experiments.posttraining_calibration.util import contract, read, sha, stamp, write

SEEDS = (20260923, 20260924, 20260925)
BOOTSTRAP_SEED = 20260925
BOOTSTRAP_DRAWS = 20_000
BRANCHES = ("C_only", "head_only")


def evaluation_runs(config, stage):
    if stage != "formal":
        return config["matrices"]["atlas"][stage]
    rows = config["evaluation_runs"]
    if isinstance(rows, dict):
        rows = rows.get("atlas", rows)
        if isinstance(rows, dict):
            rows = rows["formal"]
    expected = {"atlas_query"} | {
        f"atlas_I_s{seed}_{branch}"
        for seed in SEEDS for branch in (*BRANCHES, "parent")
    }
    assert len(rows) == 10 and {r["name"] for r in rows} == expected
    return rows


def select_multiplier(rows, runs):
    """Use all eight targets equally in each of both first-seed branches."""
    assert len(runs) == 4
    assert {(r["branch"], float(r["multiplier"])) for r in runs} == {
        (branch, q) for branch in BRANCHES for q in (0.3, 1.0)
    }
    assert {r["seed"] for r in runs} == {SEEDS[0]}
    by_run = {r["name"]: r for r in runs}
    assert len(by_run) == 4 and len(rows) == 32
    ids = None
    for name in by_run:
        records = [x for x in rows if x["system"] == name]
        this_ids = [x["target_id"] for x in records]
        assert len(records) == 8 and len(set(this_ids)) == 8
        assert all(x["status"] == "ok" for x in records), "Do not calibrate on failed predictions"
        ids = this_ids if ids is None else ids
        assert ids == this_ids
    means = {
        str(q): float(np.mean([
            x["ca_lddt"] for x in rows
            if float(by_run[x["system"]]["multiplier"]) == q
        ])) for q in (0.3, 1.0)
    }
    assert np.isfinite(list(means.values())).all()
    return (0.3 if means["0.3"] >= means["1.0"] - 1e-6 else 1.0), means


def paired_summary(values, draws):
    """Average fitted seeds before resampling whole targets, never seed rows."""
    values = np.asarray(values, dtype=np.float64)
    assert values.ndim == 2 and values.shape[0] == 3 and np.isfinite(values).all()
    per_target = values.mean(axis=0)
    return {
        "mean": float(per_target.mean()),
        "ci95": np.quantile(per_target[draws].mean(axis=1), [0.025, 0.975]).tolist(),
        "per_seed_mean": values.mean(axis=1).tolist(),
        "per_target": per_target.tolist(),
        "positive_targets": int((per_target > 0).sum()),
        "conditional_on_fitted_models": True,
    }


def analyze(rows, target_ids, metrics):
    assert len(target_ids) == 96 and len(set(target_ids)) == 96
    vals = {(x["system"], x["target_id"]): x for x in rows}
    assert len(vals) == len(rows) == 960
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, len(target_ids), size=(BOOTSTRAP_DRAWS, len(target_ids))
    )
    result = {
        "scope": "Observed Confirm96-B; target intervals conditional on three fitted parent/branch models",
        "primary": "ca_lddt C_only_minus_head_only",
        "secondary": ["C_only_minus_parent", "head_only_minus_parent", "C_only_minus_query", "head_only_minus_query", "parent_minus_query"],
        "target_ids": target_ids,
        "seeds": list(SEEDS),
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED,
                      "unit": "target after averaging the three paired fitted seeds", "stratified": False},
        "inference_note": "One prespecified primary contrast; all other contrasts and auxiliary metrics have descriptive, unadjusted intervals",
        "metrics": {},
    }
    for metric in metrics:
        def branch(name):
            return np.array([[vals[f"atlas_I_s{s}_{name}", tid][metric]
                              for tid in target_ids] for s in SEEDS], dtype=np.float64)
        C, H, P = (branch(name) for name in (*BRANCHES, "parent"))
        Q = np.array([vals["atlas_query", tid][metric] for tid in target_ids], dtype=np.float64)
        differences = {
            "C_only_minus_head_only": C - H,
            "C_only_minus_parent": C - P,
            "head_only_minus_parent": H - P,
            "C_only_minus_query": C - Q,
            "head_only_minus_query": H - Q,
            "parent_minus_query": P - Q,
        }
        contrasts = {name: paired_summary(v, draws) for name, v in differences.items()}
        result["metrics"][metric] = {
            "main": contrasts["C_only_minus_head_only"],
            "contrasts": contrasts,
            "absolute_means": {"query": float(Q.mean()), "parent": float(P.mean()),
                               "C_only": float(C.mean()), "head_only": float(H.mean())},
            "per_seed_absolute_means": {"parent": P.mean(1).tolist(),
                                        "C_only": C.mean(1).tolist(), "head_only": H.mean(1).tolist()},
        }
    result["failure_counts"] = {
        name: sum(x["status"] != "ok" for x in rows if x["system"] == name)
        for name in sorted({x["system"] for x in rows})
    }
    return result


def smoke_gate(root, config, lock_sha):
    inputs = {}
    runs = config["matrices"]["atlas"]["smoke"]
    assert len(runs) == 2 and {r["branch"] for r in runs} == set(BRANCHES)
    assert len({r["name"] for r in runs}) == 2
    for run in runs:
        for phase, steps in (("continuous", 8), ("prefix", 4), ("resume", 4)):
            path = root / "atlas" / "smoke" / run["name"] / (phase + ".json")
            receipt = read(path)
            assert receipt["passed"] and receipt["execution_lock_sha256"] == lock_sha
            assert receipt["steps"] == steps and receipt["run"] == run and receipt["phase"] == phase
            assert sha(path.with_suffix(".pt")) == receipt["checkpoint_sha256"]
            inputs[str(path)] = sha(path)
    write(root / "atlas" / "smoke" / "complete.json", {
        "passed": True, "execution_lock_sha256": lock_sha, "inputs": inputs, "time": stamp()
    })


def verify_checkpoint_receipt(root, stage, run, done, lock_sha):
    """Bind each inference receipt to its declared adapted, parent, or Query model."""
    assert done["complete"] and done["execution_lock_sha256"] == lock_sha
    assert done["run"] == run
    kind = run.get("kind", "adapted")
    assert kind in ("adapted", "parent", "query")
    if kind == "query":
        assert stage == "formal" and run["name"] == "atlas_query"
        assert done.get("checkpoint_sha256") is None
        return
    if kind == "parent":
        assert stage == "formal" and run["name"] == f"atlas_I_s{run['seed']}_parent"
        assets = [x for x in read(root / "asset_audit.json")["atlas"] if x["seed"] == run["seed"]]
        assert len(assets) == 1
        asset = assets[0]
        assert sha(asset["path"]) == asset["sha256"] == done["checkpoint_sha256"]
        return
    assert run["branch"] in BRANCHES
    folder = root / "atlas" / stage / run["name"]
    training = read(folder / "training_complete.json")
    step = 64 if stage == "calibration" else 512
    checkpoint = folder / f"checkpoint_{step}.pt"
    assert training["complete"] and training["execution_lock_sha256"] == lock_sha
    assert training["run"] == run and training["new_steps"] == step
    assert sha(checkpoint) == training["checkpoint_sha256"] == done["checkpoint_sha256"]


def score(root, config, stage, lock_sha):
    from engramfold.evaluation.independent import fixed_mask_metrics
    from engramfold.evaluation.structure import read_atom_site_positions
    from engramfold.evaluation.post_validation import ca_pdb, parse_tm_score, load_reference_lddt

    # The inference manifest has no structure labels. Verify every terminal
    # receipt before opening the separately sealed reference manifest or CIFs.
    targets_inference = read(root / "panels" / "atlas" / (stage + "_inference.json"))["targets"]
    ids = [t["target_id"] for t in targets_inference]
    expected_n = 8 if stage == "calibration" else 96
    assert len(ids) == expected_n and len(set(ids)) == expected_n
    runs = evaluation_runs(config, stage)
    assert len({r["name"] for r in runs}) == len(runs)
    receipts = {}
    for run in runs:
        folder = root / "atlas" / stage / run["name"]
        done = read(folder / "complete.json")
        verify_checkpoint_receipt(root, stage, run, done, lock_sha)
        guard = read(folder / "label_guard.json")
        assert guard["passed"] and guard["probe_blocked"] and guard["guarded_reference_paths"] > 0
        assert done["no_evaluation_labels_parsed"]
        assert [x["target_id"] for x in done["records"]] == ids
        assert done["n_predictions"] == len(ids)
        assert done["failures"] == sum(x["status"] != "ok" for x in done["records"])
        for record in done["records"]:
            assert record["execution_lock_sha256"] == lock_sha
            assert record["status"] in ("ok", "failed")
            if record["status"] == "ok":
                assert sha(record["prediction_path"]) == record["prediction_sha256"]
        if stage == "calibration":
            assert all(x["status"] == "ok" for x in done["records"])
        receipts[run["name"]] = done

    out = root / "atlas" / (stage + "_analysis")
    write(out / "scoring_release.json", {"time": stamp(), "execution_lock_sha256": lock_sha,
                                         "predictions": len(runs) * len(ids)})
    targets = read(root / "panels" / "atlas" / (stage + "_reference.json"))["targets"]
    assert [t["target_id"] for t in targets] == ids
    refs = {}
    for t, inference in zip(targets, targets_inference, strict=True):
        assert t["sequence"] == inference["sequence"]
        path = Path(t["raw_mmcif_path"])
        assert sha(path) == t["raw_mmcif_sha256"]
        _, atoms = read_atom_site_positions(path, label_asym_id=t["source_label_asym_id"])
        ref = {i: np.asarray(v, dtype=np.float64) for (i, atom), v in atoms.items() if atom == "CA"}
        assert ref and set(ref) <= set(range(1, len(t["sequence"]) + 1))
        assert np.isfinite(np.stack(list(ref.values()))).all()
        if "reference_ca_indices" in t:
            assert sorted(ref) == t["reference_ca_indices"]
        refs[t["target_id"]] = ref

    assets = config.get("score", {}).get("atlas", {})
    lddt_path, tm_path = assets.get("lddt_path"), assets.get("tm_path")
    # An asset explicitly named in the lock must exist; missing configured
    # tools are engineering failures, not silently omitted outcome measures.
    lddt = load_reference_lddt(Path(lddt_path)) if lddt_path else None
    if tm_path:
        assert Path(tm_path).is_file(), tm_path
    metrics = ["ca_lddt"]
    if stage == "formal":
        if lddt is not None:
            metrics.append("residue_ca_lddt")
        if tm_path:
            metrics.append("tm_score_fixed_full_length")

    def zero(base, error=None):
        return dict(base, status="failed", **{m: 0.0 for m in metrics}, error=error)

    def calc(job):
        name, t, record = job
        base = {"system": name, "target_id": t["target_id"], "status": record["status"]}
        if record["status"] != "ok":
            return zero(base, record.get("error"))
        path = Path(record["prediction_path"])
        # A malformed/nonfinite/incomplete full-chain prediction is a retained
        # zero-score target. Reference or metric-tool errors remain fatal.
        try:
            _, atoms = read_atom_site_positions(path)
            pred = {i: np.asarray(v, dtype=np.float64) for (i, atom), v in atoms.items() if atom == "CA"}
            assert sorted(pred) == list(range(1, len(t["sequence"]) + 1)), "Incomplete predicted chain"
            assert np.isfinite(np.stack(list(pred.values()))).all()
        except Exception:
            return zero(base, traceback.format_exc())
        ref = refs[t["target_id"]]
        indices = sorted(ref)
        pair = float(fixed_mask_metrics(ref, pred)["ca_lddt"])
        assert np.isfinite(pair)
        row = dict(base, ca_lddt=pair, prediction_sha256=record["prediction_sha256"])
        px = np.stack([pred[i] for i in indices])
        rx = np.stack([ref[i] for i in indices])
        mask = np.ones((1, len(indices), 1))
        if lddt is not None:
            reference_pair = float(lddt(px[None], rx[None], mask)[0])
            assert abs(pair - reference_pair) < 1e-6, (name, t["target_id"], pair, reference_pair)
            row["implementation_check_abs"] = abs(pair - reference_pair)
            if stage == "formal":
                row["residue_ca_lddt"] = float(lddt(px[None], rx[None], mask, per_residue=True).mean())
        if stage == "formal" and tm_path:
            folder = out / "coordinates" / name / t["target_id"]
            folder.mkdir(parents=True, exist_ok=True)
            reference_pdb, prediction_pdb = folder / "reference.pdb", folder / "prediction.pdb"
            reference_pdb.write_text(ca_pdb(ref))
            prediction_pdb.write_text(ca_pdb({i: pred[i] for i in indices}))
            result = subprocess.run([str(tm_path), str(prediction_pdb), str(reference_pdb),
                                     "-l", str(len(t["sequence"]))], capture_output=True, text=True, check=True)
            (folder / "tm.txt").write_text(result.stdout + result.stderr)
            row["tm_score_fixed_full_length"] = parse_tm_score(result.stdout)
        assert all(np.isfinite(row[m]) for m in metrics)
        return row

    jobs = [(run["name"], t, record) for run in runs
            for t, record in zip(targets, receipts[run["name"]]["records"], strict=True)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(calc, jobs))
    write(out / "metric_records.json", rows)
    if stage == "calibration":
        chosen, means = select_multiplier(rows, runs)
        release = root / "atlas" / "release.json"
        result = {"passed": True, "multiplier": chosen, "pooled_absolute_dev_means": means,
                  "criterion": "equal-weight absolute Dev8 pair-lDDT over C_only/head_only; tie <=1e-6 chooses .3",
                  "execution_lock_sha256": lock_sha, "records_sha256": sha(out / "metric_records.json"),
                  "time": stamp()}
        if release.exists():
            old = read(release)
            assert old["passed"] and old["execution_lock_sha256"] == lock_sha
            assert old["multiplier"] == chosen and old["records_sha256"] == result["records_sha256"]
        else:
            write(release, result)
        return
    result = analyze(rows, ids, metrics)
    result.update(execution_lock_sha256=lock_sha, time=stamp())
    write(out / "analysis.json", result)
    write(root / "atlas" / "COMPLETE.json", {
        "complete": True, "execution_lock_sha256": lock_sha,
        "records_sha256": sha(out / "metric_records.json"), "analysis_sha256": sha(out / "analysis.json"),
        "n_predictions": len(rows), "time": stamp()
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--backbone", choices=["atlas"], default="atlas")
    parser.add_argument("--stage", choices=["smoke", "calibration", "formal"], required=True)
    args = parser.parse_args()
    config, lock_sha = contract(args.root)
    if args.stage == "smoke":
        smoke_gate(args.root, config, lock_sha)
    else:
        score(args.root, config, args.stage, lock_sha)


if __name__ == "__main__":
    main()
