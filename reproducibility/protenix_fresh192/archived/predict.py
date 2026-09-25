"""Matched G+ inference and historical native replay under explicit locks."""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
import time
from pathlib import Path

import torch

from engramfold.experiments.protenix_fresh_fourcell.common import deny_evidence as deny_target_evidence_reads, validate_manifest, frozen_hash, read, sha, writer_engineering
from engramfold.experiments.interface_runtime import query_context
from engramfold.experiments.sequence_feature_cache import file_sha256, load_sequence_features
from engramfold.experiments.train_structure_control import make_runner
from engramfold.models.interface_heads import FrozenOPMDecoder, InterfaceHead
from engramfold.models.native_geometry import load_geometry_writer
from engramfold.models.plm_extension import make_writer
from engramfold.experiments.plm_extension_runtime import load_esmc, feature_provenance
from engramfold.protenix.query_only import make_student_query_only_features
from engramfold.protenix.replay import make_replay_module


def write_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def verify_feature_model(actual, trained):
    for key in ("model_name", "checkpoint_sha256", "input_dim", "layer"):
        if actual[key] != trained[key]:
            raise ValueError(f"PLM model provenance changed: {key}")


@torch.no_grad()
def run(args):
    os.environ["PROTENIX_ROOT_DIR"] = str(args.protenix_root)
    os.environ.setdefault("LAYERNORM_TYPE", "torch")
    from protenix.data.inference.infer_dataloader import get_inference_dataloader
    from protenix.utils.seed import seed_everything
    from protenix.utils.torch_utils import to_device
    from runner.batch_inference import get_default_runner
    from runner.dumper import DataDumper

    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest)
    engineering = bool(manifest.get("engineering"))
    lock = json.loads(args.checkpoint_lock.read_text())
    if args.seed != 101:
        raise ValueError("independent inference seed is fixed at101")
    if lock.get("schema") != "engramfold.protenix_fresh192.models.v1":
        raise ValueError("requires native geometry execution lock")
    if lock["manifest_sha256"] != file_sha256(args.manifest):
        raise ValueError("manifest/checkpoint lock mismatch")
    if manifest.get("role") == "confirmation" and len(manifest["targets"]) != 96:
        raise ValueError("confirmation requires exactly96 targets")
    if args.system not in lock["systems"]:
        raise ValueError("unlocked system")
    if not engineering:
        execution = read(args.output_root.parent / "execution_lock.json")
        if sha(args.manifest) != execution["inference_manifest_sha256"] or sha(args.checkpoint_lock) != execution["prediction_lock_sha256"]:
            raise ValueError("formal execution identity changed")
        if sha(args.feature_root / "index.json") != execution["feature_index_sha256"]:
            raise ValueError("feature identity changed")
    root = args.output_root / args.system
    root.mkdir(parents=True, exist_ok=True)
    file_lock = (root / "run.lock").open("a")
    fcntl.flock(file_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.chdir(root)  # Official ESM cache is isolated from other jobs.
    report_path = root / "report.json"
    identities = {
        "manifest_sha256": file_sha256(args.manifest),
        "checkpoint_lock_sha256": file_sha256(args.checkpoint_lock),
        "source_sha256": file_sha256(Path(__file__)),
        "system": args.system,
        "model_lock_sha256": lock["model_lock_sha256"],
    }
    report = {
        "schema": "engramfold.independent_folds.v1",
        **identities,
        "records": [],
        "complete": False,
        "seed": 101,
        "cycles": 4,
        "steps": 5,
        "samples": 1,
    }
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if any(report[k] != v for k, v in identities.items()):
            raise ValueError("resume identity changed")
        for rec in report["records"]:
            if rec["status"] == "ok" and file_sha256(rec["prediction_path"]) != rec["prediction_sha256"]:
                raise ValueError("completed prediction changed")
        if report["complete"]:
            write_json(root / "complete_checked.json", {"complete":True,"records":len(report["records"])})
            return
    task, writer, decoder = None, None, None
    is_official = args.system == "official_mini_esm"
    is_query = args.system == "query"
    base_name = "protenix_mini_esm_v0.5.0.pt" if is_official else "protenix_mini_default_v0.5.0.pt"
    base = lock["base"][base_name]
    if file_sha256(Path(base["path"])) != base["sha256"]:
        raise ValueError("base checkpoint changed")
    if is_official:
        for name in ("esm2_t36_3B_UR50D.pt", "esm2_t36_3B_UR50D-contact-regression.pt"):
            entry = lock["base"][name]
            if file_sha256(Path(entry["path"])) != entry["sha256"]:
                raise ValueError("official PLM checkpoint changed")
    if not is_official and not is_query:
        item = lock["adapters"][args.system]
        if file_sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError("adapter checkpoint changed")
        task = torch.load(item["path"], map_location="cpu", weights_only=False)
        if task["schema"] not in {"engramfold.native_geometry_checkpoint.v1", "engramfold.matched_gplus_checkpoint.v1", "engramfold.plm_extension_checkpoint.v1"}:
            raise ValueError("wrong adapter schema")
        if (
            task["step"] != item["step"]
            or task["geometry"] != item["geometry"]
            or task["stage"] != "task"
        ):
            raise ValueError("not the locked checkpoint/geometry")
        if task["protenix_checkpoint_sha256"] != base["sha256"]:
            raise ValueError("adapter/base mismatch")
        if task.get("feature_kind") == "C":
            plm = {t["target_id"]:load_esmc(args.feature_root,t["target_id"],__import__("hashlib").sha256(t["sequence"].encode()).hexdigest(),len(t["sequence"])) for t in manifest["targets"]}
            provenance = feature_provenance(args.feature_root)
        else:
            plm, provenance = load_sequence_features(args.feature_root, manifest["targets"])
        verify_feature_model(provenance, task["feature_provenance"])
        index = json.loads((args.feature_root / "index.json").read_text())
    model_started = time.monotonic()
    if is_official:
        runner = get_default_runner(
            seeds=[101],
            n_cycle=4,
            n_step=5,
            n_sample=1,
            dtype="fp32",
            model_name="protenix_mini_esm_v0.5.0",
            use_msa=False,
            use_template=False,
            trimul_kernel="torch",
            triatt_kernel="torch",
            enable_cache=False,
            enable_fusion=False,
            enable_tf32=False,
        )
        runner.configs.mc_dropout_apply_rate = 0.0
        runner.model.configs.mc_dropout_rate = 0.0
    else:
        runner = make_runner(args)
    torch.cuda.synchronize()
    report["folding_model_load_seconds"] = time.monotonic() - model_started
    model = runner.model.eval().requires_grad_(False)
    frozen_before = frozen_hash(model)
    checkpoint_state = torch.load(base["path"], map_location="cpu", weights_only=False)["model"]
    expected_keys = {k.removeprefix("module.") for k in checkpoint_state}
    actual_keys = set(model.state_dict())
    report["checkpoint_missing_keys"] = sorted(actual_keys - expected_keys)
    report["checkpoint_unexpected_keys"] = sorted(expected_keys - actual_keys)
    del checkpoint_state
    if expected_keys != actual_keys:
        raise ValueError("native checkpoint/model keys do not match exactly")
    original_msa = model.msa_module
    if task:
        report["plm_model_load_seconds"] = index.get("model_load_seconds")
        common = to_device(task["common"], runner.device)
        decoder = FrozenOPMDecoder(
            common["weight"],
            common["bias"],
            factor_dim=32,
            depth=common["depth"],
            eps=common["eps"],
        ).to(runner.device)
        if task["schema"] == "engramfold.plm_extension_checkpoint.v1":
            writer = make_writer("protenix",task["geometry"]["kind"],task["geometry"]["rotation_seed"],task["seed"],task["input_dim"],runner.device)
            writer.load_state_dict(task["writer"],strict=True)
            writer.eval().requires_grad_(False)
        elif task["schema"] == "engramfold.matched_gplus_checkpoint.v1":
            if task["kind"] != "generic_plus" or task["train_size"] != 384 or task["budget"] != 1536:
                raise ValueError("not matched Train384 G+")
            writer = InterfaceHead("generic_plus").to(runner.device)
            writer.load_state_dict(task["writer"], strict=True)
            writer.eval()
        else:
            writer = load_geometry_writer(task, runner.device).eval()
    completed = {r["target_id"] for r in report["records"]}
    todo = [t for t in manifest["targets"] if t["target_id"] not in completed]
    input_path = root / "inputs.json"
    input_path.write_text(
        json.dumps(
            [
                {
                    "name": t["target_id"],
                    "sequences": [{"proteinChain": {"sequence": t["sequence"], "count": 1}}],
                }
                for t in todo
            ]
        )
    )
    runner.configs.input_json_path = str(input_path)
    runner.configs.num_workers = 0
    plm_timing = {}
    # Time native official ESM computation without changing its inputs/outputs.
    if is_official:
        from protenix.data.esm import esm_featurizer as ef

        original_compute, original_load = ef.compute_ESM_embeddings, ef.load_esm_model

        def timed_compute(*a, **kw):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.monotonic()
            result = original_compute(*a, **kw)
            torch.cuda.synchronize()
            for label in a[3]:
                plm_timing[label] = {
                    "seconds": time.monotonic() - start,
                    "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "cached": label not in result,
                }
            return result

        def timed_load(*a, **kw):
            start = time.monotonic()
            result = original_load(*a, **kw)
            torch.cuda.synchronize()
            report["plm_model_load_seconds"] = time.monotonic() - start
            return result

        ef.compute_ESM_embeddings, ef.load_esm_model = timed_compute, timed_load
    teacher_root = Path(task["teacher_root"]) if task else args.forbidden_root
    cache_root = Path(task["cache_root"]) if task else args.forbidden_root
    with (
        deny_target_evidence_reads(manifest, teacher_root),
        deny_target_evidence_reads(manifest, cache_root),
    ):
        seed_everything(seed=101, deterministic=True)
        started = time.monotonic()
        loader = get_inference_dataloader(configs=runner.configs)
        report["dataset_initialization_seconds"] = time.monotonic() - started
        iterator = iter(loader)
        for target in todo:
            name = target["target_id"]
            record = {"target_id": name, "status": "failed"}
            seed_everything(seed=101, deterministic=True)
            started = time.monotonic()
            try:
                data, atoms, errors = next(iterator)[0]
                if errors:
                    raise RuntimeError(f"feature construction: {errors}")
                # Dataloader order must match the declared input list.
                if str(data.get("sample_name", name)) != name:
                    raise RuntimeError("dataloader order mismatch")
                record["feature_seconds"] = time.monotonic() - started
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started = time.monotonic()
                model.msa_module = original_msa
                if is_official:
                    emb = data["input_feature_dict"]["esm_token_embedding"]
                    if not torch.isfinite(emb).all() or not emb.abs().sum() > 0:
                        raise ValueError("official ESM missing/invalid; zero fallback forbidden")
                    record["plm"] = plm_timing[name + "_1"]
                else:
                    features = make_student_query_only_features(data["input_feature_dict"])
                    for key in list(features):
                        if "msa" in key or key in {
                            "deletion_matrix",
                            "num_alignments",
                            "num_alignments_all_seq",
                        }:
                            features.pop(key)
                    query = query_context(model, to_device(copy.deepcopy(features), runner.device))
                    if task and (plm[name].shape[0] != len(target["sequence"]) or query["query_update"].shape[:2] != (len(target["sequence"]),) * 2):
                        raise ValueError("full sequence feature length mismatch")
                    if engineering and task:
                        record["writer_engineering"] = writer_engineering(writer, task, plm[name].to(runner.device), query, decoder)
                    update = (
                        query["query_update"]
                        if is_query
                        else writer(plm[name].to(runner.device), decoder=decoder, **query)
                    )
                    model.msa_module = (
                        make_replay_module(original_msa, update).to(runner.device).eval()
                    )
                    calls = []
                    model.msa_module.provider.register_forward_hook(lambda mod, inp, out: calls.append(tuple(out.shape)))
                    data["input_feature_dict"] = features
                    if task:
                        record["plm"] = {
                            "seconds": index["records"][name].get("embedding_seconds"),
                            "peak_gib": index["records"][name].get("peak_gib"),
                            "cached": False,
                        }
                torch.cuda.synchronize()
                record["adapter_seconds"] = time.monotonic() - started
                record["adapter_peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
                seed_everything(seed=101, deterministic=True)
                torch.cuda.reset_peak_memory_stats()
                started = time.monotonic()
                prediction = runner.predict(data)
                if len(calls) != 4:
                    raise RuntimeError("recycle injection count changed")
                record["injection_shapes"] = calls
                torch.cuda.synchronize()
                record["fold_seconds"] = time.monotonic() - started
                record["fold_peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
                DataDumper(
                    base_dir=str(root / "predictions"),
                    need_atom_confidence=False,
                    sorted_by_ranking_score=True,
                ).dump(
                    dataset_name="",
                    pdb_id=name,
                    seed=101,
                    pred_dict=prediction,
                    atom_array=atoms,
                    entity_poly_type=data["entity_poly_type"],
                )
                path = (
                    root
                    / "predictions"
                    / name
                    / "seed_101"
                    / "predictions"
                    / f"{name}_sample_0.cif"
                )
                from engramfold.evaluation.structure import read_atom_site_positions
                import numpy as np
                _, pos = read_atom_site_positions(path)
                ca = {i:np.asarray(v) for (i,a),v in pos.items() if a=="CA"}
                if sorted(ca) != list(range(1,len(target["sequence"])+1)) or not np.isfinite(np.stack(list(ca.values()))).all():
                    raise RuntimeError("full output length or finite-coordinate contract failed")
                record.update(status="ok", prediction_path=str(path), prediction_sha256=file_sha256(path), full_length=len(ca))
                del prediction
            except Exception as exc:
                import re
                error = f"{type(exc).__name__}: {exc}"
                retryable = bool(re.search(r"hip error|hsa_status_error|memory access fault|out of memory|timed out|temporarily unavailable",error,re.I))
                attempt_file=root / "attempts" / (name+".json")
                attempt_file.parent.mkdir(exist_ok=True)
                attempts=read(attempt_file) if attempt_file.exists() else []
                attempts.append(dict(error=error,time=time.time(),retryable=retryable))
                write_json(attempt_file,attempts)
                if not retryable:
                    raise
                if len(attempts)<3:
                    raise SystemExit(75)
                record["error"]=error
                record["attempts"]=3
                report["records"].append(record)
                write_json(report_path,report)
                # HIP context may be poisoned: restart before the next target.
                raise SystemExit(75)
            report["records"].append(record)
            write_json(report_path, report)
            print(
                json.dumps(
                    {
                        "target_id": name,
                        "status": record["status"],
                        "completed": len(report["records"]),
                    }
                ),
                flush=True,
            )
    model.msa_module = original_msa
    if frozen_hash(model) != frozen_before:
        raise RuntimeError("frozen backbone changed")
    report["frozen_sha256"]=frozen_before
    report["frozen_unchanged"]=True
    report["complete"] = len(report["records"]) == len(manifest["targets"])
    write_json(report_path, report)
    if report["complete"]: write_json(root / "complete_checked.json", {"complete":True,"records":len(report["records"])})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in [
        "manifest",
        "checkpoint-lock",
        "output-root",
        "protenix-root",
        "feature-root",
        "forbidden-root",
    ]:
        p.add_argument("--" + name, type=lambda x: Path(x).resolve(), required=True)
    p.add_argument("--system", required=True)
    p.add_argument("--seed", type=int, default=101)
    run(p.parse_args())


if __name__ == "__main__":
    main()
