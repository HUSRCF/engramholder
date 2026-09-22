"""Native geometry inference under explicit manifest/checkpoint locks."""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
import time
from pathlib import Path

import torch

from engramfold.experiments.extension_runtime import MODELS, make_runner
from engramfold.experiments.inference_read_guard import deny_target_evidence_reads
from engramfold.experiments.interface_runtime import query_context
from engramfold.experiments.sequence_feature_cache import file_sha256, load_sequence_features
from engramfold.models.direction_extension import load_extension_writer
from engramfold.models.interface_heads import FrozenOPMDecoder
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


def capture_ln(module_name, ln_records):
    def hook(_module, inputs):
        if not ln_records and inputs[0].shape[-1] == 128:
            value = inputs[0].detach().float()
            means = value.mean(-1).flatten()
            variances = value.var(-1, unbiased=False).flatten()
            q = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=value.device)
            ln_records.append(
                {
                    "module": module_name,
                    "channel_mean_quantiles": torch.quantile(means, q).cpu().tolist(),
                    "channel_variance_quantiles": torch.quantile(variances, q).cpu().tolist(),
                }
            )

    return hook


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
    lock = json.loads(args.checkpoint_lock.read_text())
    if args.seed != 101:
        raise ValueError("independent inference seed is fixed at101")
    if lock.get("schema") != "engramfold.direction_extension_lock.v1":
        raise ValueError("requires native geometry execution lock")
    if lock["manifest_sha256"] != file_sha256(args.manifest):
        raise ValueError("manifest/checkpoint lock mismatch")
    if manifest.get("role") == "confirmation" and len(manifest["targets"]) != 96:
        raise ValueError("confirmation requires exactly96 targets")
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
        if report["complete"]:
            return
    task, writer, decoder = None, None, None
    is_official = args.system == "official_mini_esm"
    is_query = args.system in {"query", "query_tiny", "query_mini"}
    base_name = "protenix_mini_esm_v0.5.0.pt" if is_official else MODELS[args.backbone] + ".pt"
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
        if task["schema"] not in {
            "engramfold.native_geometry_checkpoint.v1",
            "engramfold.direction_extension_checkpoint.v1",
        }:
            raise ValueError("wrong adapter schema")
        if (
            task["step"] != item["step"]
            or task["geometry"] != item["geometry"]
            or task["stage"] != "task"
        ):
            raise ValueError("not the locked checkpoint/geometry")
        if task["protenix_checkpoint_sha256"] != base["sha256"]:
            raise ValueError("adapter/base mismatch")
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
    model = runner.model.eval()
    checkpoint_state = torch.load(base["path"], map_location="cpu", weights_only=False)["model"]
    expected_keys = {k.removeprefix("module.") for k in checkpoint_state}
    actual_keys = set(model.state_dict())
    report["checkpoint_missing_keys"] = sorted(actual_keys - expected_keys)
    report["checkpoint_unexpected_keys"] = sorted(expected_keys - actual_keys)
    del checkpoint_state
    allowed_unused = {"input_embedder.linear_esm.weight"} if args.backbone == "tiny" else set()
    if actual_keys - expected_keys or expected_keys - actual_keys != allowed_unused:
        raise ValueError("checkpoint mismatch outside documented unused Tiny ESM projection")
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
        writer = load_extension_writer(task, runner.device).eval()
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
                    update = (
                        query["query_update"]
                        if is_query
                        else writer(plm[name].to(runner.device), decoder=decoder, **query)
                    )
                    residual = update - query["query_update"]
                    centered = residual - residual.mean(-1, keepdim=True)
                    record["residual_diagnostics"] = {
                        "norm_over_query": float(
                            residual.norm() / (query["query_update"].norm() + 1e-12)
                        ),
                        "centered_norm_fraction": float(
                            centered.norm() / (residual.norm() + 1e-12)
                        ),
                    }
                    model.msa_module = (
                        make_replay_module(original_msa, update).to(runner.device).eval()
                    )
                    data["input_feature_dict"] = features
                    if task:
                        record["plm"] = {
                            "seconds": index["records"][name]["embedding_seconds"],
                            "peak_gib": index["records"][name].get("peak_gib"),
                            "cached": False,
                        }
                torch.cuda.synchronize()
                record["adapter_seconds"] = time.monotonic() - started
                record["adapter_peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
                seed_everything(seed=101, deterministic=True)
                torch.cuda.reset_peak_memory_stats()
                started = time.monotonic()
                ln_records = []
                hooks = []

                for module_name, module in model.msa_module.named_modules():
                    if "pair_stack" in module_name and "layernorm" in type(module).__name__.lower():
                        hooks.append(
                            module.register_forward_pre_hook(capture_ln(module_name, ln_records))
                        )
                try:
                    prediction = runner.predict(data)
                finally:
                    for hook in hooks:
                        hook.remove()
                record["first_pair_stack_layernorm"] = ln_records[0] if ln_records else None
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
                record.update(
                    status="ok", prediction_path=str(path), prediction_sha256=file_sha256(path)
                )
                del prediction
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                if "out of memory" in str(exc).lower():
                    torch.cuda.empty_cache()
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
    report["complete"] = len(report["records"]) == len(manifest["targets"])
    write_json(report_path, report)


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
    p.add_argument("--backbone", choices=["mini", "tiny"], required=True)
    p.add_argument("--system", required=True)
    p.add_argument("--seed", type=int, default=101)
    run(p.parse_args())


if __name__ == "__main__":
    main()
