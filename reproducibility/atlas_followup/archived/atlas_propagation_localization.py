"""Read-only, same-site Atlas versus Query propagation observations.

No training, structure labels, architecture edits, or cross-layer raw norm ratios.
The original inference runner and sealed trained-head implementation are reused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def real_state(tensor, length, pair):
    """Clone immediately: Atlas uses in-place additions after many hook sites."""
    expected = 4 if pair else 3
    if tensor.ndim != expected or tensor.shape[0] != 1:
        raise ValueError('Expected a single complete sequence batch')
    if tensor.shape[1] < length or (pair and tensor.shape[2] < length):
        raise ValueError('Truncated sequence')
    value = tensor[0, :length, :length] if pair else tensor[0, :length]
    value = value.detach().cpu().clone()
    if not torch.isfinite(value).all():
        raise ValueError('Nonfinite observed state')
    return value


def compare_state(value, reference):
    """FP64 reduction of saved finite-precision states at one identical site."""
    if value.shape != reference.shape or value.dtype != reference.dtype:
        raise ValueError('Site shape/dtype mismatch')
    a, q = value.double(), reference.double()
    delta = a - q
    energy = float(delta.square().sum())
    qenergy, aenergy = float(q.square().sum()), float(a.square().sum())
    mean_energy = float(delta.mean(-1).square().sum()) * delta.shape[-1]
    return dict(shape=list(a.shape), dtype=str(value.dtype),
                query_rms=(qenergy / q.numel()) ** .5,
                adapted_rms=(aenergy / a.numel()) ** .5,
                delta_rms=(energy / delta.numel()) ** .5,
                relative_delta_rms=(energy / qenergy) ** .5 if qenergy else None,
                max_absolute_delta=float(delta.abs().max()),
                changed_fraction=float(torch.count_nonzero(delta)) / delta.numel(),
                cosine=float((a * q).sum()) / (aenergy * qenergy) ** .5 if aenergy * qenergy else None,
                channel_constant_delta_energy_fraction=mean_energy / energy if energy else None)


class Observer:
    def __init__(self, length, reference=None):
        self.length, self.reference = length, reference
        self.records, self.states = defaultdict(list), defaultdict(list)
        self.handles = []
        self.residuals = []

    def take(self, name, tensor, pair=True):
        value = real_state(tensor, self.length, pair)
        k = len(self.records[name])
        if self.reference is None:
            self.states[name].append(value)
            ref = value
        else:
            ref = self.reference[name][k]
        self.records[name].append(compare_state(value, ref))

    def residual(self, value):
        x = value.detach().cpu().double()
        assert x.shape == (self.length, self.length, 128) and torch.isfinite(x).all()
        self.residuals.append(dict(rms=float(x.square().mean().sqrt()),
                                   max_abs=float(x.abs().max()), dtype=str(value.dtype)))

    def install(self, model):
        def pre(module, name, index, pair):
            def hook(m, args, kwargs):
                key = 'z' if pair else 's'
                self.take(name, args[index] if len(args) > index else kwargs[key], pair)
            self.handles.append(module.register_forward_pre_hook(hook, with_kwargs=True))

        def post(module, name, index=None, pair=True):
            def hook(m, args, out):
                self.take(name, out if index is None else out[index], pair)
            self.handles.append(module.register_forward_hook(hook))

        first = model.lm_stack.blocks[0]
        assert len(model.lm_stack.blocks) == 4
        pre(first, 'lm_input.z', 1, True)
        pre(first, 'lm_input.s', 0, False)
        # Installed after the adapter hook: this is the full actual OPM-like update.
        post(first.pairwise_prod_diff, 'interface_update.z')
        pre(first.tri_mul_out, 'after_injection.z', 0, True)
        for i, block in enumerate(model.lm_stack.blocks, 1):
            post(block, f'lm_block{i}.z', 1)
            post(block, f'lm_block{i}.s', 0, False)
        for stream in ('s', 'z'):
            pair = stream == 'z'
            projection = getattr(model, f'proj_{stream}_lm')
            post(projection[0], f'projection_ln.{stream}', pair=pair)
            post(projection, f'projection_out.{stream}', pair=pair)
            post(getattr(model, f'recycle_{stream}'), f'recycle_out.{stream}', pair=pair)
            pre(model.main_stack, f'trunk_input.{stream}', 1 if pair else 0, pair)
            post(model.main_stack, f'trunk_end.{stream}', 1 if pair else 0, pair)

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def validate(self):
        assert len(self.records) == 22, list(self.records)
        assert all(len(v) == 5 for v in self.records.values())
        if self.reference is not None:
            assert set(self.records) == set(self.reference)
        assert len(self.residuals) in (0, 5)


def structure_difference(a, q):
    # All complete-sequence C-alpha pairs, no label-derived mask/cutoff.
    a, q = torch.as_tensor(a).double(), torch.as_tensor(q).double()
    da, dq = torch.cdist(a, a), torch.cdist(q, q)
    mask = torch.triu(torch.ones(len(a), len(a), dtype=torch.bool), diagonal=1)
    delta = (da - dq)[mask]
    ac, qc = a - a.mean(0), q - q.mean(0)
    # This ROCm build omits CPU LAPACK; use NumPy for the 3x3 diagnostic only.
    u, _, vh = np.linalg.svd((ac.T @ qc).numpy())
    sign = np.eye(3, dtype=np.float64)
    sign[-1, -1] = np.linalg.det(u @ vh)
    fit = ac @ torch.from_numpy(u @ sign @ vh)
    return dict(ca_pair_distance_delta_rms=float(delta.square().mean().sqrt()),
                aligned_ca_rmsd=float((fit - qc).square().sum(-1).mean().sqrt()),
                unaligned_ca_max_difference=float((a-q).abs().max()),
                interpretation='Change relative to Query, not structure quality')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--index', type=int, required=True)
    a = p.parse_args()
    root = a.root
    config = read(root/'lock.json')
    target = config['targets'][a.index]
    out = root/'results'/target['target_id']
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    lease = (out/'lease').open('a')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for path, expected in config['files'].items():
        assert sha(path) == expected, path
    lock_sha = sha(root/'lock.json')
    if (out/'complete.json').exists():
        assert read(out/'complete.json')['lock_sha256'] == lock_sha
        return
    from engramfold.experiments import atlas_adapter_runtime as legacy
    from engramfold.experiments.atlas_posttraining_calibration.predict import install_label_guard
    from engramfold.models.live_prod_diff import LiveProdDiffAdapter, LiveProdDiffHook
    from engramfold.models.atlas_posttraining_channels import AtlasCalibratedHead
    from atlasfold.pretrained import get_runner
    from atlasfold.runner import get_sampling_config

    parent_root = Path(config['parent_root'])
    blocked = install_label_guard(parent_root, read(parent_root/'execution_lock.json'), out)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    base = Path(config['base'])
    model = legacy.load_model(base).eval()
    frozen = legacy.state_hash(model)
    assert frozen == config['frozen_hash']
    runner = get_runner(model)
    length = len(target['sequence'])
    cache = torch.load(base/'data'/f"{target['target_id']}.pt", map_location='cpu', weights_only=False)
    assert cache['sequence_sha256'] == hashlib.sha256(target['sequence'].encode()).hexdigest()
    features = cache['features'].float().cuda()
    assert features.shape == (length, 480)
    query = None
    qcoords = None

    def fold(writer, observer):
        hook = None
        original = None
        if writer is not None:
            if observer is not None:
                original = writer.residual
                def observe_residual(*args, **kwargs):
                    delta = original(*args, **kwargs)
                    observer.residual(delta)
                    return delta
                writer.residual = observe_residual
            hook = LiveProdDiffHook(model.lm_stack.blocks[0].pairwise_prod_diff, writer, features)
        if observer is not None:
            observer.install(model)
        rng_before = (torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone())
        try:
            result = runner.fold(target['target_id'], target['sequence'], num_samples=1,
                                 seeds=1, num_recycles=4, mlm_prob=.15,
                                 sampling_config=get_sampling_config(length))
            assert all(np.isfinite(x).all() for x in (result.best.coordinates, result.best.plddt, result.best.pae))
            assert result.best.coordinates.shape == (length, 14, 3)
            if hook:
                assert len(hook.calls) == 5
            if observer:
                observer.validate()
            assert torch.equal(rng_before[0], torch.get_rng_state())
            assert torch.equal(rng_before[1], torch.cuda.get_rng_state())
            return result
        finally:
            if hook:
                hook.remove()
            if observer:
                observer.remove()
            if original:
                writer.residual = original

    def exact_output(x, y):
        return all(np.array_equal(getattr(x.best, k), getattr(y.best, k))
                   for k in ('coordinates', 'plddt', 'pae')) and x.best.ptm == y.best.ptm

    rows = []
    for run in config['runs']:
        start = time.monotonic()
        writer = None
        if run['branch'] != 'query':
            ck = torch.load(run['checkpoint'], map_location='cpu', weights_only=False)
            head = LiveProdDiffAdapter(kind='factor', rotation_seed=None).cuda().eval()
            writer = head if run['branch'] == 'parent' else AtlasCalibratedHead(head, run['branch']).cuda().eval()
            writer.load_state_dict(ck['writer'], strict=True)
            writer.requires_grad_(False)
        obs = Observer(length, query)
        result = fold(writer, obs)
        if query is None:
            query = dict(obs.states)
            qcoords = result.best.coordinates[:, 1].copy()
            path = out/'query_states.pt'
            torch.save(query, path)
            np.save(out/'query_ca.npy', qcoords)
            repeat = Observer(length, query)
            again = fold(None, repeat)
            assert exact_output(result, again)
            assert all(r['max_absolute_delta'] == 0 for values in repeat.records.values() for r in values)
            plain = fold(None, None)
            assert exact_output(result, plain)
            write(out/'query_smoke.json', dict(passed=True, repeated_states_exact=True,
                  observer_vs_plain_output_exact=True, query_states_sha256=sha(path)))
        elif run['branch'] == 'parent' and run['seed'] == 20260923:
            plain = fold(writer, None)
            assert exact_output(result, plain)
            write(out/'adapted_smoke.json', dict(passed=True, observer_vs_plain_output_exact=True))
        ca = result.best.coordinates[:, 1].copy()
        np.save(out/f"{run['name']}_ca.npy", ca)
        (out/f"{run['name']}.cif").write_text(result.best.to_mmcif())
        row = dict(run=run, target_id=target['target_id'], length=length,
                   sites=dict(obs.records), residuals=obs.residuals,
                   structure_change=structure_difference(ca, qcoords), seconds=time.monotonic()-start,
                   lock_sha256=lock_sha, all_passes=5, no_structure_labels=True)
        write(out/f"{run['name']}.json", row)
        rows.append(row)
        write(out/'progress.json', dict(completed=len(rows), total=len(config['runs']), latest=run['name']))
        print('OBSERVED', target['target_id'], run['name'], row['seconds'], flush=True)
        del writer, result, obs
    assert legacy.state_hash(model) == frozen and not blocked
    write(out/'complete.json', dict(complete=True, lock_sha256=lock_sha,
          forward_count=13, systems=len(rows), frozen_hash=frozen, no_structure_scoring=True,
          files={f.name:sha(f) for f in out.iterdir() if f.suffix in ('.json','.npy','.pt','.cif') and f.name!='complete.json'}))


if __name__ == '__main__':
    main()
