"""Locked, resumable training workers for the OpenFold schedule study.

The folding function is the A800-validated differentiable_recycling module.
Labels are constructed separately; deployed forward calls cannot read them.
"""
import argparse
import copy
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
import random
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from engramfold.experiments.cross_backbone_protocol import schedule
from engramfold.experiments.differentiable_recycling import (
    TrainingScheduleHook, bind_pass, checkpoint_context, recurrent_forward, SCHEDULES,
)
from engramfold.experiments.openfold_adapter_runtime import structure_labels, native_loss
from engramfold.experiments.schedule_retraining_protocol import choose_lr
from engramfold.models.live_opm import LiveOPMAdapter


def read(p):
    return json.loads(Path(p).read_text())


def write(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic_torch(p, value):
    p = Path(p)
    tmp = p.with_name(p.name + f'.{os.getpid()}.tmp')
    with tmp.open('wb') as f:
        torch.save(value, f)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(p)


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_tree(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal_tree(x, y) for x, y in zip(a, b))
    return a == b


def verify(root):
    lock = read(root / 'execution_lock.json')
    for name, digest in lock['files'].items():
        assert sha(root / name) == digest, name
    parent = Path(lock['engineering_root'])
    assert sha(parent / 'preparation_lock.json') == lock['engineering_lock_sha256']
    for path, digest in lock['data_hashes'].items():
        assert sha(path) == digest, path
    return lock


class Engine:
    def __init__(self, root, lock):
        self.root, self.lock = root, lock
        self.lock_sha = sha(root / 'execution_lock.json')
        self.parent = Path(lock['engineering_root'])
        spec = importlib.util.spec_from_file_location('schedule_engineering_common', self.parent / 'staging/common.py')
        self.common = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.common)
        self.c = self.common.contract(self.parent, True)
        self.model, self.cfg = self.common.load_model(self.parent, self.c)
        self.cache = {}
        self.guard_active = False

        def audit(event, args):
            if not self.guard_active or event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
                return
            path = os.fsdecode(args[0])
            if path.endswith(('.cif.gz', '.a3m', '.sto')) or path.endswith('reference_manifest.json'):
                raise RuntimeError('Inference attempted target-evidence read: ' + path)
        sys.addaudithook(audit)

    @contextmanager
    def inference_guard(self):
        assert not self.guard_active
        self.guard_active = True
        try:
            yield
        finally:
            self.guard_active = False

    def prepare(self, target, labels=False):
        tid = target['target_id']
        if tid not in self.cache:
            f, esm = self.common.features(self.parent, target, self.cfg, self.c)
            self.cache[tid] = ({k: v.cpu() for k, v in f.items()}, esm.cpu(), None)
        f, esm, lab = self.cache[tid]
        if labels and lab is None:
            assert not self.guard_active
            path = self.parent / 'data' / (tid + '.cif.gz')
            assert sha(path) == self.lock['references'][tid]['sha256']
            lab = structure_labels(path, target, f['aatype'][..., 0])
            self.cache[tid] = (f, esm, lab)
        return ({k: v.cuda() for k, v in f.items()}, esm.cuda(),
                {k: v.cuda() for k, v in lab.items()} if labels else None)

    def frozen_check(self):
        assert all(not p.requires_grad and p.grad is None for p in self.model.parameters())
        assert self.common.state_hash(self.model) == self.c['frozen_backbone_sha256']

    def train_step(self, writer, optimizer, target, run, step, trace=False):
        f, esm, labels = self.prepare(target, True)
        torch.manual_seed(run['seed'] + step)
        random.seed(run['seed'] + step)
        np.random.seed(run['seed'] + step)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        tick = time.monotonic()
        state_gradients = [] if trace else None
        hook = TrainingScheduleHook(self.model.evoformer.blocks[0].outer_product_mean, writer, esm, run['strategy'])
        try:
            with checkpoint_context():
                pred = recurrent_forward(self.model, f, gradient_records=state_gradients)
                loss, terms = native_loss(pred, labels, self.cfg)
                assert torch.isfinite(loss), 'nonfinite loss'
                loss.backward()
                hook.validate()
            diagnostics = {}
            for name, p in writer.named_parameters():
                assert p.grad is not None, ('disconnected gradient', name)
                assert torch.isfinite(p.grad).all(), ('nonfinite gradient', name)
                diagnostics[name] = dict(norm=float(p.grad.double().norm()), nonzero=int(torch.count_nonzero(p.grad)))
            assert all(p.grad is None for p in self.model.parameters())
            if trace and run['strategy'] == 'first':
                assert any(x['pass_id'] == 1 and x['state'] == 'pair' and x['norm'] > 0 for x in state_gradients)
                assert any(x['pass_id'] == 3 and x['state'] == 'pair' and x['norm'] > 0 for x in state_gradients)
            norm = torch.nn.utils.clip_grad_norm_(writer.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            assert all(torch.isfinite(p).all() for p in writer.parameters())
            torch.cuda.synchronize()
            return dict(step=step + 1, target_id=target['target_id'], loss=float(loss.detach()),
                        components={k: float(v.detach()) for k, v in terms.items()}, gradient_norm=float(norm),
                        parameter_gradients=diagnostics, state_gradients=state_gradients,
                        seconds=time.monotonic() - tick, peak_gib=torch.cuda.max_memory_allocated() / 2**30,
                        events=hook.events, utc=now())
        finally:
            hook.remove()

    def checkpoint(self, out, writer, optimizer, run, step, order_hash, initial_sha):
        self.frozen_check()
        ck = dict(writer=writer.state_dict(), optimizer=optimizer.state_dict(), run=run, step=step,
                  execution_lock_sha256=self.lock_sha, frozen_backbone_sha256=self.c['frozen_backbone_sha256'],
                  order_sha256=order_hash, initial_writer_sha256=initial_sha,
                  rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state_all(),
                  rng_numpy=np.random.get_state(), rng_python=random.getstate(), scheduler=None)
        atomic_torch(out / 'latest.pt', ck)
        if step in (0, 384, 768, 1536):
            atomic_torch(out / f'step{step}.pt', ck)

    def restore(self, path, writer, optimizer, run, order_hash):
        ck = torch.load(path, map_location='cpu', weights_only=False)
        assert ck['run'] == run and ck['execution_lock_sha256'] == self.lock_sha
        assert ck['order_sha256'] == order_hash and ck['scheduler'] is None
        writer.load_state_dict(ck['writer'], strict=True)
        optimizer.load_state_dict(ck['optimizer'])
        assert equal_tree(writer.state_dict(), ck['writer'])
        assert equal_tree(optimizer.state_dict(), ck['optimizer'])
        torch.set_rng_state(ck['rng_cpu'])
        torch.cuda.set_rng_state_all(ck['rng_cuda'])
        np.random.set_state(ck['rng_numpy'])
        random.setstate(ck['rng_python'])
        return ck['step'], ck['initial_writer_sha256']

    def fit(self, run, out, calibration=False):
        train = read(self.root / 'train_manifest.json')
        order = [train[i] for i in schedule(run['seed'], len(train), run['steps'])]
        ids = [t['target_id'] for t in order]
        order_hash = hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()
        write(out / 'schedule.json', ids)
        torch.manual_seed(run['seed'])
        writer = LiveOPMAdapter(backbone='openfold', kind=run['kind'], rotation_seed=run['rotation']).cuda().eval()
        optimizer = torch.optim.AdamW(writer.parameters(), lr=run['lr'], weight_decay=.01)
        initial_sha = self.common.state_hash(writer)
        start = 0
        if (out / 'latest.pt').exists():
            start, saved_sha = self.restore(out / 'latest.pt', writer, optimizer, run, order_hash)
            assert saved_sha == initial_sha
            self.recover_log(out, start)
            write(out / f'recovery_{time.time_ns()}.json', dict(step=start, utc=now(), execution_lock_sha256=self.lock_sha))
        else:
            self.checkpoint(out, writer, optimizer, run, 0, order_hash, initial_sha)
        for step in range(start, run['steps']):
            row = self.train_step(writer, optimizer, order[step], run, step, trace=(step == 0))
            if calibration and step == 1:
                # Compare a real on-disk resume, then retain the replay's state.
                before = {k: v.detach().cpu().clone() for k, v in writer.state_dict().items()}
                restored, _ = self.restore(out / 'latest.pt', writer, optimizer, run, order_hash)
                assert restored == 1
                replay = self.train_step(writer, optimizer, order[step], run, step)
                diff = abs(row['loss'] - replay['loss']) / max(abs(row['loss']), 1e-12)
                assert diff <= 1e-5, ('resume forward mismatch', diff)
                delta = sum(float((v.detach().cpu().double() - before[k].double()).square().sum()) for k, v in writer.state_dict().items())
                energy = sum(float(v.double().square().sum()) for v in before.values())
                write(out / 'h100_resume_smoke.json', dict(passed=True, loss_relative_error=diff,
                      state_relative_l2=(delta / max(energy, 1e-30))**.5, first=row, replay=replay))
                row = replay
                del before
            with (out / 'training.jsonl').open('a') as f:
                f.write(json.dumps(row, allow_nan=False) + '\n')
                f.flush()
            write(out / 'progress.json', dict(run=run, step=step + 1, loss=row['loss'], seconds=row['seconds'], utc=now()))
            print(json.dumps(dict(run=run['name'], **{k: row[k] for k in ('step', 'loss', 'seconds', 'peak_gib')})), flush=True)
            if (step + 1) % 96 == 0 or step + 1 == run['steps'] or (calibration and step == 0):
                self.checkpoint(out, writer, optimizer, run, step + 1, order_hash, initial_sha)
        if calibration:
            assert read(out / 'h100_resume_smoke.json')['passed']
        self.frozen_check()
        write(out / 'training_complete.json', dict(complete=True, run=run, step=run['steps'],
              execution_lock_sha256=self.lock_sha, checkpoint_sha256=sha(out / 'latest.pt'), utc=now()))
        del optimizer
        return writer.requires_grad_(False)

    @staticmethod
    def recover_log(out, step):
        path = out / 'training.jsonl'
        if not path.exists():
            assert step == 0
            return
        raw = path.read_text().splitlines()
        retained = []
        for line in raw:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                break
            if row['step'] > step:
                break
            assert row['step'] == len(retained) + 1
            retained.append(line)
        assert len(retained) == step, ('incomplete checkpoint log', len(retained), step)
        if len(retained) != len(raw):
            backup = out / f'training_before_resume_{time.time_ns()}.jsonl'
            path.rename(backup)
            path.write_text('\n'.join(retained) + ('\n' if retained else ''))

    def predict(self, writer, run, out, targets, calibration=False):
        records = []
        for t in targets:
            tid = t['target_id']
            folder = out / 'predictions'
            folder.mkdir(exist_ok=True)
            receipt = folder / (tid + '.json')
            path = folder / (tid + '.cif')
            if receipt.exists():
                row = read(receipt)
                assert row['execution_lock_sha256'] == self.lock_sha
                if row['status'] == 'ok':
                    assert sha(path) == row['cif_sha256']
                records.append(row)
                continue
            tick = time.monotonic()
            try:
                with self.inference_guard(), torch.no_grad():
                    f, esm, _ = self.prepare(t)
                    torch.manual_seed(20260921)
                    prev = [None, None, None]
                    hook = TrainingScheduleHook(self.model.evoformer.blocks[0].outer_product_mean, writer, esm, run['strategy']) if writer is not None else None
                    try:
                        for i in range(4):
                            with bind_pass(i + 1):
                                pred, m, z, x, early = self.model.iteration({k: v[..., i] for k, v in f.items()}, prev, _recycle=True)
                            assert not early
                            prev = [m, z, x]
                        if hook is not None:
                            assert [x['pass_id'] for x in hook.events] == [1, 2, 3, 4]
                            assert [int(x['injected']) for x in hook.events] == list(SCHEDULES[run['strategy']])
                            assert all(not x['grad_enabled'] for x in hook.events)
                        xyz = pred['final_atom_positions'].detach().cpu().numpy()
                        mask = pred['final_atom_mask'].detach().cpu().numpy()
                        assert xyz.shape == (len(t['sequence']), 37, 3) and np.isfinite(xyz).all()
                        assert mask[:, 1].all()
                        self.common.export_cif(path, t['sequence'], xyz, mask)
                        npz = folder / (tid + '.npz')
                        tmp = folder / (tid + f'.{os.getpid()}.tmp')
                        with tmp.open('wb') as fsave:
                            np.savez(fsave, coordinates=xyz, mask=mask, sequence=t['sequence'])
                        tmp.replace(npz)
                        row = dict(target_id=tid, status='ok', cif_sha256=sha(path), npz_sha256=sha(npz), seconds=time.monotonic() - tick)
                        del pred, prev, m, z, x
                    finally:
                        if hook is not None:
                            hook.remove()
            except (RuntimeError, MemoryError) as exc:
                if calibration or 'target-evidence read' in str(exc):
                    raise
                row = dict(target_id=tid, status='failed', error=traceback.format_exc(), seconds=time.monotonic() - tick)
                torch.cuda.empty_cache()
            row['execution_lock_sha256'] = self.lock_sha
            write(receipt, row)
            records.append(row)
            write(out / 'prediction_progress.json', dict(done=len(records), total=len(targets), failures=sum(x['status'] != 'ok' for x in records)))
        self.frozen_check()
        return records

    def dev_scores(self, out, records):
        from engramfold.evaluation.structure import read_atom_site_positions
        from engramfold.evaluation.independent import fixed_mask_metrics
        targets = read(self.root / 'dev_manifest.json')
        scores = []
        assert [x['target_id'] for x in records] == [t['target_id'] for t in targets]
        for t, row in zip(targets, records):
            assert row['status'] == 'ok'
            tid = t['target_id']
            refpath = Path(self.lock['references'][tid]['path'])
            assert sha(refpath) == self.lock['references'][tid]['sha256']
            _, atoms = read_atom_site_positions(refpath, label_asym_id=t['source_label_asym_id'])
            ref = {i: v for (i, atom), v in atoms.items() if atom == 'CA'}
            _, atoms = read_atom_site_positions(out / 'predictions' / (tid + '.cif'), label_asym_id='A')
            pred = {i: v for (i, atom), v in atoms.items() if atom == 'CA'}
            assert sorted(pred) == list(range(1, len(t['sequence']) + 1))
            scores.append(dict(target_id=tid, ca_lddt=fixed_mask_metrics(ref, pred)['ca_lddt']))
        return scores


def select_if_complete(root, lock):
    reports = {}
    for run in lock['calibration']:
        p = root / 'calibration' / run['name'] / 'complete.json'
        if not p.exists():
            return False
        reports[run['name']] = read(p)
        assert reports[run['name']]['execution_lock_sha256'] == sha(root / 'execution_lock.json')
    chosen = choose_lr(reports, [t['target_id'] for t in read(root / 'dev_manifest.json')])
    selection = dict(**chosen, execution_lock_sha256=sha(root / 'execution_lock.json'),
                     calibration_receipts={run['name']: sha(root / 'calibration' / run['name'] / 'complete.json') for run in lock['calibration']})
    target = root / 'calibration_selection.json'
    with (root / 'selection.lck').open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        if target.exists():
            assert read(target) == selection
        else:
            write(target, selection)
    return True


def worker(root, stage):
    lock = verify(root)
    lk = sha(root / 'execution_lock.json')
    if stage == 'calibration':
        tasks = lock['calibration']
        category = 'calibration'
    else:
        assert select_if_complete(root, lock)
        selected = read(root / 'calibration_selection.json')
        tasks = [dict(r, lr=selected['learning_rate']) for r in lock['formal'] if r['wave'] == 1]
        tasks += [dict(name='query', kind='query', strategy='query', seed=20260921, steps=0)]
        category = 'formal'
    engine = None
    for run in tasks:
        out = root / category / run['name']
        out.mkdir(parents=True, exist_ok=True)
        with (out / 'worker.lck').open('a') as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            if (out / 'complete.json').exists():
                done = read(out / 'complete.json')
                assert done['execution_lock_sha256'] == lk and done['run'] == run
                continue
            assert not (out / 'failed.json').exists(), f'Explicit failure requires audit before retry: {out}'
            write(out / 'run.json', dict(run=run, execution_lock_sha256=lk))
            write(out / 'active.json', dict(job=os.environ.get('SLURM_JOB_ID'), task=os.environ.get('SLURM_ARRAY_TASK_ID'), host=os.uname().nodename, pid=os.getpid(), utc=now()))
            try:
                if engine is None:
                    engine = Engine(root, lock)
                writer = engine.fit(run, out, stage == 'calibration') if run['kind'] != 'query' else None
                manifest = 'dev_sequences.json' if stage == 'calibration' else 'inference_manifest.json'
                records = engine.predict(writer, run, out, read(root / manifest), stage == 'calibration')
                done = dict(complete=True, run=run, execution_lock_sha256=lk, records=records,
                            failures=sum(x['status'] != 'ok' for x in records), utc=now())
                if stage == 'calibration':
                    done['scores'] = engine.dev_scores(out, records)
                write(out / 'complete.json', done)
                del writer
                torch.cuda.empty_cache()
                (out / 'active.json').unlink()
            except BaseException:
                write(out / 'failed.json', dict(utc=now(), execution_lock_sha256=lk, error=traceback.format_exc(), job=os.environ.get('SLURM_JOB_ID')))
                raise
    if stage == 'calibration':
        select_if_complete(root, lock)
    else:
        receipts = [root / 'formal' / r['name'] / 'complete.json' for r in tasks]
        if all(p.exists() for p in receipts):
            write(root / 'wave1_complete.json', dict(complete=True, formal_fits=36, query_fits=0,
                  predictions=37 * 96, execution_lock_sha256=lk, utc=now(),
                  failures=sum(read(p)['failures'] for p in receipts),
                  receipts={str(p.relative_to(root)): sha(p) for p in receipts}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--stage', choices=('calibration', 'wave1'), required=True)
    args = p.parse_args()
    worker(args.root, args.stage)


if __name__ == '__main__':
    main()
