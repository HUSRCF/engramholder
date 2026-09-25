"""Rebuild saved Atlas probe contrasts and ESMC prediction-distance changes.

This checks records and source bindings, not hidden feature extraction, GPU
training, checkpoint tensors or structure quality against reference coordinates.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


def verify(root):
    root = Path(root)
    folder = root / 'reproducibility/atlas_followup/complementarity'
    read = lambda p: json.loads(p.read_text())
    evidence = lambda name: read(root / 'evidence' / (name + '.json'))
    provenance = root / 'notes/writing_branch_20260922/bundled_source_provenance.json'
    bindings = read(provenance) if provenance.exists() else {}

    def original_sha(path):
        key = str(path.relative_to(root))
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if key in bindings:
            assert actual == bindings[key]['bundled_sha256'], key
            return bindings[key]['original_sha256']
        return actual

    checks, max_error = 0, 0.0

    def equal(actual, expected, label):
        nonlocal checks, max_error
        a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), label
        error = float(np.max(np.abs(a - b)))
        assert error < 1e-12, (label, error)
        checks += 1
        max_error = max(max_error, error)

    lock = evidence('atlas_probe_execution_lock')
    lock_sha = original_sha(root / 'evidence/atlas_probe_execution_lock.json')
    summary = evidence('atlas_probe_summary')
    statistics = evidence('atlas_probe_statistics')
    assert summary['complete'] and summary['lock_sha256'] == lock_sha
    assert summary['fits'] == 9 and summary['updates'] == 13824
    assert lock['steps'] == 1536 and lock['parameter_count'] == 131520
    assert lock['checkpoint_rule'] == 'fixed step1536' and lock['new_folding_training'] == 0
    assert lock['engineering_amendment']['formal_updates_before_repair'] == 0
    for local, suffix in [('PROTOCOL.md', '/PROTOCOL.md'),
                          ('archived/atlas_plm_probe.py', '/src/engramfold/experiments/atlas_plm_probe.py')]:
        assert original_sha(folder / local) in {v for k, v in lock['code_files'].items() if k.endswith(suffix)}, local
    ids = [t['target_id'] for t in lock['dev']]
    train_ids = [t['target_id'] for t in lock['train']]
    assert len(set(ids)) == 8 and len(set(train_ids)) == 96 and not set(ids) & set(train_ids)
    assert summary['target_ids'] == ids
    arms, seeds = lock['arms'], lock['seeds']
    assert set(arms) == {'native_view', 'esmc', 'esmc_permuted'}
    assert seeds == [20260923, 20260924, 20260925]
    smoke = read(folder / 'smoke/complete.json')
    assert smoke['passed'] and smoke['lock_sha256'] == lock_sha and smoke['physical_updates'] == 36
    assert all(r['exact_resume'] and r['parameters'] == 131520 for r in smoke['runs'])
    integrity = read(folder / 'analysis/remote_integrity.json')
    assert integrity['passed'] and integrity['code_lock_sha256'] == lock_sha
    receipts = {r['name']: r for r in integrity['fits']}
    values = {metric: {} for metric in ['ce', 'long_ce', 'close_ce']}
    pair_counts = None
    for arm in arms:
        train_means, final = [], []
        for index, seed in enumerate(seeds):
            name = f'{arm}_s{seed}'
            run = folder / 'formal' / name
            done = read(run / 'complete.json')
            assert done['complete'] and done['lock_sha256'] == lock_sha
            assert done['arm'] == arm and done['seed'] == seed
            assert done['parameters'] == 131520 and done['steps'] == 1536
            receipt = receipts[name]
            assert receipt['checkpoint_sha256'] == done['checkpoint_sha256']
            assert receipt['finite'] and receipt['optimizer_steps'] == [1536]
            for step in [0, 384, 768, 1536]:
                path = run / f'dev_step{step}.json'
                rows = read(path)
                assert original_sha(path) == summary['files'][str(path.relative_to(folder))]
                assert [r['target_id'] for r in rows] == ids
                counts = [r['pair_count'] for r in rows]
                if pair_counts is None:
                    pair_counts = counts
                assert counts == pair_counts and min(counts) > 0
                equal(np.mean([r['ce'] for r in rows]), summary['curves'][arm][str(step)][index], (name, step))
            final.append(rows)
            train = read(run / 'train_final.json')
            assert [r['target_id'] for r in train] == train_ids
            train_means.append(np.mean([r['ce'] for r in train]))
        for metric in values:
            values[metric][arm] = np.array([[r[metric] for r in rows] for rows in final])
        equal(values['ce'][arm], summary['values'][arm], arm)
        equal(values['ce'][arm].mean(), summary['means'][arm], arm)
        equal(np.mean(train_means), summary['train_means'][arm], arm)

    draws = np.random.default_rng(20260926).integers(0, 8, (20000, 8))
    comparisons = 0
    for metric, arrays in values.items():
        for name, left in [('primary_native_minus_esmc', 'native_view'),
                           ('secondary_permuted_minus_esmc', 'esmc_permuted')]:
            difference = arrays[left] - arrays['esmc']
            target = difference.mean(0)
            actual = dict(mean=float(target.mean()), per_target=target,
                          per_seed=difference.mean(1), ci95=np.quantile(target[draws].mean(1), [.025, .975]))
            for key, value in actual.items():
                equal(value, statistics['comparisons'][metric + '_' + name][key], (metric, name, key))
                if metric == 'ce':
                    equal(value, summary['comparisons'][name][key], (name, key))
            if metric == 'ce':
                assert int((target > 0).sum()) == summary['comparisons'][name]['positive_targets']
            comparisons += 1
    assert comparisons == 6

    # The propagation side study retained the original lock through the
    # pre-fit cache repair; its identity is explicit in the amended probe lock.
    propagation = evidence('atlas_esmc_propagation_summary')
    done = read(folder / 'esmc_propagation/complete.json')
    assert done['complete'] and done['no_structure_scoring'] and done['full_forwards'] == 10
    assert done['lock_sha256'] == lock['reused_esmc_replay_parent_lock']
    assert propagation['complete'] and not propagation['structure_quality_scored']
    assert len(propagation['rows']) == 6
    assert {(r['target_id'], r['seed']) for r in propagation['rows']} == {tuple(v) for v in done['observations']}
    for name, expected in done['hashes'].items():
        path = folder / 'esmc_propagation' / name
        if path.suffix != '.cif':
            assert original_sha(path) == expected, name
    distance_errors = []
    for result in propagation['rows']:
        tid, seed = result['target_id'], result['seed']
        base = folder / 'esmc_propagation' / tid
        query_check, observer_check = read(base / 'query_replay.json'), read(base / 'observer_replay.json')
        assert query_check['passed'] and query_check['all22_sites_5passes_exact'] and query_check['coordinates_exact']
        assert observer_check['passed'] and observer_check['full_output_exact']
        row = read(base / f'native_esmc_s{seed}.json')
        assert row['lock_sha256'] == done['lock_sha256'] and row['target_id'] == tid and row['seed'] == seed
        assert len(row['sites']) == 22 and all(len(v) == 5 for v in row['sites'].values())
        for site, key in [('z', 'pair_relative_final'), ('s', 'single_relative_final')]:
            final = row['sites']['trunk_end.' + site][-1]
            equal(final['relative_delta_rms'], final['delta_rms'] / final['query_rms'], (tid, seed, site))
            equal(final['relative_delta_rms'], result[key], (tid, seed, key))
        qcoord = np.load(folder.parent / 'records' / tid / 'query_ca.npy', allow_pickle=False).astype(float)
        coord = np.load(base / f'native_esmc_s{seed}_ca.npy', allow_pickle=False).astype(float)
        assert coord.shape == qcoord.shape
        distances = lambda x: np.linalg.norm(x[:, None] - x[None], axis=-1)
        pairs = np.triu_indices(len(coord), k=1)
        distance = np.sqrt(np.mean((distances(coord) - distances(qcoord))[pairs] ** 2))
        equal(distance, result['distance_change'], (tid, seed, 'saved-coordinate distance'))
        equal(distance, row['structure_change']['ca_pair_distance_delta_rms'], (tid, seed, 'record distance'))
        distance_errors.append(abs(distance - result['distance_change']))
    return dict(passed=True, fits=9, independent_dev_targets=8, final_dev_records=72,
                contrasts=6, propagation_instances=6, numeric_checks=checks,
                maximum_absolute_error=max_error, maximum_coordinate_distance_error=max(distance_errors),
                scope='Saved probe scores/curves, receipt bindings and prediction-distance changes; no new fitting, cache extraction or structure-quality rescoring.')


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parents[1]), indent=2))
