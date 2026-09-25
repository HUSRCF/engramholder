"""Verify bounded Atlas propagation summaries and Native-only calibration.

Rebuilds aggregation from recorded same-site statistics, geometry from saved
coordinates, and calibration contrasts from scores. It does not recompute
hidden-state tensors, regenerate structures, rescore CIFs, or audit training.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


def verify(root):
    root = Path(root)
    load = lambda name: json.loads((root / 'evidence' / f'{name}.json').read_text())
    folder = 'reproducibility/atlas_followup/'
    provenance = root / 'notes/writing_branch_20260922/bundled_source_provenance.json'
    bindings = json.loads(provenance.read_text()) if provenance.exists() else {}
    hashes = {}

    def original_sha(path):
        actual = hashlib.sha256((root / path).read_bytes()).hexdigest()
        if path in bindings:
            assert actual == bindings[path]['bundled_sha256'], path
            actual = bindings[path]['original_sha256']
        hashes[path] = actual
        return actual

    def sha(name):
        return original_sha(f'evidence/{name}.json')

    checks, max_error = 0, 0.0

    def equal(actual, expected, label):
        nonlocal checks, max_error
        a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), label
        error = float(np.max(np.abs(a - b)))
        assert error < 1e-12, (label, error)
        checks += 1
        max_error = max(max_error, error)

    lock = load('atlas_propagation_execution_lock')
    summary = load('atlas_propagation_summary')
    lock_sha = sha('atlas_propagation_execution_lock')
    assert summary['lock_sha256'] == lock_sha
    assert summary['records'] == 20 and summary['full_forwards'] == 26
    assert lock['new_training'] == 0 and len(lock['targets']) == 2
    assert len(lock['runs']) == 10
    assert {r['branch'] for r in lock['runs']} == {'query', 'parent', 'C_only', 'head_only'}
    for local, suffix in [
        ('propagation_protocol.md', '/atlas_propagation_localization_20260926/PROTOCOL.md'),
        ('archived/atlas_propagation_localization.py', '/src/engramfold/experiments/atlas_propagation_localization.py'),
        ('archived/atlas_posttraining_score.py', '/src/engramfold/experiments/atlas_posttraining_calibration/score.py'),
    ]:
        expected = {v for k, v in lock['files'].items() if k.endswith(suffix)}
        assert original_sha(folder + local) in expected, local

    observations = []
    for target in lock['targets']:
        tid, length = target['target_id'], len(target['sequence'])
        path = root / folder / 'records' / tid
        done = json.loads((path / 'complete.json').read_text())
        assert done['complete'] and done['systems'] == 10 and done['forward_count'] == 13
        assert done['lock_sha256'] == lock_sha and done['frozen_hash'] == lock['frozen_hash']
        assert done['no_structure_scoring']
        for local in path.iterdir():
            if local.name in done['files']:
                assert original_sha(str(local.relative_to(root))) == done['files'][local.name]
        query = json.loads((path / 'query.json').read_text())
        qcoord = np.load(path / 'query_ca.npy', allow_pickle=False).astype(float)
        assert qcoord.shape == (length, 3) and np.isfinite(qcoord).all()
        qdist = np.linalg.norm(qcoord[:, None] - qcoord[None], axis=-1)
        pairs = np.triu_indices(length, k=1)
        smoke = json.loads((path / 'query_smoke.json').read_text())
        assert smoke['passed'] and smoke['repeated_states_exact'] and smoke['observer_vs_plain_output_exact']
        adapted_smoke = json.loads((path / 'adapted_smoke.json').read_text())
        assert adapted_smoke['passed'] and adapted_smoke['observer_vs_plain_output_exact']
        for run in lock['runs']:
            row = json.loads((path / (run['name'] + '.json')).read_text())
            assert row['run'] == run and row['target_id'] == tid and row['length'] == length
            assert row['lock_sha256'] == lock_sha and row['all_passes'] == 5
            assert row['no_structure_labels'] and len(row['sites']) == 22
            assert set(row['sites']) == set(query['sites'])
            for site, rounds in row['sites'].items():
                assert len(rounds) == 5
                for index, values in enumerate(rounds):
                    equal(values['query_rms'], query['sites'][site][index]['query_rms'], (tid, site, index))
                    if values['query_rms']:
                        equal(values['relative_delta_rms'], values['delta_rms'] / values['query_rms'], site)
            coord = np.load(path / (run['name'] + '_ca.npy'), allow_pickle=False).astype(float)
            assert coord.shape == qcoord.shape and np.isfinite(coord).all()
            dist = np.linalg.norm(coord[:, None] - coord[None], axis=-1)
            equal(np.sqrt(np.mean((dist - qdist)[pairs] ** 2)),
                  row['structure_change']['ca_pair_distance_delta_rms'], (tid, run['name'], 'distance'))
            observations.append(row)

    # Rebuild medians/ranges over the same two chains, seeds and five passes.
    # These are descriptive observations, not independent target replicates.
    for branch, streams in summary['summary'].items():
        rows = [r for r in observations if r['run']['branch'] == branch]
        assert len(rows) == 6
        for stream, sites in streams.items():
            for site, statistics in sites.items():
                values = [v for r in rows for v in r['sites'][site + '.' + stream]]
                assert len(values) == 30
                for statistic, expected in statistics.items():
                    values_for_stat = [v[statistic] for v in values if v[statistic] is not None]
                    if not values_for_stat:
                        assert expected is None
                        continue
                    for name, reducer in [('median', np.median), ('min', np.min), ('max', np.max)]:
                        equal(reducer(values_for_stat), expected[name], (branch, site, statistic, name))

    calibration = load('atlas_posttraining_summary')
    records = load('atlas_posttraining_records')
    complete = load('atlas_posttraining_complete')
    cal_lock_sha = sha('atlas_posttraining_execution_lock')
    assert complete['complete'] and complete['n_predictions'] == 960
    assert complete['execution_lock_sha256'] == calibration['execution_lock_sha256'] == cal_lock_sha
    assert complete['analysis_sha256'] == sha('atlas_posttraining_summary')
    assert complete['records_sha256'] == sha('atlas_posttraining_records')
    assert lock['prior_receipt_sha256'] == sha('atlas_posttraining_complete')
    assert calibration['primary'] == 'ca_lddt C_only_minus_head_only'
    ids, seeds = calibration['target_ids'], calibration['seeds']
    assert len(ids) == len(set(ids)) == 96 and seeds == [20260923, 20260924, 20260925]
    systems = {'atlas_query'} | {f'atlas_I_s{s}_{b}' for s in seeds for b in ['parent', 'C_only', 'head_only']}
    by = {(r['system'], r['target_id']): r for r in records}
    assert len(records) == len(by) == 960 and set(by) == {(s, t) for s in systems for t in ids}
    assert all(r['status'] == 'ok' for r in records)
    assert all(v == 0 for v in calibration['failure_counts'].values())
    assert calibration['bootstrap']['draws'] == 20000 and calibration['bootstrap']['seed'] == 20260925
    draws = np.random.default_rng(20260925).integers(0, 96, size=(20000, 96))
    contrast_checks = 0
    for metric, expected in calibration['metrics'].items():
        values = {b: np.array([[by[f'atlas_I_s{s}_{b}', t][metric] for t in ids] for s in seeds])
                  for b in ['parent', 'C_only', 'head_only']}
        values['query'] = np.array([by['atlas_query', t][metric] for t in ids])
        for branch, array in values.items():
            equal(array.mean(), expected['absolute_means'][branch], (metric, branch, 'absolute'))
            if branch != 'query':
                equal(array.mean(1), expected['per_seed_absolute_means'][branch], (metric, branch, 'seeds'))
        for name, reported in expected['contrasts'].items():
            left, right = name.split('_minus_')
            diff = values[left] - values[right]
            target = diff.mean(0)
            equal(target, reported['per_target'], (metric, name, 'target'))
            equal(target.mean(), reported['mean'], (metric, name, 'mean'))
            equal(diff.mean(1), reported['per_seed_mean'], (metric, name, 'seed'))
            equal(np.quantile(target[draws].mean(1), [.025, .975]), reported['ci95'], (metric, name, 'ci'))
            assert int((target > 0).sum()) == reported['positive_targets']
            assert reported['conditional_on_fitted_models']
            contrast_checks += 1
        assert expected['main'] == expected['contrasts']['C_only_minus_head_only']
    assert contrast_checks == 18
    return dict(passed=True, observation_instances=len(observations), independent_targets=2,
                calibration_predictions=len(records), calibration_contrasts=contrast_checks,
                numeric_checks=checks, maximum_absolute_error=max_error, source_hashes=hashes,
                scope='Recorded state-summary aggregation, saved-coordinate distances and score-level bootstrap; no hidden-tensor or CIF rescoring, no retraining.')


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parents[1]), indent=2))
