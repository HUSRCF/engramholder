"""Reconstruct two completed follow-ups from saved model/target score grids.

No production analysis is imported. This checks scores, identities and receipt
bindings; it does not rerun folding, checkpoint recovery or CIF scoring.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

SEEDS = [20260923, 20260924, 20260925]
ROTATIONS = [20261001, 20261002, 20261003]
METRICS = ['ca_lddt', 'residue_ca_lddt', 'tm_score_fixed_full_length']
ARMS = ['factor_native', 'factor_rotated', 'gplus_native', 'gplus_rotated']


def verify(root):
    root = Path(root)
    folder = root / 'reproducibility/budget_schedule_followups'
    read = lambda p: json.loads(p.read_text())
    evidence = lambda name: read(root / 'evidence' / (name + '.json'))
    provenance = root / 'notes/writing_branch_20260922/bundled_source_provenance.json'
    bindings = read(provenance) if provenance.exists() else {}

    def sha(path):
        key = str(path.relative_to(root))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if key in bindings:
            assert digest == bindings[key]['bundled_sha256'], key
            return bindings[key]['original_sha256']
        return digest

    for name, entry in read(folder / 'import_manifest.json').items():
        assert sha(root / name) == entry['sha256'], name

    comparisons, field_checks, maximum_error = 0, 0, 0.0

    def equal(actual, expected, label):
        nonlocal field_checks, maximum_error
        a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), label
        error = float(np.max(np.abs(a - b)))
        assert error < 1e-12, (label, error)
        maximum_error = max(maximum_error, error)
        field_checks += 1

    def check(array, expected, draws, label):
        nonlocal comparisons
        array = np.asarray(array, dtype=float)
        target = array.mean(tuple(range(array.ndim - 1)))
        computed = dict(mean=target.mean(), per_target=target,
                        ci95=np.quantile(target[draws].mean(1), [.025, .975]))
        if array.ndim == 2:
            computed['per_seed'] = array.mean(1)
        else:
            assert array.ndim == 3
            computed.update(per_seed=array.mean((0, 2)), per_rotation=array.mean((1, 2)),
                            rotation_by_seed=array.mean(2))
        for key, value in computed.items():
            if key in expected:
                equal(value, expected[key], (label, key))
        if 'positive_targets' in expected:
            assert int((target > 0).sum()) == expected['positive_targets'], label
        comparisons += 1

    def contrast(c):
        df = c['factor_native'] - c['factor_rotated']
        dg = c['gplus_native'] - c['gplus_rotated']
        return dict(factor_rotation=df, gplus_rotation=dg, interaction=df-dg,
                    native_minus_gplus=c['factor_native']-c['gplus_native'])

    # OpenFold: the retained 36 fits are a single-rotation reporting amendment,
    # not completion of the originally locked three-rotation primary matrix.
    of = evidence('of_schedule_summary')
    rows = evidence('of_schedule_records')
    done = evidence('of_schedule_complete')
    amendment = evidence('of_schedule_reporting_amendment')
    lock = evidence('of_schedule_execution_lock')
    calibration = evidence('of_schedule_calibration')
    assert done['complete'] and of['n_predictions'] == len(rows) == 3552 and of['failures'] == 0
    for suffix, field in [('summary', 'analysis_sha256'), ('records', 'metric_records_sha256'),
                          ('reporting_amendment', 'reporting_amendment_sha256')]:
        assert sha(root / f'evidence/of_schedule_{suffix}.json') == done[field]
    assert of['execution_lock_sha256'] == sha(root / 'evidence/of_schedule_execution_lock.json')
    assert of['reporting_amendment_sha256'] == done['reporting_amendment_sha256']
    assert amendment['full_matrix_primary_unavailable'] and amendment['rotation'] == 20261001
    assert amendment['bootstrap'] == dict(unit='target', repeats=20000, seed=20260926, paired=True, stratified=False)
    assert of['seeds'] == SEEDS and amendment['targets'] == of['target_ids']
    ids = of['target_ids']
    assert len(ids) == len(set(ids)) == 96
    formal = {r['name'] for r in lock['formal'] if r['wave'] == 1}
    assert len(formal) == 36
    expected = {(s, t) for s in formal | {'query'} for t in ids}
    by = {(r['system'], r['target_id']): r for r in rows}
    assert len(by) == len(rows) and set(by) == expected
    assert all(r['status'] == 'ok' for r in rows)
    assert calibration['execution_lock_sha256'] == of['execution_lock_sha256']
    chosen = min(calibration['scores'], key=lambda k: (-calibration['scores'][k], float(k)))
    assert float(chosen) == calibration['learning_rate'] == 5e-5
    draws = np.random.default_rng(20260926).integers(96, size=(20000, 96))
    of_begin = comparisons
    for metric in METRICS:
        q = np.array([by['query', t][metric] for t in ids])
        m, curves = of['statistics'][metric], {}
        check(np.tile(q, (3, 1)), m['query'], draws, (metric, 'query'))
        for schedule in ['first', 'all', 'last']:
            c = {}
            for arm, kind, rotation in [('factor_native', 'factor', 'native'),
                                        ('factor_rotated', 'factor', 'r20261001'),
                                        ('gplus_native', 'generic_plus', 'native'),
                                        ('gplus_rotated', 'generic_plus', 'r20261001')]:
                c[arm] = np.array([[by[f'{schedule}_{kind}_{rotation}_s{s}', t][metric]
                                    for t in ids] for s in SEEDS])
                check(c[arm], m['conditions'][schedule]['absolute'][arm], draws, (metric, schedule, arm))
                check(c[arm]-q, m['conditions'][schedule]['gain_over_query'][arm], draws, (metric, schedule, arm, 'gain'))
            d = contrast(c)
            curves[schedule] = d
            for name, values in d.items():
                field = 'native_factor_minus_gplus' if name == 'native_minus_gplus' else name
                check(values, m['conditions'][schedule][field], draws, (metric, schedule, name))
        for name, left, right in [('D_write_R1', 'all', 'first'), ('D_timing_R1', 'first', 'last'), ('D_last_R1', 'all', 'last')]:
            for quantity, field in [('interaction', 'interaction'), ('factor_rotation', 'factor_contribution'), ('gplus_rotation', 'gplus_contribution')]:
                check(curves[left][quantity]-curves[right][quantity], m['differences'][name][field], draws, (metric, name, field))
    of_comparisons = comparisons - of_begin

    # Protenix: continuation and reference-aggregation recovery preserve the
    # original model identities, panel membership and all saved score records.
    pt, rows = evidence('pt_budget_summary'), evidence('pt_budget_records')
    lock, done = evidence('pt_budget_execution_lock'), evidence('pt_budget_complete')
    recovery = evidence('pt_budget_recovery_amendment')
    assert pt['observed_followup'] and pt['failures'] == 0 and len(rows) == 9984
    assert done['complete'] and pt['execution_lock_sha256'] == done['execution_lock_sha256'] == sha(root / 'evidence/pt_budget_execution_lock.json')
    assert pt['recovery_amendment_sha256'] == sha(root / 'evidence/pt_budget_recovery_amendment.json')
    for suffix, ending in [('summary', '/analysis.json'), ('records', '/metric_records.json')]:
        expected_sha = [v for k, v in done['files'].items() if k.endswith(ending)]
        assert expected_sha == [sha(root / f'evidence/pt_budget_{suffix}.json')]
    assert recovery['original_metric_records_sha256'] == sha(root / 'evidence/pt_budget_records.json')
    assert not recovery['changes_to_models_or_predictions'] and not recovery['changes_to_metric_or_bootstrap']
    receipt = read(folder / 'pt_budget/recovery_results/recovery_complete.json')
    assert receipt['complete'] and receipt['all_9984_metric_records_identical']
    assert receipt['analysis_sha256'] == sha(root / 'evidence/pt_budget_summary.json')
    manifest = evidence('pt_budget_manifest')['targets']
    ids = [r['target_id'] for r in manifest]
    assert len(ids) == len(set(ids)) == 192
    assert ids == [t['target_id'] for t in evidence('protenix_fresh192_inference_manifest')['targets']]
    dev = read(folder / 'pt_budget/recovery_results/preflight.json')['actual_dev_targets']
    assert len(dev) == len(set(dev)) == 8 and not set(ids) & set(dev)
    models = set(lock['parents'])
    assert len(models) == 24 and all(x['step'] == 1536 for x in lock['parents'].values())
    expected = {('fresh', n, s, t) for n in [1536, 3072] for s in models for t in ids}
    expected |= {('fresh', 1536, 'query', t) for t in ids}
    expected |= {('dev', n, s, t) for n in [1536, 2304, 3072] for s in models for t in dev}
    by = {(r['panel'], r['node'], r['system'], r['target_id']): r for r in rows}
    assert len(by) == len(rows) and set(by) == expected and all(r['status'] == 'ok' for r in rows)
    assert Counter(r['panel'] for r in rows) == dict(fresh=9408, dev=576)
    historical = evidence('protenix_fresh192_records')
    assert len(historical) == 4800
    for row in historical:
        replay = by['fresh', 1536, row['system'], row['target_id']]
        for metric in METRICS:
            assert row[metric] == replay[metric], (row['system'], row['target_id'], metric)
    rng = np.random.default_rng(20260926)
    pieces = []
    for label in ['128-191', '192-255', '256-319', '320-384']:
        index = np.array([i for i, t in enumerate(manifest) if t['length_bin'] == label])
        assert len(index) == 48
        pieces.append(index[rng.integers(48, size=(20000, 48))])
    draws = np.concatenate(pieces, axis=1)

    def cells(panel, node, targets, metric):
        result = {}
        for kind, name in [('factor', 'factor'), ('generic_plus', 'gplus')]:
            for suffix, rotations in [('native', ['native']), ('rotated', [f'r{r}' for r in ROTATIONS])]:
                result[name+'_'+suffix] = np.array([[[by[panel, node, f'C_{kind}_{r}_s{s}', t][metric]
                    for t in targets] for s in SEEDS] for r in rotations])
        return result

    pt_begin = comparisons
    for metric in METRICS:
        q = np.array([by['fresh', 1536, 'query', t][metric] for t in ids])
        cs, ds = {}, {}
        for n in [1536, 3072]:
            m = pt['metrics'][metric][str(n)]
            cs[n] = cells('fresh', n, ids, metric)
            ds[n] = contrast(cs[n])
            for name, array in cs[n].items():
                check(array, m['cells'][name], draws, (metric, n, name))
            for name, array in {**ds[n], **{k+'_minus_query': v-q for k, v in cs[n].items()}}.items():
                check(array, m['contrasts'][name], draws, (metric, n, name))
            equal(q.mean(), m['query_mean'], (metric, n, 'query'))
            for model, value in m['system_means'].items():
                equal(np.mean([by['fresh', n, model, t][metric] for t in ids]), value, (metric, n, model))
        m = pt['metrics'][metric]['budget_change']
        for name, field in [('interaction', 'K'), ('factor_rotation', 'factor_rotation_change'), ('gplus_rotation', 'gplus_rotation_change')]:
            check(ds[3072][name]-ds[1536][name], m[field], draws, (metric, field))
        for name in ARMS:
            check(cs[3072][name]-cs[1536][name], m['cell_gains'][name], draws, (metric, name, 'gain'))
        for n in [1536, 2304, 3072]:
            c = cells('dev', n, dev, metric)
            m = pt['dev_curves'][metric][str(n)]
            for name, array in c.items():
                equal(array.mean(), m['cell_means'][name], (metric, n, 'dev', name))
            for name, array in contrast(c).items():
                check(array, m['contrasts'][name], np.arange(8)[None], (metric, n, 'dev', name))
    return dict(passed=True, openfold_records=3552, openfold_summaries=of_comparisons,
                protenix_records=9984, protenix_summaries=comparisons-pt_begin,
                exact_historical_replay_rows=4800, numeric_field_checks=field_checks,
                maximum_absolute_error=maximum_error,
                scope='Saved score grids, paired bootstrap, marginals, source/receipt bindings; no GPU training or independent CIF rescoring.')


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parents[1]), indent=2))
