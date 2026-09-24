"""Reconstruct the signed-permutation control and saved-checkpoint Dev8 curves.

This checks locked score arrays, their identities, provenance and the original
target-bootstrap streams.  It does not rerun training, prediction, CIF scoring,
or a full-model optimizer-trajectory equivalence test.  No new K statistic or
checkpoint-selection rule is introduced.
"""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np


METRICS = ('ca_lddt', 'residue_ca_lddt', 'tm_score_fixed_full_length')


def index_records(rows, fields, expected):
    """Reject silent overwrite, missing cells and unplanned successful subsets."""
    by = {tuple(row[k] for k in fields): row for row in rows}
    assert len(by) == len(rows), 'Duplicate record identity'
    assert set(by) == set(expected), 'Missing or unexpected record identity'
    assert all(row['status'] == 'ok' for row in rows), 'Non-success record'
    return by


def summarize(delta, draw):
    """Keep each whole target paired across transform and training-seed axes."""
    delta = np.asarray(delta, dtype=float)
    draw = np.asarray(draw)
    assert delta.ndim == 3 and np.isfinite(delta).all()
    assert draw.ndim == 2 and draw.shape[1] == delta.shape[2]
    assert np.issubdtype(draw.dtype, np.integer)
    assert draw.min() >= 0 and draw.max() < delta.shape[2]
    per_target = delta.mean((0, 1))
    return dict(mean=float(per_target.mean()), per_target=per_target,
                ci95=np.quantile(per_target[draw].mean(1), [.025, .975]),
                per_seed=delta.mean((0, 2)), per_rotation=delta.mean((1, 2)),
                positive_targets=int((per_target > 0).sum()))


def signed_transform_properties(specification):
    """Inspect the declared transformation, without assuming preservation of 1."""
    p = np.asarray(specification['permutation'])
    signs = np.asarray(specification['signs'])
    assert p.ndim == signs.ndim == 1 and p.shape == signs.shape
    assert np.issubdtype(p.dtype, np.integer)
    assert sorted(p.tolist()) == list(range(len(p))), 'Not a permutation'
    assert set(signs.tolist()) <= {-1, 1}, 'Non-unit sign'
    assert specification['convention'] == '(T x)[j] = signs[j] * x[permutation[j]]'
    # A signed permutation is orthogonal, but T1 equals the signs vector.
    matrix = np.eye(len(p), dtype=int)[p] * signs[:, None]
    assert np.array_equal(matrix @ matrix.T, np.eye(len(p)))
    return dict(dimension=len(p), negative_signs=int((signs < 0).sum()),
                preserves_all_ones=bool(np.array_equal(matrix @ np.ones(len(p)),
                                                      np.ones(len(p)))))


def verify(root):
    root = Path(root)
    ev = root / 'evidence'
    load = lambda name: json.loads((ev / (name + '.json')).read_text())
    provenance = root / 'notes/writing_branch_20260922/bundled_source_provenance.json'
    bindings = json.loads(provenance.read_text()) if provenance.exists() else {}

    def source_sha(name):
        actual = hashlib.sha256((ev / (name + '.json')).read_bytes()).hexdigest()
        key = 'evidence/' + name + '.json'
        if key in bindings:
            assert actual == bindings[key]['bundled_sha256'], key
            return bindings[key]['original_sha256']
        return actual

    checks, contrasts, max_error = 0, 0, 0.

    def equal(actual, expected, label):
        nonlocal checks, max_error
        actual, expected = np.asarray(actual), np.asarray(expected)
        assert actual.shape == expected.shape, (label, actual.shape, expected.shape)
        assert np.isfinite(actual).all() and np.isfinite(expected).all(), label
        error = float(np.max(np.abs(actual - expected)))
        assert error < 1e-12, (label, error)
        checks += 1
        max_error = max(max_error, error)

    def contrast(delta, obj, ids, draw, label):
        nonlocal contrasts
        if 'target_ids' in obj:
            assert obj['target_ids'] == ids, label
        result = summarize(delta, draw)
        for field, value in result.items():
            equal(value, obj[field], label + ':' + field)
        contrasts += 1

    # Signed permutation: a different, locked transformation family from dense R.
    a, rows = load('signed_summary'), load('signed_records')
    done, c = load('signed_complete'), load('signed_execution_lock')
    candidate = load('signed_candidate_lock')
    execution_sha = source_sha('signed_execution_lock')
    assert a['execution_lock_sha256'] == done['execution_lock_sha256'] == execution_sha
    assert done['analysis_sha256'] == source_sha('signed_summary')
    assert done['metric_records_sha256'] == source_sha('signed_records')
    assert c['candidate_lock_sha256'] == source_sha('signed_candidate_lock')
    assert done['complete'] and c['formal_released'] and c['observed_followup']
    assert a['new_failures'] == done['new_failures'] == 0
    assert a['n_new_predictions'] == done['n_new_predictions'] == 2592
    assert done['n_combined_records'] == len(rows) == 3600
    assert a['historical_lock'] == c['historical_execution_lock_sha256']
    assert c['train_size'] == c['train96_size'] == 96 and c['steps'] == 1536
    assert c['early_stop'] is False and c['formal_checkpoint_selection'] == 'fixed1536'
    assert c['optimizer'] == {'kind': 'AdamW', 'weight_decay': .01, 'clip_norm': 1}
    for field in ('formal_seeds', 'transform_seeds', 'transforms', 'new_runs',
                  'primary', 'secondary', 'bootstrap'):
        assert c[field] == candidate[field], field
    assert candidate['no_curve_dependent_configuration_selection']
    assert candidate['no_automatic_native_retraining']
    assert datetime.datetime.fromisoformat(candidate['created_utc']) < datetime.datetime.fromisoformat(c['released_utc'])
    assert c['bootstrap']['seed'] == 20260921 and c['bootstrap']['draws'] == 20000
    seeds, transforms = c['formal_seeds'], c['transform_seeds']
    assert len(seeds) == len(set(seeds)) == len(transforms) == len(set(transforms)) == 3
    new_names = {f'{kind}_t{t}_s{s}' for kind in ('factor', 'generic_plus')
                 for t in transforms for s in seeds}
    assert len(c['new_runs']) == 18
    assert {(r['name'], r['kind'], r['transform_seed'], r['seed'], r['steps'], r['rotation'])
            for r in c['new_runs']} == {
                (f'{kind}_t{t}_s{s}', kind, t, s, 1536, None)
                for kind in ('factor', 'generic_plus') for t in transforms for s in seeds}
    assert all(r['lr'] == 1e-4 for r in c['new_runs'])
    native_names = {f'{head}_s{s}' for head in ('native', 'gplus') for s in seeds}
    assert {r['name'] for r in c['native_checkpoints']} == native_names
    assert all(r['run']['steps'] == 1536 and r['complete']['failures'] == 0
               for r in c['native_checkpoints'])
    transform_properties = {}
    for tid in transforms:
        item = c['transforms'][str(tid)]
        payload = json.dumps(item['specification'], sort_keys=True, separators=(',', ':')).encode()
        assert hashlib.sha256(payload).hexdigest() == item['canonical_json_sha256']
        transform_properties[str(tid)] = signed_transform_properties(item['specification'])
    assert [v['negative_signs'] for v in transform_properties.values()] == [63, 63, 55]
    assert all(v['dimension'] == 128 and not v['preserves_all_ones']
               for v in transform_properties.values())

    # Historical Native/G+ and Query scores must really be the reused scores.
    history = load('openfold_train96_records')
    assert source_sha('openfold_train96_records') == c['historical_metrics_sha256']
    history_by = {(r['panel'], r['system'], r['target_id']): r for r in history}
    panel_ids = {panel: list(dict.fromkeys(r['target_id'] for r in history
                                         if r['panel'] == panel))
                 for panel in c['evaluation_panels']}
    expected = {(panel, name, tid) for panel, ids in panel_ids.items()
                for name in native_names | new_names | {'query'} for tid in ids}
    by = index_records(rows, ('panel', 'system', 'target_id'), expected)
    assert sum(r['system'] in new_names for r in rows) == 2592
    for key, row in by.items():
        if row['system'] not in new_names:
            assert row == history_by[key], key
    signed_contrasts_start = contrasts
    signed_primary = {}
    for panel, count in [('confirm96', 96), ('length48', 48)]:
        ids = panel_ids[panel]
        assert len(ids) == len(set(ids)) == count == c['evaluation_panels'][panel]
        draw = np.random.default_rng(c['bootstrap']['seed']).integers(count, size=(20000, count))
        for metric in METRICS:
            obj = a['panels'][panel][metric]
            get = lambda name: np.asarray([by[panel, name, i][metric] for i in ids])
            f = np.stack([get(f'native_s{s}') for s in seeds])
            g = np.stack([get(f'gplus_s{s}') for s in seeds])
            fr = np.stack([[get(f'factor_t{t}_s{s}') for s in seeds] for t in transforms])
            gr = np.stack([[get(f'generic_plus_t{t}_s{s}') for s in seeds] for t in transforms])
            df, dg = f[None] - fr, g[None] - gr
            for name, x in dict(factor_rotation=df, gplus_rotation=dg, interaction=df-dg).items():
                contrast(x, obj[name], ids, draw, f'signed/{panel}/{metric}/{name}')
            for name, x in dict(factor_native=f, factor_rotated=fr, gplus_native=g,
                                gplus_rotated=gr, query=get('query')).items():
                equal(x.mean(), obj['group_means'][name], f'signed/{panel}/{metric}/{name}')
            assert set(obj['system_means']) == native_names | new_names | {'query'}
            for name, value in obj['system_means'].items():
                equal(get(name).mean(), value, f'signed/{panel}/{metric}/{name}')
            if metric == 'ca_lddt':
                signed_primary[panel] = dict(
                    interaction_mean=obj['interaction']['mean'],
                    interaction_ci95=obj['interaction']['ci95'],
                    positive_seed_transform_cells=int(((df-dg).mean(2) > 0).sum()))
    signed_contrasts = contrasts - signed_contrasts_start
    assert signed_contrasts == 18

    # Saved checkpoints: the same eight already observed Dev targets throughout.
    a, rows = load('checkpoint_curves_summary'), load('checkpoint_curves_records')
    done, c = load('checkpoint_curves_complete'), load('checkpoint_curves_execution_lock')
    execution_sha = source_sha('checkpoint_curves_execution_lock')
    assert a['execution_lock_sha256'] == done['execution_lock_sha256'] == execution_sha
    assert done['analysis_sha256'] == source_sha('checkpoint_curves_summary')
    assert done['records_sha256'] == source_sha('checkpoint_curves_records')
    assert done['complete'] and done['no_new_training'] and done['no_checkpoint_selection']
    assert c['no_new_training'] and c['no_checkpoint_selection']
    assert a['exploratory'] and a['checkpoint_selection'] is False
    assert done['attempted'] == a['attempted'] == c['prediction_budget'] == len(rows) == 1160
    assert done['failures'] == done['missing'] == a['failures'] == a['missing'] == 0
    ids, seeds, rotations = c['target_ids'], c['formal_seeds'], c['rotation_seeds']
    assert a['targets'] == ids and len(ids) == len(set(ids)) == 8
    assert len(seeds) == len(set(seeds)) == len(rotations) == len(set(rotations)) == 3
    assert c['steps'] == [384, 768, 1536]
    assert c['bootstrap'] == dict(draws=20000, seed=20260924, unit='target',
                                 conditional_on_fitted_models=True, exploratory=True,
                                 multiplicity_adjusted=False)
    models = {(r['train_size'], r['run']['kind'], r['run']['seed'], r['run']['rotation']): r
              for r in c['models']}
    assert len(models) == len(c['models']) == 48
    assert len({r['id'] for r in models.values()}) == 48
    assert set(models) == {(n, kind, s, r) for n in (96, 384)
                           for kind in ('factor', 'generic_plus') for s in seeds
                           for r in [None] + rotations}
    for row in models.values():
        assert set(row['checkpoints']) == {'384', '768', '1536'}
        assert all(v['exists'] and v['step'] == int(k)
                   for k, v in row['checkpoints'].items())
    expected = {(row['id'], step, tid) for row in models.values()
                for step in c['steps'] for tid in ids} | {('query', 0, tid) for tid in ids}
    by = index_records(rows, ('model', 'step', 'target_id'), expected)
    assert all(r['train_size'] == (0 if r['model'] == 'query' else
               next(x['train_size'] for x in models.values() if x['id'] == r['model']))
               for r in rows)
    draw = np.random.default_rng(c['bootstrap']['seed']).integers(8, size=(20000, 8))
    curve_contrasts_start = contrasts
    curve_nodes = 0
    descriptive_changes = {}
    for metric in METRICS:
        for size in (96, 384):
            for step in c['steps']:
                obj = a['curves'][metric][str(size)][str(step)]
                def get(kind, seed, rotation):
                    name = models[size, kind, seed, rotation]['id']
                    return np.asarray([by[name, step, i][metric] for i in ids])
                f = np.stack([get('factor', s, None) for s in seeds])
                g = np.stack([get('generic_plus', s, None) for s in seeds])
                fr = np.stack([[get('factor', s, r) for s in seeds] for r in rotations])
                gr = np.stack([[get('generic_plus', s, r) for s in seeds] for r in rotations])
                df, dg = f[None] - fr, g[None] - gr
                for name, x in dict(factor_rotation=df, gplus_rotation=dg,
                                    interaction=df-dg, native_minus_gplus=(f-g)[None]).items():
                    contrast(x, obj[name], ids, draw, f'curves/{metric}/{size}/{step}/{name}')
                for name, x in dict(factor_native=f, factor_rotated=fr,
                                    gplus_native=g, gplus_rotated=gr).items():
                    equal(x.mean(), obj['means'][name], f'curves/{metric}/{size}/{step}/{name}')
                query = [by['query', 0, i][metric] for i in ids]
                equal(np.mean(query), obj['query_mean'], f'curves/{metric}/{size}/{step}/Query')
                curve_nodes += 1
            # Arithmetic descriptions only: no newly inferred interval, p, or K.
            if metric == 'ca_lddt':
                nodes = a['curves'][metric][str(size)]
                descriptive_changes[str(size)] = {
                    f'{start}_to_1536': {
                        'means': {head: nodes['1536']['means'][head] - nodes[str(start)]['means'][head]
                                  for head in nodes['1536']['means']},
                        'factor_rotation_change': nodes['1536']['factor_rotation']['mean'] - nodes[str(start)]['factor_rotation']['mean'],
                        'gplus_rotation_change': nodes['1536']['gplus_rotation']['mean'] - nodes[str(start)]['gplus_rotation']['mean']}
                    for start in (384, 768)}
    curve_contrasts = contrasts - curve_contrasts_start
    assert curve_nodes == 18 and curve_contrasts == 72
    return dict(passed=True, signed_records=3600, signed_new_predictions=2592,
                signed_historical_reused_predictions=1008, signed_verified_contrasts=signed_contrasts,
                signed_transform_properties=transform_properties,
                signed_pair_lddt=signed_primary, curve_records=1160, curve_shared_query_records=8,
                curve_metric_nodes=curve_nodes, curve_verified_contrasts=curve_contrasts,
                descriptive_pair_lddt_changes=descriptive_changes,
                scalar_or_array_checks=checks, max_abs_error=max_error,
                scope=('Locked score-array reconstruction and original pointwise target bootstraps; '
                       'no CIF scoring, new fit, full-model trajectory equivalence, convergence, '
                       'new K test or new-target confirmation. Signed transformations do not '
                       'preserve the all-ones direction. Dev8 nodes do not select checkpoints.'))


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parents[1]), indent=2))
