"""Reassemble Fresh192 from saved scores and the frozen target/model manifests.

No CIF rescoring, GPU prediction, new hypothesis or historical-result overwrite.
The bootstrap unit is a target within a prespecified stratum, not a fitted model.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

SEEDS = (20260923, 20260924, 20260925)
ROTATIONS = (20261001, 20261002, 20261003)
BINS = ('128-191', '192-255', '256-319', '320-384')
METRICS = ('ca_lddt', 'residue_ca_lddt', 'tm_score_fixed_full_length')


def system_names():
    return ['query'] + [f'C_{head}_{r}_s{s}' for head in ('factor', 'generic_plus')
                       for s in SEEDS for r in ('native', *[f'r{x}' for x in ROTATIONS])]


def score_grid(records, ids):
    by = {(r['system'], r['target_id']): r for r in records}
    expected = {(s, t) for s in system_names() for t in ids}
    if len(ids) != len(set(ids)) or len(by) != len(records) or set(by) != expected:
        raise ValueError('Incomplete, duplicate or unexpected model-target grid')
    for row in records:
        if row['status'] not in ('ok', 'failed'):
            raise ValueError('Nonterminal prediction')
        values = np.asarray([row[m] for m in METRICS])
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError('Invalid structure score')
        if row['status'] == 'failed' and np.any(values != 0):
            raise ValueError('Locked failures must retain zero scores and denominator')
    return by


def stratified_draws(targets, draws=20000, seed=20260925):
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero([t['length_bin'] == b for t in targets]) for b in BINS]
    if [len(g) for g in groups] != [48] * 4 or len(targets) != 192:
        raise ValueError('Expected 48 targets in each of the four locked strata')
    return np.concatenate([rng.choice(g, size=(draws, 48)) for g in groups], axis=1)


def effects(n, r, g, gr, q):
    """Use [seed,target] native and [rotation,seed,target] rotated arrays."""
    df, dg = n[None] - r, g[None] - gr
    return dict(factor_rotation=df, gplus_rotation=dg, interaction=df-dg,
                native_minus_gplus=(n-g)[None], native_minus_query=(n-q)[None],
                gplus_minus_query=(g-q)[None], rotated_factor_minus_query=r-q,
                rotated_gplus_minus_query=gr-q)


def verify(root):
    root = Path(root)
    load = lambda name: json.loads((root/'evidence'/f'protenix_fresh192_{name}.json').read_text())
    provenance = root/'notes/writing_branch_20260922/bundled_source_provenance.json'
    bindings = json.loads(provenance.read_text()) if provenance.exists() else {}
    checked_sources = {}

    def original_sha(name):
        actual = hashlib.sha256((root/name).read_bytes()).hexdigest()
        if name in bindings:
            assert actual == bindings[name]['bundled_sha256'], name
            actual = bindings[name]['original_sha256']
        checked_sources[name] = actual
        return actual

    def sha(name):
        return original_sha(f'evidence/protenix_fresh192_{name}.json')

    result, records = load('summary'), load('records')
    lock, models, done = load('execution_lock'), load('model_lock'), load('complete')
    score_lock, prediction = load('score_lock'), load('prediction_completion')
    ref, inf = load('reference_manifest'), load('inference_manifest')
    selection, exposure, accepted = load('selection_audit'), load('exposure'), load('acceptance')
    ex_sha = sha('execution_lock')
    for obj in (result, done, score_lock, prediction, load('preflight'), load('formal_start')):
        assert obj['execution_lock_sha256'] == ex_sha
    assert done['analysis_sha256'] == accepted['analysis_sha256'] == sha('summary')
    assert done['records_sha256'] == sha('records')
    mapping = {'model_lock.json': 'model_lock', 'scoring_exposure_audit.json': 'exposure',
               'selection_audit.json': 'selection_audit', 'panel/selection_lock.json': 'selection_lock',
               'panel/inference_manifest.json': 'inference_manifest',
               'panel/reference_manifest.json': 'reference_manifest', 'engineering/complete.json': 'engineering'}
    for suffix, name in mapping.items():
        matches = [v for k, v in lock['files'].items() if k.endswith('/'+suffix)]
        assert matches == [sha(name)], suffix
    folder = 'reproducibility/protenix_fresh192/'
    protocol = [v for k, v in lock['files'].items()
                if k.endswith('/protenix_fresh192_20260925/protocol.md')]
    assert protocol == [original_sha(folder+'protocol.md')]
    for p in sorted((root/folder/'archived').glob('*.py')):
        expected = [v for k, v in lock['files'].items()
                    if k.endswith('/protenix_fresh_fourcell/'+p.name)]
        assert expected == [original_sha(str(p.relative_to(root)))], p.name
    assert ref['selection_lock_sha256'] == selection['selection_lock_sha256'] == sha('selection_lock')
    assert lock['inference_manifest_sha256'] == sha('inference_manifest')
    assert (lock['created_time'] <= load('preflight')['time'] < load('formal_start')['started']
            < prediction['completed_time'] < done['completed_time'])
    assert result['scoring_started_after_prediction_completion']
    assert lock['primary'] == result['primary'] == 'ca_lddt interaction Psi'
    assert lock['bootstrap'] == {'draws': 20000, 'seed': 20260925, 'strata': list(BINS)}
    assert lock['selected_from_observed_recipe'] and result['conditional_on_fixed_models']
    assert lock['models'] == 25 and lock['targets'] == 192 and lock['predictions'] == 4800
    assert lock['new_training'] == models['new_training'] == result['new_training'] == 0
    assert done['complete'] and prediction['complete'] and accepted['passed']
    assert result['failures'] == 0 and len(records) == 4800

    ids = result['targets']
    targets = ref['targets']
    assert len(ids) == len(set(ids)) == ref['target_count'] == 192
    assert [t['target_id'] for t in targets] == [t['target_id'] for t in inf['targets']] == ids
    assert set(models['systems']) == set(system_names()) and len(models['systems']) == 25
    assert set(models['adapters']) == set(system_names()) - {'query'}
    for name, m in models['adapters'].items():
        assert m['step'] == 1536 and m['train_size'] == 384 and m['depth'] == 508.5
        assert m['seed'] in SEEDS and m['rotation_seed'] in (None, *ROTATIONS)
        arm = 'native' if m['rotation_seed'] is None else f"r{m['rotation_seed']}"
        assert name == f"C_{m['kind']}_{arm}_s{m['seed']}"
        assert m['trained_provenance']['input_dim'] == 1152
        assert m['trained_provenance']['layer'] == 36

    by = score_grid(records, ids)
    assert all(r['status'] == 'ok' for r in records)
    predicted = {(r['system'], r['target_id']): r for r in prediction['records']}
    assert len(predicted) == len(prediction['records']) == 4800 and set(predicted) == set(by)
    lengths = {t['target_id']: t['sequence_length'] for t in targets}
    for key, row in by.items():
        p = predicted[key]; length = lengths[key[1]]
        assert row['prediction_sha256'] == p['prediction_sha256']
        assert p['status'] == 'ok' and p['full_length'] == length
        assert p['injection_shapes'] == [[length, length, 128]] * 4
    assert min(lengths.values()) == 128 and max(lengths.values()) == 383
    assert selection['passed'] and selection['known_id_pdb_sequence_overlap'] == 0
    assert selection['no_predicted_scores_used'] and exposure['all_scans_parsed']
    assert not exposure['complete_historical_inventory_certified'] and not ref['family_isolation']
    known_ids = set(exposure['target_ids'])
    assert set(ids).isdisjoint(known_ids)
    assert {t.split('_')[0].lower() for t in ids}.isdisjoint({t.split('_')[0].lower() for t in known_ids})
    assert {t['sequence'] for t in targets}.isdisjoint(exposure['query_sequences'])
    assert len({t['pdb_id'].lower() for t in targets}) == 192
    for t, seq in zip(targets, inf['targets'], strict=True):
        assert set(seq) == {'target_id', 'sequence', 'sequence_sha256', 'sequence_length', 'length_bin'}
        assert all(t[k] == v for k, v in seq.items())
        assert len(t['sequence']) == t['sequence_length']
        assert hashlib.sha256(t['sequence'].encode()).hexdigest() == t['sequence_sha256']
        lo, hi = map(int, t['length_bin'].split('-'))
        assert lo <= t['sequence_length'] <= hi
        assert t['initial_release_date'] <= '2021-09-30' and t['model_count'] == 1
        assert t['resolution_high_angstrom'] <= 2.5 and t['reference_ca_coverage'] >= .9
    draw = stratified_draws(targets)
    checks, comparisons, max_error = 0, 0, 0.

    def equal(got, expected, label):
        nonlocal checks, max_error
        a, b = np.asarray(got), np.asarray(expected)
        assert a.shape == b.shape and np.isfinite(a).all(), label
        error = float(np.max(np.abs(a-b)))
        assert error < 1e-12, (label, error)
        max_error = max(max_error, error); checks += 1

    for metric in METRICS:
        vec = lambda system: np.asarray([by[system, t][metric] for t in ids])
        n = np.stack([vec(f'C_factor_native_s{s}') for s in SEEDS])
        g = np.stack([vec(f'C_generic_plus_native_s{s}') for s in SEEDS])
        r = np.stack([[vec(f'C_factor_r{r}_s{s}') for s in SEEDS] for r in ROTATIONS])
        gr = np.stack([[vec(f'C_generic_plus_r{r}_s{s}') for s in SEEDS] for r in ROTATIONS])
        q = vec('query'); md = result['metrics'][metric]
        for system in system_names():
            equal(vec(system).mean(), md['system_means'][system], metric+'/'+system)
        for key, a in dict(query=q, factor_native=n, factor_rotated=r, gplus_native=g, gplus_rotated=gr).items():
            equal(a.mean(), md['four_cell_means'][key], metric+'/'+key)
        contrasts = effects(n, r, g, gr, q)
        for key, a in contrasts.items():
            v = a.mean((0, 1)); report = md[key]
            computed = dict(mean=v.mean(), ci95=np.quantile(v[draw].mean(1), [.025, .975]),
                            per_target=v, positive_targets=int((v > 0).sum()),
                            per_rotation=a.mean((1, 2)), per_seed=a.mean((0, 2)),
                            rotation_by_seed=a.mean(2))
            for field, value in computed.items():
                equal(value, report[field], metric+'/'+key+'/'+field)
            comparisons += 1
        # Assemble Psi a second way: complete within-target means, then contrast.
        equal(n.mean(0)-r.mean((0, 1))-g.mean(0)+gr.mean((0, 1)),
              md['interaction']['per_target'], metric+'/direct four-cell identity')
        v = contrasts['interaction'].mean((0, 1))
        for name in BINS:
            ix = np.flatnonzero([t['length_bin'] == name for t in targets])
            sample = np.random.default_rng(20260925).choice(ix, size=(20000, 48))
            secondary = md['length_strata_secondary'][name]
            assert secondary['n'] == 48
            equal(v[ix].mean(), secondary['mean'], metric+'/'+name+'/mean')
            equal(np.quantile(v[sample].mean(1), [.025, .975]), secondary['ci95_unadjusted'], metric+'/'+name+'/ci')
    csv_rows = list(csv.DictReader((root/folder/'per_target.csv').open()))
    assert [r['target_id'] for r in csv_rows] == ids
    assert [r['length_bin'] for r in csv_rows] == [t['length_bin'] for t in targets]
    for col, name in [('psi', 'interaction'), ('factor_rotation', 'factor_rotation'), ('gplus_rotation', 'gplus_rotation')]:
        equal([float(r[col]) for r in csv_rows], result['metrics']['ca_lddt'][name]['per_target'], 'CSV/'+col)
    return dict(passed=True, records=4800, targets=192, systems=25, failures=0,
                verified_contrasts=comparisons, supplementary_stratum_intervals=12,
                scalar_or_array_checks=checks, max_abs_error=max_error,
                bootstrap=dict(draws=20000, seed=20260925, unit='target within four fixed strata'),
                source_sha256=checked_sources, chronology_and_manifest_checks=True,
                independent_CIF_rescoring=False, new_training=False,
                scope='Fixed-model, new-target primary Psi; other contrasts unadjusted secondary. '
                      'Historical recipe selection disclosed; no retraining or pretraining-isolation claim.')


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parents[1]), indent=2))
