"""Reconstruct A66 and Protenix Train96 statistics from immutable score arrays.

No training, inference, CIF scoring, endpoint changes or multiplicity correction.
The bootstrap matches each study, including Length48's fixed length strata.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


def verify(root):
    root = Path(root)
    ev = root / 'evidence'
    load = lambda name: json.loads((ev / (name + '.json')).read_text())
    provenance = root / 'notes/writing_branch_20260922/bundled_source_provenance.json'
    bindings = json.loads(provenance.read_text()) if provenance.exists() else {}

    def sha(name):
        actual = hashlib.sha256((ev / (name + '.json')).read_bytes()).hexdigest()
        key = 'evidence/' + name + '.json'
        if key in bindings:
            assert actual == bindings[key]['bundled_sha256']
            return bindings[key]['original_sha256']
        return actual
    a = load('dh_A66_summary')
    rows = load('dh_A66_records')
    sl = load('dh_A66_scoring_lock')
    assert a['scoring_lock_sha256'] == sha('dh_A66_scoring_lock')
    assert sl['science_execution_lock_sha256'] == sha('dh_A66_execution_lock')
    assert sl['multiple_testing_adjusted'] is False
    assert a['primary'] == 'per backbone Confirm96 C-final Factor minus trained Rotated'
    assert a['new_predictions'] == 9504 and a['new_failures'] == 0
    assert len(rows) == 14112
    assert len({(x['backbone'], x['panel'], x['system'], x['target_id']) for x in rows}) == len(rows)
    assert all(x['status'] == 'ok' for x in rows)
    assert sum(x['origin'] == 'new_CIF_scoring' for x in rows) == 9504
    done = load('dh_A66_completion')
    assert done['complete'] and done['records_sha256'] == sha('dh_A66_records')
    assert done['analysis_sha256'] == sha('dh_A66_summary')
    assert load('dh_A66_source_verification')['complete']
    assert load('dh_A66_deployment_audit')['complete']
    seeds, rotations = a['fixed_seeds'], a['fixed_rotations']
    assert len(seeds) == len(rotations) == 3
    metrics = ['ca_lddt', 'residue_ca_lddt', 'tm_score_fixed_full_length']
    checks, contrasts, max_error = 0, 0, 0.

    def equal(actual, expected, label):
        nonlocal checks, max_error
        actual, expected = np.asarray(actual), np.asarray(expected)
        assert actual.shape == expected.shape, (label, actual.shape, expected.shape)
        assert np.isfinite(actual).all() and np.isfinite(expected).all(), label
        error = float(np.max(np.abs(actual - expected)))
        assert error < 1e-12, (label, error)
        max_error = max(max_error, error)
        checks += 1

    def check_contrast(delta, obj, ids, draw, label):
        nonlocal contrasts
        # Independent seed x rotation x target convention.
        target = delta.mean((0, 1))
        if 'target_ids' in obj:
            assert obj['target_ids'] == ids, label
        equal(target, obj['per_target'], label + ':targets')
        equal(target.mean(), obj['mean'], label + ':mean')
        equal(np.quantile(target[draw].mean(1), [.025, .975]), obj['ci95'], label + ':CI')
        equal(delta.mean((1, 2)), obj['per_seed'], label + ':seed')
        equal(delta.mean((0, 2)), obj['per_rotation'], label + ':rotation')
        if 'rotation_by_seed' in obj:
            equal(delta.mean(2).T, obj['rotation_by_seed'], label + ':cells')
        assert int((target > 0).sum()) == obj['positive_targets'], label
        contrasts += 1

    length_targets = {t['target_id']: t for t in load('length48_manifest')['targets']}
    expected_ids = load('data_composition_audit')['datasets']['Confirm96-B']['records']
    expected_ids = {x['target_id'] for x in expected_ids}
    for backbone, panels in a['panels'].items():
        for panel, md in panels.items():
            ids = md['ca_lddt']['cells']['C']['factor_rotation']['target_ids']
            assert len(ids) == len(set(ids)) == (96 if panel == 'confirm96' else 48)
            assert set(ids) == (expected_ids if panel == 'confirm96' else set(length_targets))
            rng = np.random.default_rng(sl['bootstrap_seed'])
            if panel == 'length48':
                groups = [[i for i, t in enumerate(ids) if length_targets[t]['length_bin'] == b]
                          for b in ['385-512', '513-640', '641-768']]
                assert [len(g) for g in groups] == [16, 16, 16]
                draw = np.concatenate([np.asarray(g)[rng.integers(len(g), size=(20000, len(g)))]
                                       for g in groups], axis=1)
            else:
                draw = rng.integers(96, size=(20000, 96))
            subset = [x for x in rows if x['backbone'] == backbone and x['panel'] == panel]
            assert len(subset) == 49 * len(ids)
            for metric in metrics:
                by = {(x['system'], x['target_id']): x[metric] for x in subset}
                get = lambda model: np.array([by[model, t] for t in ids])
                arrays = {}
                for feature in ['E', 'C']:
                    f = np.stack([get(f'{feature}_factor_native_s{s}') for s in seeds])
                    g = np.stack([get(f'{feature}_generic_plus_native_s{s}') for s in seeds])
                    fr = np.stack([[get(f'{feature}_factor_r{r}_s{s}') for r in rotations] for s in seeds])
                    gr = np.stack([[get(f'{feature}_generic_plus_r{r}_s{s}') for r in rotations] for s in seeds])
                    q = get('query')
                    df, dg = f[:, None] - fr, g[:, None] - gr
                    arrays[feature] = (f, g, df, dg, df - dg)
                    cell = md[metric]['cells'][feature]
                    for name, x in dict(query=q, native=f, rotated_factor=fr, gplus=g, rotated_gplus=gr).items():
                        equal(x.mean(), cell['means'][name], f'{backbone}/{panel}/{metric}/{feature}/{name}')
                    for name, x in dict(factor_rotation=df, gplus_rotation=dg, interaction=df-dg,
                                        native_minus_gplus=(f-g)[:, None], native_minus_query=(f-q)[:, None],
                                        gplus_minus_query=(g-q)[:, None]).items():
                        check_contrast(x, cell[name], ids, draw, f'{backbone}/{panel}/{metric}/{feature}/{name}')
                ef, eg, edf, edg, ep = arrays['E']
                cf, cg, cdf, cdg, cp = arrays['C']
                for name, x in dict(direction_change=cdf-edf, gplus_direction_change=cdg-edg,
                                    psi_change=cp-ep, native_gain=(cf-ef)[:, None],
                                    gplus_gain=(cg-eg)[:, None],
                                    native_gplus_gap_change=((cf-cg)-(ef-eg))[:, None]).items():
                    check_contrast(x, md[metric]['plm_interactions'][name], ids, draw,
                                   f'{backbone}/{panel}/{metric}/{name}')
    a66_contrasts = contrasts
    p = load('pt96_fourcell_summary')
    pr = load('pt96_fourcell_records')
    pl = load('pt96_fourcell_scoring_lock')
    assert p['execution_lock_sha256'] == sha('pt96_fourcell_execution_lock')
    assert p['scoring_prelock_sha256'] == sha('pt96_fourcell_scoring_lock')
    assert p['new_attempted'] == 1152 and p['failures'] == 0
    assert len(pr) == len({(x['system'], x['target_id']) for x in pr}) == 2400
    assert all(x['status'] == 'ok' for x in pr)
    pdone = load('pt96_fourcell_completion')
    assert pdone['complete'] and pdone['artifacts']['analysis.json'] == sha('pt96_fourcell_summary')
    assert pdone['artifacts']['metric_records.json'] == sha('pt96_fourcell_records')
    ids = p['target_ids']
    assert len(ids) == len(set(ids)) == 96 and set(ids) == expected_ids
    draw = np.random.default_rng(pl['bootstrap_seed']).integers(96, size=(pl['bootstrap_replicates'], 96))
    for metric, md in p['metrics'].items():
        by = {(x['system'], x['target_id']): x[metric] for x in pr}
        get = lambda name: np.array([by[name, t] for t in ids])
        f = np.stack([get(f'factor_native_s{s}') for s in seeds])
        g = np.stack([get(f'gplus_native_s{s}') for s in seeds])
        fr = np.stack([[get(f'factor_r{r}_s{s}') for r in rotations] for s in seeds])
        gr = np.stack([[get(f'gplus_r{r}_s{s}') for r in rotations] for s in seeds])
        q = get('query')
        df, dg = f[:, None]-fr, g[:, None]-gr
        for name, x in dict(query=q, native=f, rotated_factor=fr, gplus=g, rotated_gplus=gr).items():
            equal(x.mean(), md['means'][name], 'pt96/'+metric+'/'+name)
        for name, x in dict(factor_rotation=df, gplus_rotation=dg, interaction=df-dg,
                            native_minus_gplus=(f-g)[:, None], native_minus_query=(f-q)[:, None],
                            gplus_minus_query=(g-q)[:, None]).items():
            check_contrast(x, md[name], ids, draw, 'pt96/'+metric+'/'+name)
        for name, value in md['system_means'].items():
            equal(get(name).mean(), value, 'pt96/system/'+metric+'/'+name)
    return dict(passed=True, a66_records=len(rows), train96_records=len(pr),
                a66_contrasts=a66_contrasts, train96_contrasts=contrasts-a66_contrasts,
                scalar_or_array_checks=checks, max_abs_error=max_error,
                scope='Score-array/target-bootstrap reconstruction; no inference or CIF rescoring; no multiplicity correction.')


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parents[1]), indent=2))
