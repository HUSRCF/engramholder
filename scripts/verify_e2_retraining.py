"""Verify the same-seed E2 repeat using new scores and the fixed original baseline.

Runs are never pooled with the original execution or counted as extra seeds.
This audits archived CIF-score checks; it does not regenerate or rescore CIFs.
"""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

from verify_e2_intervention import conditional_summary, paired_effects


def verify(root):
    root = Path(root)
    load = lambda name: json.loads((root/'evidence'/f'{name}.json').read_text())
    provenance = root/'notes/writing_branch_20260922/bundled_source_provenance.json'
    bindings = json.loads(provenance.read_text()) if provenance.exists() else {}
    hashes = {}

    def source_sha(path):
        actual = hashlib.sha256((root/path).read_bytes()).hexdigest()
        if path in bindings:
            assert actual == bindings[path]['bundled_sha256'], path
            actual = bindings[path]['original_sha256']
        hashes[path] = actual
        return actual

    sha = lambda name: source_sha(f'evidence/{name}.json')
    lock, done = load('e2_retraining_lock'), load('e2_retraining_complete')
    summary, records = load('e2_retraining_summary'), load('e2_retraining_records')
    scored = load('e2_retraining_score_complete')
    cif, comparison = load('e2_retraining_cif_audit'), load('e2_retraining_historical_comparison')
    engineering = load('e2_retraining_engineering')
    parent = load('e1_execution_lock'); prediction = load('e1_prediction_lock')
    old = load('e2_intervention_summary')
    repeat_sha, execution_sha = sha('e2_retraining_lock'), sha('e1_execution_lock')
    assert lock['parent_execution_lock_sha256'] == execution_sha
    assert lock['parent_e2_release_sha256'] == sha('e2_intervention_execution_lock')
    assert lock['parent_e1_records_sha256'] == sha('e1_prediction_records')
    assert lock['parent_e2_analysis_sha256'] == sha('e2_intervention_summary')
    assert done['repetition_lock_sha256'] == engineering['repetition_lock_sha256'] == repeat_sha
    assert done['analysis_sha256'] == scored['analysis_sha256'] == sha('e2_retraining_summary')
    assert scored['records_sha256'] == sha('e2_retraining_records')
    assert done['comparison_sha256'] == sha('e2_retraining_historical_comparison')
    assert done['cif_audit_sha256'] == sha('e2_retraining_cif_audit')
    assert scored['execution_lock_sha256'] == summary['execution_lock_sha256'] == execution_sha
    assert summary['prediction_lock_sha256'] == sha('e1_prediction_lock')
    assert summary['scoring_amendment_sha256'] == sha('e1_scoring_amendment')
    folder = 'reproducibility/e2_retraining/'
    assert lock['protocol_sha256'] == source_sha(folder+'protocol.md')
    for name, expected in lock['files'].items():
        assert source_sha(folder+'archived/'+name) == expected
    assert done['complete'] and scored['complete'] and engineering['passed']
    assert done['fits'] == 9 and done['training_updates'] == 13824
    assert done['new_predictions'] == 864 and done['failures'] == summary['new_failures'] == 0
    assert lock['fresh_plm_features'] is False and lock['formal']['baseline'] == 'fixed_original_E1'
    assert engineering['engineering_updates'] == 384 and not engineering['trajectory_equivalence_claim']
    assert len(engineering['records']) == 6
    for row in engineering['records']:
        audit = row['resume_audit']
        assert all(audit[k] for k in ('state_exact', 'optimizer_exact', 'rng_exact', 'data_position_exact', 'lr_exact'))
        assert audit['forward_relative'] <= 1e-6 and audit['loss_abs'] == 0
    time = datetime.datetime.fromisoformat
    assert time(lock['created_utc']) < time(engineering['utc']) < time(summary['utc']) < time(done['utc'])
    ids, seeds = parent['confirm_ids'], parent['formal_seeds']
    rotations = [prediction['selected_low'], prediction['selected_high']]
    assert lock['directions'] == ['I', *rotations]
    assert lock['runs'] == load('e2_intervention_execution_lock')['runs']
    new_by = {(r['system'], r['target_id']): r for r in records}
    assert len(new_by) == len(records) == 864
    expected = {(run['name'], tid) for run in lock['runs'] for tid in ids}
    assert set(new_by) == expected and all(r['status'] == 'ok' for r in records)
    for run in lock['runs']:
        path = root/folder/'runs'/run['name']
        receipt = {name: json.loads((path/name).read_text()) for name in
                   ('repetition.json', 'training_complete.json', 'run.json', 'complete.json')}
        for name in receipt:
            source_sha(str((path/name).relative_to(root)))
        assert receipt['repetition.json']['repetition_lock_sha256'] == repeat_sha
        assert receipt['training_complete.json']['complete']
        assert receipt['training_complete.json']['steps'] == 1536
        assert receipt['run.json']['run'] == run and receipt['complete.json']['run'] == run
        assert receipt['complete.json']['n_predictions'] == 96 and receipt['complete.json']['failures'] == 0
        assert receipt['complete.json']['no_evaluation_labels_parsed']
        assert [r['target_id'] for r in receipt['complete.json']['records']] == ids
        for rec in receipt['complete.json']['records']:
            assert rec['prediction_sha256'] == new_by[run['name'], rec['target_id']]['prediction_sha256']
    assert len(cif['records']) == 864
    assert {(r['system'], r['target_id']) for r in cif['records']} == expected
    assert all(r['masked_coordinates_fp32_exact'] for r in cif['records'])
    assert cif['scoring_maxabs'] == dict.fromkeys(('ca_lddt', 'residue_ca_lddt', 'tm_score_fixed_full_length'), 0.)
    fixed = {(r['system'], r['target_id']): r for r in load('e1_prediction_records')}
    by = {**fixed, **new_by}; assert not set(fixed) & set(new_by)
    draw = np.random.default_rng(20260924).integers(96, size=(20000, 96))
    comparisons, checks, max_error = 0, 0, 0.

    def equal(actual, expected, label):
        nonlocal checks, max_error
        a, b = np.asarray(actual), np.asarray(expected)
        assert a.shape == b.shape and np.isfinite(a).all(), label
        error = float(np.max(np.abs(a-b)))
        assert error < 1e-12, (label, error)
        max_error = max(max_error, error); checks += 1

    def contrast(array, reported, label):
        nonlocal comparisons
        for field, value in conditional_summary(array, draw).items():
            equal(value, reported[field], label+'/'+field)
        comparisons += 1

    for metric in ('ca_lddt', 'residue_ca_lddt', 'tm_score_fixed_full_length'):
        get = lambda model: np.asarray([by[model, tid][metric] for tid in ids])
        n = np.stack([get(f'native_s{s}') for s in seeds])
        nc = np.stack([get(f'I_C_s{s}') for s in seeds])
        r = np.stack([[get(f'{r}_s{s}') for s in seeds] for r in rotations])
        rc = np.stack([[get(f'{r}_C_s{s}') for s in seeds] for r in rotations])
        eff = paired_effects(n, nc, r, rc); md = summary['metrics'][metric]
        equal(eff['T_C'], eff['B_R']-eff['native_change'], metric+'/paired identity')
        for key, val in [('native_fixed', n.mean()), ('native_learned', nc.mean()), ('query_mean', get('query').mean())]:
            equal(val, md[key], metric+'/'+key)
        equal(md['native_fixed'], old['metrics'][metric]['native_fixed'], metric+'/unchanged Native baseline')
        for key in ('native_change', 'B_R', 'T_C'):
            contrast(eff[key], md[key], metric+'/'+key)
            historical = comparison[metric][key]
            equal(md[key]['mean'], historical['repeat'], metric+'/repeat/'+key)
            equal(md[key]['ci95'], historical['repeat_ci95'], metric+'/repeat interval/'+key)
            equal(old['metrics'][metric][key]['mean'], historical['original'], metric+'/original/'+key)
            equal(historical['repeat']-historical['original'], historical['difference'], metric+'/descriptive change/'+key)
        for j, rid in enumerate(rotations):
            arm = md['per_rotation'][rid]
            equal(r[j].mean(), arm['fixed_mean'], metric+'/'+rid+'/fixed')
            equal(rc[j].mean(), arm['learned_mean'], metric+'/'+rid+'/learned')
            equal(arm['fixed_mean'], old['metrics'][metric]['per_rotation'][rid]['fixed_mean'], metric+'/'+rid+'/unchanged baseline')
            for key in ('B_R', 'T_C'):
                contrast(eff['rotation_'+key][j], arm[key], metric+'/'+rid+'/'+key)
    assert comparisons == 21
    return dict(passed=True, records=864, fits=9, updates=13824, failures=0,
                verified_contrasts=comparisons, scalar_or_array_checks=checks,
                max_abs_error=max_error, source_sha256=hashes,
                archived_CIF_audit_consistent=True, independent_CIF_rescoring=False,
                same_seeds_not_pooled=True, fixed_baseline='original E1',
                engineering_conditions=6, engineering_updates=384,
                fresh_PLM_feature_replay=False, training_divergence_cause='not identified')


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parents[1]), indent=2))
