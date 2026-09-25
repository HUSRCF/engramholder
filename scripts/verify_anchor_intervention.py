"""Reconstruct the 42 sealed anchor-intervention contrasts from saved scores.

This verifies pairing, original/bundled provenance and archived audit consistency.
It does not rerun training, reference trajectories, coordinate scoring or TMscore.
"""
import datetime
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from verify_e2_intervention import conditional_summary


def paired_effects(native_live, native_q, rotated_live, rotated_q):
    """L/Q arrays are [seed,target] or [rotation,seed,target], respectively."""
    nl, nq, rl, rq = [np.asarray(x, dtype=float) for x in
                      (native_live, native_q, rotated_live, rotated_q)]
    if (nl.ndim != 2 or nl.shape != nq.shape or not all(nl.shape)
            or rl.ndim != 3 or rl.shape != rq.shape or not rl.shape[0]
            or rl.shape[1:] != nl.shape):
        raise ValueError('Expected paired [seed,target] and [rotation,seed,target] arrays')
    if not all(np.isfinite(x).all() for x in (nl, nq, rl, rq)):
        raise ValueError('Scores must be finite')
    native_change = nq - nl
    live_gap, q_gap = nl[None] - rl, nq[None] - rq
    benefit = rl - rq
    effect = q_gap - live_gap
    return dict(T_A=effect.mean(0), B_R_live_minus_Q=benefit.mean(0),
                native_Q_minus_live=native_change, rotation_T_A=effect,
                rotation_B_R_live_minus_Q=benefit,
                live_gap=live_gap, Q_gap=q_gap)


def verify(root):
    root = Path(root)
    load = lambda name: json.loads((root/'evidence'/f'{name}.json').read_text())
    provenance = root/'notes/writing_branch_20260922/bundled_source_provenance.json'
    bindings = json.loads(provenance.read_text()) if provenance.exists() else {}
    source_checks = {}

    def original_sha(path):
        actual = hashlib.sha256((root/path).read_bytes()).hexdigest()
        original = actual
        if path in bindings:
            assert actual == bindings[path]['bundled_sha256'], path
            original = bindings[path]['original_sha256']
        source_checks[path] = original
        return original

    def evidence_sha(name):
        return original_sha(f'evidence/{name}.json')

    summary = load('e3_anchor_summary')
    records = load('e3_anchor_records')
    new_records = load('e3_anchor_new_records')
    lock = load('e3_anchor_execution_lock')
    done = load('e3_anchor_complete')
    runtime = load('e3_anchor_runtime_audit')
    readback = load('e3_anchor_statistical_readback')
    accepted = load('e3_anchor_acceptance')
    refs = load('e3_anchor_references')
    amendment = load('e3_anchor_engineering_amendment')
    parent = load('e1_execution_lock')
    prediction = load('e1_prediction_lock')
    execution_sha = evidence_sha('e3_anchor_execution_lock')
    assert lock['parent_execution_lock_sha256'] == evidence_sha('e1_execution_lock')
    for obj in (summary, done, runtime, accepted, refs, amendment):
        assert obj['execution_lock_sha256'] == execution_sha
    assert done['analysis_sha256'] == evidence_sha('e3_anchor_summary')
    assert done['records_sha256'] == evidence_sha('e3_anchor_new_records')
    assert done['combined_sha256'] == evidence_sha('e3_anchor_records')
    assert readback['input_sha256'] == {
        'analysis.json': evidence_sha('e3_anchor_summary'),
        'combined_metric_records.json': evidence_sha('e3_anchor_records')}
    for name, source in [('prediction_lock.json', 'e1_prediction_lock'),
                         ('e1/analysis/complete.json', 'e1_prediction_complete'),
                         ('e2/analysis/complete.json', 'e2_intervention_complete')]:
        assert lock['parent_receipts'][name] == evidence_sha(source)
    folder = 'reproducibility/anchor_intervention/'
    assert original_sha(folder+'protocol.md') == lock['protocol_sha256']
    for name, expected in lock['files'].items():
        assert original_sha(folder+'archived/'+name) == expected
    receipt_paths = {
        'runtime_audit.json': 'evidence/e3_anchor_runtime_audit.json',
        'statistical_readback.json': 'evidence/e3_anchor_statistical_readback.json',
        'per_target.csv': folder+'per_target.csv', 'acceptance.md': folder+'acceptance.md'}
    assert set(accepted['files']) == set(receipt_paths)
    for name, path in receipt_paths.items():
        assert original_sha(path) == accepted['files'][name]
    assert amendment['original_smoke_sha256'] == lock['files']['smoke.py']
    assert amendment['diagnosis_sha256'] == evidence_sha('e3_anchor_resume_diagnosis')
    # Read the original diagnostic failure as a boundary, not a recovered pass.
    assert 'failed' in amendment['change'] and 'No scientific/training source change' in amendment['change']
    assert accepted['accepted'] and accepted['engineering_and_statistics_passed']
    assert accepted['scientific_hypothesis_supported'] is False
    assert done['complete'] and summary['stage'] == 'E3'
    assert done['new_predictions'] == lock['formal_predictions'] == 864
    assert lock['training_updates'] == 13824
    assert done['new_failures'] == summary['new_failures'] == 0
    time = datetime.datetime.fromisoformat
    assert (time(prediction['utc']) < time(lock['utc']) < time(refs['utc'])
            < time(done['utc']) < time(accepted['accepted_at']))

    ids, seeds = parent['confirm_ids'], parent['formal_seeds']
    rotations = [prediction['selected_low'], prediction['selected_high']]
    assert summary['targets'] == ids and len(ids) == len(set(ids)) == 96
    assert len(seeds) == len(set(seeds)) == 3 and lock['selected_rotations'] == rotations
    assert len(set(rotations)) == 2
    expected_runs = {(f'{r}_Q_s{s}', r, s, 1536, 'factor_query_anchor')
                     for r in ['I', *rotations] for s in seeds}
    assert len(lock['runs']) == len(expected_runs) == 9
    assert {(r['name'], r['rotation_id'], r['seed'], r['steps'], r['kind'])
            for r in lock['runs']} == expected_runs
    assert summary['runs'] == lock['runs']
    new_by = {(r['system'], r['target_id']): r for r in new_records}
    by = {(r['system'], r['target_id']): r for r in records}
    assert len(new_by) == len(new_records) == 864
    assert set(new_by) == {(r[0], t) for r in expected_runs for t in ids}
    assert len(by) == len(records) == 1824
    old_by = {(r['system'], r['target_id']): r for r in load('e1_prediction_records')}
    baseline_names = {'query'} | {f'native_s{s}' for s in seeds}
    baseline_names |= {f'{r}_s{s}' for r in rotations for s in seeds}
    baseline = {k: v for k, v in old_by.items() if k[0] in baseline_names}
    assert len(baseline) == 960 and not set(baseline) & set(new_by)
    assert by == {**baseline, **new_by}, 'Reused baselines and every new record must be exact'
    assert all(r['status'] == 'ok' for r in records)
    assert all(re.fullmatch('[0-9a-f]{64}', r['prediction_sha256']) for r in new_records)
    assert all(abs(r['ca_lddt']-r['reference_check_same_input_dtypes']) < 1e-6
               for r in new_records)

    assert refs['complete'] and refs['unique_caches'] == len(refs['records']) == 192
    assert refs['reference_forward_trajectories'] == 384
    reference_ids = {(r['phase'], r['target_id']) for r in refs['records']}
    assert len(reference_ids) == 192
    train_ids = {tid for phase, tid in reference_ids if phase == 'train'}
    eval_ids = {tid for phase, tid in reference_ids if phase == 'eval'}
    assert len(train_ids) == 96 and eval_ids == set(ids) and not train_ids & eval_ids
    for r in refs['records']:
        assert r['execution_lock_sha256'] == execution_sha
        assert r['two_seed_exact'] and r['rng_preserved']
        assert r['gradient_path'] == (r['phase'] == 'train')
    for flag in ['passed', 'all_checkpoint_and_prediction_hashes_verified',
                 'all_initializations_reconstructed_exactly', 'all_runs_without_C',
                 'optimizer_and_schedule_verified', 'scoring_after_all_predictions',
                 'score_function_AST_equal_to_accepted_parent']:
        assert runtime[flag]
    assert runtime['training_runs'] == 9 and runtime['training_updates'] == 13824
    assert runtime['predictions_new'] == 864 and runtime['reference_replays_verified'] == 27
    assert runtime['raw_primary_scores_checked'] == runtime['tm_output_records_parsed'] == 1824
    assert runtime['raw_primary_maxabs'] == runtime['tm_record_maxabs'] == 0
    assert len(runtime['runs']) == 9
    assert {r['run'] for r in runtime['runs']} == {r[0] for r in expected_runs}
    assert all(r['steps'] == 1536 and r['predictions'] == 96 and r['without_C']
               and r['initialization_exact'] and r['optimizer_steps_exact']
               and r['reference_replays'] == 3 for r in runtime['runs'])
    assert readback['passed'] and readback['verified_contrasts'] == 42
    assert parent['bootstrap'] == {'seed': 20260924, 'draws': 20000}
    draws = np.random.default_rng(20260924).integers(96, size=(20000, 96))
    checks, comparisons, max_error = 0, 0, 0.

    def equal(actual, expected, label):
        nonlocal checks, max_error
        actual, expected = np.asarray(actual), np.asarray(expected)
        assert actual.shape == expected.shape and np.isfinite(actual).all(), label
        error = float(np.max(np.abs(actual-expected)))
        assert error < 1e-12, (label, error)
        max_error = max(max_error, error)
        checks += 1

    def check_contrast(array, reported, audited, label):
        nonlocal comparisons
        value = conditional_summary(array, draws)
        value['positive_targets'] = int((array.mean(0) > 0).sum())
        for key in ['mean', 'ci95', 'per_target', 'per_seed', 'positive_targets']:
            equal(value[key], reported[key], label+'/'+key)
            equal(value[key], audited[key], 'audit/'+label+'/'+key)
        comparisons += 1

    for metric in ['ca_lddt', 'residue_ca_lddt', 'tm_score_fixed_full_length']:
        get = lambda name: np.asarray([by[name, t][metric] for t in ids])
        nl = np.stack([get(f'native_s{s}') for s in seeds])
        nq = np.stack([get(f'I_Q_s{s}') for s in seeds])
        rl = np.stack([[get(f'{r}_s{s}') for s in seeds] for r in rotations])
        rq = np.stack([[get(f'{r}_Q_s{s}') for s in seeds] for r in rotations])
        q = get('query')
        effects = paired_effects(nl, nq, rl, rq)
        equal(effects['rotation_T_A'], effects['rotation_B_R_live_minus_Q']
              + effects['native_Q_minus_live'][None], metric+'/paired identity')
        effects['native_Q_minus_query'] = nq-q
        m, a = summary['metrics'][metric], readback['metrics'][metric]
        for key in ['T_A', 'B_R_live_minus_Q', 'native_Q_minus_live', 'native_Q_minus_query']:
            check_contrast(effects[key], m[key], a[key], metric+'/'+key)
        absolute = dict(query=q.mean(), native_live=nl.mean(), native_Q=nq.mean(),
                        rotated_live=rl.mean(), rotated_Q=rq.mean(),
                        live_native_minus_rotated=effects['live_gap'].mean(),
                        Q_native_minus_rotated=effects['Q_gap'].mean())
        for key, value in absolute.items():
            equal(value, a['absolute'][key], metric+'/absolute/'+key)
            if key in m:
                equal(value, m[key], metric+'/'+key)
        for j, r in enumerate(rotations):
            arm, audited = m['per_rotation'][r], a['per_rotation'][r]
            for key, arr in [('T_A', effects['rotation_T_A'][j]),
                             ('B_R_live_minus_Q', effects['rotation_B_R_live_minus_Q'][j]),
                             ('live_gap', effects['live_gap'][j]),
                             ('Q_gap', effects['Q_gap'][j]), ('rotated_Q_minus_query', rq[j]-q)]:
                check_contrast(arr, arm[key], audited[key], metric+'/'+r+'/'+key)
            for key, value in [('rotated_live', rl[j].mean()), ('rotated_Q', rq[j].mean())]:
                equal(value, arm[key], metric+'/'+r+'/'+key)
                equal(value, audited[key], 'audit/'+metric+'/'+r+'/'+key)
    assert comparisons == 42
    return dict(passed=True, records=1824, new_records=864, reused_records=960,
                targets=96, verified_contrasts=comparisons, scalar_or_array_checks=checks,
                max_abs_error=max_error, source_sha256=source_checks,
                paired_seeds=seeds, selected_rotations=rotations,
                bootstrap={'seed': 20260924, 'draws': 20000},
                protocol_and_implementation_identities_verified=True,
                archived_runtime_audit_consistent=True,
                original_failed_trajectory_check_preserved=True,
                new_coordinate_scoring=False, new_training=False,
                scope='Prespecified Factor-only anchor intervention on observed Confirm96-B; '
                      'conditional, unadjusted target intervals. No C, new-target confirmation, '
                      'between-rotation heterogeneity test or equivalence test.')


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parents[1]), indent=2))
