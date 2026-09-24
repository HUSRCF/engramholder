"""Reconstruct the sealed E2 intervention from paired structure-score arrays.

The same Native / Native+C models are shared by both selected rotations. Targets,
not model--target records, are bootstrapped after averaging paired seeds and
rotations. This verifies the 21 original contrasts; it adds neither a remaining-gap
test nor an equivalence claim, and does not rerun training or structure scoring.
"""
import ast
import datetime
import hashlib
import json
import re
from pathlib import Path

import numpy as np


def paired_effects(native, native_c, rotated, rotated_c):
    """Return seed-by-target effects with one common Native control per seed."""
    native, native_c, rotated, rotated_c = [
        np.asarray(x, dtype=float) for x in (native, native_c, rotated, rotated_c)]
    if (native.ndim != 2 or native.shape != native_c.shape
            or rotated.ndim != 3 or rotated.shape != rotated_c.shape
            or rotated.shape[1:] != native.shape or not all(native.shape)
            or not rotated.shape[0]):
        raise ValueError('Expected matched [seed,target] and [rotation,seed,target] arrays')
    if not all(np.isfinite(x).all() for x in (native, native_c, rotated, rotated_c)):
        raise ValueError('Scores must be finite')
    native_change = native_c - native
    rotated_change = rotated_c - rotated
    gap_reduction = (native[None] - rotated) - (native_c[None] - rotated_c)
    return dict(native_change=native_change, B_R=rotated_change.mean(0),
                T_C=gap_reduction.mean(0), rotation_B_R=rotated_change,
                rotation_T_C=gap_reduction)


def conditional_summary(delta, draws):
    """Use synchronized whole-target draws, retaining the seed average inside."""
    delta, draws = np.asarray(delta, dtype=float), np.asarray(draws)
    if delta.ndim != 2 or not all(delta.shape) or not np.isfinite(delta).all():
        raise ValueError('Effects must be finite [seed,target] arrays')
    if (draws.ndim != 2 or not np.issubdtype(draws.dtype, np.integer)
            or draws.shape[1] != delta.shape[1] or not len(draws)
            or draws.min() < 0 or draws.max() >= delta.shape[1]):
        raise ValueError('Bootstrap draws must resample whole target records')
    target = delta.mean(0)
    return dict(mean=float(target.mean()), per_target=target.tolist(),
                per_seed=delta.mean(1).tolist(),
                ci95=np.quantile(target[draws].mean(1), [.025, .975]).tolist())


def verify(root):
    root = Path(root)
    ev = root / 'evidence'
    load = lambda name: json.loads((ev / (name + '.json')).read_text())
    provenance = root / 'notes/writing_branch_20260922/bundled_source_provenance.json'
    bindings = json.loads(provenance.read_text()) if provenance.exists() else {}
    source_checks = {}

    def source_sha(name):
        actual = hashlib.sha256((ev / (name + '.json')).read_bytes()).hexdigest()
        key = 'evidence/' + name + '.json'
        original = actual
        if key in bindings:
            assert actual == bindings[key]['bundled_sha256'], key
            original = bindings[key]['original_sha256']
        source_checks[name] = original
        return original

    summary = load('e2_intervention_summary')
    records = load('e2_intervention_records')
    done = load('e2_intervention_complete')
    release = load('e2_intervention_execution_lock')
    runtime = load('e2_intervention_runtime_audit')
    review = load('e2_intervention_review_audit')
    execution = load('e1_execution_lock')
    prediction = load('e1_prediction_lock')
    old_done = load('e1_prediction_complete')
    old_records = load('e1_prediction_records')
    execution_sha = source_sha('e1_execution_lock')
    for record in (summary, done, release, runtime, prediction, old_done):
        assert record['execution_lock_sha256'] == execution_sha
    prediction_sha = source_sha('e1_prediction_lock')
    assert summary['prediction_lock_sha256'] == release['prediction_lock_sha256'] == prediction_sha
    assert release['e1_completion_sha256'] == source_sha('e1_prediction_complete')
    assert old_done['records_sha256'] == source_sha('e1_prediction_records')
    assert old_done['analysis_sha256'] == source_sha('e1_prediction_summary')
    assert done['analysis_sha256'] == source_sha('e2_intervention_summary')
    assert done['records_sha256'] == source_sha('e2_intervention_records')
    assert (done['scoring_amendment_sha256'] == summary['scoring_amendment_sha256']
            == old_done['scoring_amendment_sha256'] == source_sha('e1_scoring_amendment'))
    # The mean over the two preselected rotations predates the outcomes. Inspect
    # the hashed original source as data, never execute a bundled analysis script.
    prespecified = load('e2_intervention_prespecified_analysis')
    source_sha('e2_intervention_prespecified_analysis')
    sealed_analysis_sha = hashlib.sha256(prespecified['text'].encode()).hexdigest()
    assert (sealed_analysis_sha == prespecified['original_sha256']
            == execution['staging_hashes']['analyze.py']
            == load('e1_scoring_amendment')['original_analyze_sha256'])
    design = load('e2_intervention_prespecified_design')
    source_sha('e2_intervention_prespecified_design')
    assert (hashlib.sha256(design['text'].encode()).hexdigest()
            == design['original_sha256'] == execution['design_sha256'])
    assert design['source_commit'].startswith(execution['design_commit'])
    assert r'T_C=\tfrac12\sum' in design['text'] and r'B_R=\tfrac12\sum' in design['text']
    predefined_means = False
    for node in ast.walk(ast.parse(prespecified['text'])):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == 'dict'):
            continue
        keywords = {entry.arg: ast.unparse(entry.value) for entry in node.keywords}
        if (keywords.get('T_C') == 'summary(tc.mean(0))'
                and keywords.get('B_R') == 'summary(br.mean(0))'
                and keywords.get('native_change') == 'summary(Nc - N)'
                and 'per_rotation' in keywords):
            predefined_means = True
    assert predefined_means, 'The sealed implementation must define pooled and individual effects'
    assert summary['stage'] == done['stage'] == 'e2' and done['complete']
    assert old_done['stage'] == 'e1' and old_done['complete']
    assert done['new_predictions'] == 864
    assert done['new_failures'] == done['comparison_failures'] == 0
    assert summary['new_failures'] == summary['failures'] == 0
    assert release['released'] and prediction['released']
    assert execution['E2_independent_authorized']
    assert release['independent_intervention_authorized']
    assert release['outcome_positive_not_required']
    time = datetime.datetime.fromisoformat
    assert (time(execution['time_utc']) < time(prediction['utc'])
            < time(release['utc']) < time(summary['utc']))

    ids, seeds = execution['confirm_ids'], execution['formal_seeds']
    rotations = [prediction['selected_low'], prediction['selected_high']]
    assert len(ids) == len(set(ids)) == 96
    assert len(seeds) == len(set(seeds)) == 3
    assert len(set(rotations)) == 2
    order = sorted(range(8), key=lambda i: (prediction['X'][i], prediction['rotation_ids'][i]))
    assert rotations == [prediction['rotation_ids'][order[0]], prediction['rotation_ids'][order[-1]]]
    expected_runs = {(f'{r}_C_s{s}', r, s, 1536, 'factor_plus_shared_C')
                     for r in ['I', *rotations] for s in seeds}
    assert len(release['runs']) == 9
    assert {(r['name'], r['rotation_id'], r['seed'], r['steps'], r['kind'])
            for r in release['runs']} == expected_runs
    by = {(r['system'], r['target_id']): r for r in records}
    assert len(by) == len(records) == 864
    assert set(by) == {(r[0], target) for r in expected_runs for target in ids}
    old_by = {(r['system'], r['target_id']): r for r in old_records}
    assert len(old_by) == len(old_records) == 2688
    old_systems = {f'{r}_s{s}' for r in execution['rotation_ids'] for s in seeds}
    old_systems |= {f'native_s{s}' for s in seeds} | {'query'}
    assert set(old_by) == {(name, target) for name in old_systems for target in ids}
    assert not set(by) & set(old_by)
    by.update(old_by)
    assert len(by) == 3552 and all(r['status'] == 'ok' for r in by.values())
    assert all(abs(r['ca_lddt']-r['reference_check_same_input_dtypes']) < 1e-6 for r in records)
    assert all(re.fullmatch('[0-9a-f]{64}', r['prediction_sha256']) for r in records)

    # These are the sealed engineering audits, not a repeat of the diagnostics
    # or direct access to unpublished cluster checkpoints and prediction arrays.
    source_sha('e2_intervention_execution_lock')
    source_sha('e2_intervention_runtime_audit')
    source_sha('e2_intervention_review_audit')
    assert runtime['complete'] and runtime['all_periodic_diagnostics_passed']
    assert runtime['predictions'] == 864 and runtime['failures'] == 0
    assert runtime['canonical_training_steps'] == 9 * 1536
    assert len(runtime['runs']) == 9
    assert {r['run'] for r in runtime['runs']} == {r[0] for r in expected_runs}
    for r in runtime['runs']:
        assert r['canonical_steps'] == 1536 and r['periodic_diagnostics'] == 16
        assert r['predictions'] == 96
        assert 0 <= r['max_orthogonality'] < 1e-3
        assert 0 <= r['max_mean_error'] < 1e-3
        for field in ['checkpoint_sha256', 'training_complete_sha256',
                      'completion_sha256', 'engineering_amendment_sha256']:
            assert re.fullmatch('[0-9a-f]{64}', r[field]), field
    amendments = {r['engineering_amendment_sha256'] for r in runtime['runs']}
    assert len(amendments) == 1
    repair = load('e2_intervention_repair_amendment')
    assert amendments == {source_sha('e2_intervention_repair_amendment')}
    assert repair['execution_lock_sha256'] == execution_sha
    assert repair['e2_release_sha256'] == source_sha('e2_intervention_execution_lock')
    assert repair['scientific_changes'] is False and repair['new_scientific_runs'] == 0
    assert repair['resume_step'] == 0 and repair['orthogonality_and_mean_threshold'] == 1e-3
    regression = load('e2_intervention_repair_regression')
    assert repair['gpu_regression_sha256'] == source_sha('e2_intervention_repair_regression')
    assert regression['passed'] and regression['execution_lock_sha256'] == execution_sha
    assert regression['diagnostic_code_sha256'] == repair['files']['run.py']
    assert len(regression['checks']) == 3 and len(regression['recovery']) == 9
    assert {r['run'] for r in regression['recovery']} == {r[0] for r in expected_runs}
    assert all(r['checkpoint_step'] == 0 and r['logged_steps'] == 95
               for r in regression['recovery'])
    for r in regression['checks']:
        assert r['original_bug_reproduced'] and r['parameters_and_rng_unchanged']
        assert 0 <= r['metrics']['channel_orthogonality'] < 1e-3
        assert 0 <= r['metrics']['channel_mean_error'] < 1e-3
    series = load('e2_intervention_series_complete')
    source_sha('e2_intervention_series_complete')
    assert series['complete'] and series['E3_E4_not_submitted']
    assert series['execution_lock_sha256'] == execution_sha
    assert series['e1_sha256'] == source_sha('e1_prediction_complete')
    assert series['e2_sha256'] == source_sha('e2_intervention_complete')

    assert review['complete'] and review['artifacts_sha_verified']
    assert review['new_records'] == 864 and review['combined_unique_records'] == 3552
    assert review['failures'] == 0 and review['comparisons_verified'] == 21
    assert review['synchronized_target_bootstrap_recomputed']
    audit_by = {r['name']: r for r in review['checks']}
    assert len(audit_by) == len(review['checks']) == 21
    assert execution['bootstrap'] == {'seed': 20260924, 'draws': 20000}
    draws = np.random.default_rng(20260924).integers(96, size=(20000, 96))
    checks, contrasts, max_error = 0, 0, 0.

    def equal(actual, expected, label):
        nonlocal checks, max_error
        actual, expected = np.asarray(actual), np.asarray(expected)
        assert actual.shape == expected.shape, label
        assert np.isfinite(actual).all() and np.isfinite(expected).all(), label
        error = float(np.max(np.abs(actual - expected)))
        assert error < 1e-12, (label, error)
        max_error = max(max_error, error)
        checks += 1

    def contrast(delta, recorded, label):
        nonlocal contrasts
        calculated = conditional_summary(delta, draws)
        for field in ['mean', 'ci95', 'per_target', 'per_seed']:
            equal(calculated[field], recorded[field], label + '/' + field)
        for field in ['mean', 'ci95']:
            equal(calculated[field], audit_by[label][field], 'readback/' + label + '/' + field)
        contrasts += 1

    compact = {}
    for metric in ['ca_lddt', 'residue_ca_lddt', 'tm_score_fixed_full_length']:
        get = lambda name: np.asarray([by[name, target][metric] for target in ids])
        native = np.stack([get(f'native_s{s}') for s in seeds])
        native_c = np.stack([get(f'I_C_s{s}') for s in seeds])
        rotated = np.stack([[get(f'{r}_s{s}') for s in seeds] for r in rotations])
        rotated_c = np.stack([[get(f'{r}_C_s{s}') for s in seeds] for r in rotations])
        effects = paired_effects(native, native_c, rotated, rotated_c)
        equal(effects['T_C'], effects['B_R']-effects['native_change'], metric + '/paired identity')
        recorded = summary['metrics'][metric]
        assert set(recorded['per_rotation']) == set(rotations)
        equal(native.mean(), recorded['native_fixed'], metric + '/native_fixed')
        equal(native_c.mean(), recorded['native_learned'], metric + '/native_learned')
        equal(get('query').mean(), recorded['query_mean'], metric + '/query_mean')
        for name in ['native_change', 'B_R', 'T_C']:
            contrast(effects[name], recorded[name], metric + '/' + name)
        for j, rotation in enumerate(rotations):
            arm = recorded['per_rotation'][rotation]
            equal(rotated[j].mean(), arm['fixed_mean'], metric + '/' + rotation + '/fixed_mean')
            equal(rotated_c[j].mean(), arm['learned_mean'], metric + '/' + rotation + '/learned_mean')
            for name in ['B_R', 'T_C']:
                contrast(effects['rotation_' + name][j], arm[name], metric + '/' + rotation + '/' + name)
        compact[metric] = {name: {key: recorded[name][key] for key in ['mean', 'ci95', 'per_seed']}
                           for name in ['native_change', 'B_R', 'T_C']}
    assert contrasts == 21
    return dict(passed=True, records=864, combined_unique_records=3552, failures=0,
                verified_contrasts=contrasts, scalar_or_array_checks=checks,
                max_abs_error=max_error, paired_seeds=seeds, selected_rotations=rotations,
                synchronized_target_bootstrap=execution['bootstrap'], results=compact,
                diagnostic_repair_hash_link_verified=True,
                predefined_rotation_average_verified=True,
                statistical_identity=('Cross-rotation averages predeclared in the original '
                                      'written design and sealed analysis. Prespecified '
                                      'mechanistic follow-up on observed targets, not a new '
                                      'sole confirmatory primary endpoint or independent '
                                      'new-target confirmation; both rotations and Native '
                                      'changes retained.'),
                source_sha256=source_checks,
                scope=('Original 21 paired score-array contrasts and sealed runtime-audit '
                       'consistency; no new remaining-gap or equivalence test, no training '
                       'or structure-scoring replay. Observed Confirm96-B; intervals '
                       'condition on the fitted models and two preselected rotations.'))


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parents[1]), indent=2))
