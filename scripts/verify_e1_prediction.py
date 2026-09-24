"""Reconstruct E1 from fixed score arrays and enumerate the eight-unit rank test.

This checks the completed prediction experiment, not CIF scoring or the existence
of an optimally compensating update.  Target bootstrap intervals are conditional
sensitivity summaries; their 96 targets are not additional rotation test units.
"""
import datetime
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


def doubled_ranks(values):
    """Twice the average ranks, represented exactly as integers (including ties)."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError('Ranks require a finite one-dimensional array')
    return (2 * (values[:, None] > values[None, :]).sum(1)
            + (values[:, None] == values[None, :]).sum(1) + 1)


def spearman(x, y):
    rx, ry = doubled_ranks(x), doubled_ranks(y)
    if rx.shape != ry.shape:
        raise ValueError('Rank arrays must have matching shapes')
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denominator = np.linalg.norm(rx) * np.linalg.norm(ry)
    return float(rx @ ry / denominator) if denominator else float('nan')


def exact_positive_rank_test(x, y):
    """All labelled Y permutations; integer rank products include equal statistics.

    No Monte Carlo correction or asymptotic p-value is used.  Duplicated rank
    values are intentionally repeated when tied labelled observations permute.
    """
    rx, ry = doubled_ranks(x), doubled_ranks(y)
    if rx.shape != ry.shape or not 2 <= len(rx) <= 8:
        raise ValueError('Exact enumeration requires matching arrays of 2--8 units')
    rx, ry = rx - (len(rx) + 1), ry - (len(ry) + 1)
    denominator = np.linalg.norm(rx) * np.linalg.norm(ry)
    if not denominator:
        raise ValueError('A constant variable does not define a rank test')
    observed = int(rx @ ry)
    greater_equal = total = 0
    for permuted in itertools.permutations(ry.tolist()):
        greater_equal += sum(int(a) * b for a, b in zip(rx, permuted)) >= observed
        total += 1
    return dict(rho=float(observed / denominator), p_one_sided=greater_equal / total,
                extreme_count=greater_equal, permutations=total, n_rotations=len(rx))


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

    a = load('e1_prediction_summary')
    p = load('e1_prediction_lock')
    rows = load('e1_prediction_records')
    c = load('e1_execution_lock')
    done = load('e1_prediction_complete')
    execution_sha = source_sha('e1_execution_lock')
    assert a['execution_lock_sha256'] == p['execution_lock_sha256'] == execution_sha
    assert done['execution_lock_sha256'] == execution_sha
    assert a['prediction_lock_sha256'] == source_sha('e1_prediction_lock')
    assert done['analysis_sha256'] == source_sha('e1_prediction_summary')
    assert done['records_sha256'] == source_sha('e1_prediction_records')
    assert done['scoring_amendment_sha256'] == a['scoring_amendment_sha256']
    assert a['stage'] == done['stage'] == 'e1' and done['complete']
    assert done['new_predictions'] == 2304
    assert done['new_failures'] == done['comparison_failures'] == 0
    assert a['new_failures'] == a['failures'] == 0
    assert p['released'] and p['structure_training_started'] is False
    parse_time = datetime.datetime.fromisoformat
    assert parse_time(c['time_utc']) < parse_time(p['utc']) < parse_time(a['utc'])

    ids, seeds, rotations = c['confirm_ids'], c['formal_seeds'], c['rotation_ids']
    assert len(ids) == len(set(ids)) == 96
    assert len(seeds) == len(set(seeds)) == 3
    assert len(rotations) == len(set(rotations)) == 8
    assert rotations == p['rotation_ids']
    expected_runs = {(f'{r}_s{s}', r, s, 1536, 'factor') for r in rotations for s in seeds}
    assert len(c['new_runs']) == 24
    assert {(v['name'], v['rotation_id'], v['seed'], v['steps'], v['kind'])
            for v in c['new_runs']} == expected_runs
    assert [t['seed'] for t in c['teachers']] == seeds
    assert all(t['train_size'] == 96 for t in c['teachers'])
    expected_systems = {v[0] for v in expected_runs} | {f'native_s{s}' for s in seeds} | {'query'}
    by = {(v['system'], v['target_id']): v for v in rows}
    assert len(rows) == len(by) == 2688
    assert set(by) == {(name, tid) for name in expected_systems for tid in ids}
    assert all(v['status'] == 'ok' for v in rows)

    # The recorded repair changed an auxiliary check's input dtypes, not the
    # primary scores, thresholds, targets, or reference-pair membership.
    amendment = load('e1_scoring_amendment')
    diagnosis = load('e1_scoring_diagnosis')
    regression = load('e1_scoring_regression')
    assert a['scoring_amendment_sha256'] == source_sha('e1_scoring_amendment')
    assert amendment['execution_lock_sha256'] == execution_sha
    assert amendment['prediction_lock_sha256'] == source_sha('e1_prediction_lock')
    assert amendment['diagnosis_sha256'] == source_sha('e1_scoring_diagnosis')
    assert amendment['regression_sha256'] == source_sha('e1_scoring_regression')
    assert amendment['scientific_changes'] is False
    assert amendment['new_training_or_prediction'] is False
    assert amendment['threshold_unchanged'] == 1e-6
    assert amendment['coverage'] == diagnosis['checked'] == len(rows)
    assert diagnosis['primary_recomputed_exactly']
    assert regression['passed'] and regression['primary_unchanged']
    assert regression['secondary_dtype_unchanged'] and regression['no_threshold_relaxation']
    mismatches = {(v['system'], v['target_id']) for v in rows
                  if abs(v['ca_lddt'] - v['reference_check_fp64']) >= 1e-6}
    assert len(mismatches) == amendment['precision_mismatches'] == diagnosis['mismatches'] == 4
    assert mismatches == {(v['system'], v['target_id']) for v in diagnosis['details']}
    assert all(abs(v['ca_lddt'] - v['reference_check_same_input_dtypes']) < 1e-6 for v in rows)
    assert all(v['pair_mask_differences'] == 0 and len(v['threshold_crossings']) == 1
               for v in diagnosis['details'])
    assert {(v['system'], v['target_id']) for v in regression['rows']} == mismatches
    assert all(v['primary_unchanged'] and v['same_dtype_difference'] < 1e-6
               for v in regression['rows'])
    for item in diagnosis['details']:
        record = by[item['system'], item['target_id']]
        assert item['primary'] == record['ca_lddt']
        assert item['reference64'] == record['reference_check_fp64']
        assert item['prediction_sha256'] == record['prediction_sha256']

    checks, contrasts, max_error = 0, 0, 0.

    def equal(actual, expected, label):
        nonlocal checks, max_error
        actual, expected = np.asarray(actual), np.asarray(expected)
        assert actual.shape == expected.shape, (label, actual.shape, expected.shape)
        assert np.isfinite(actual).all() and np.isfinite(expected).all(), label
        error = float(np.max(np.abs(actual - expected)))
        assert error < 1e-12, (label, error)
        checks += 1
        max_error = max(error, max_error)

    # The prediction seal is verified from its saved teacher summaries.  This is
    # not a fresh reconstruction fit or a replay of the Dev diagnostic structures.
    x_teachers, d_teachers = np.asarray(p['X_by_teacher']), np.asarray(p['D_by_teacher'])
    assert x_teachers.shape == d_teachers.shape == (8, 3)
    equal(x_teachers.mean(1), p['X'], 'teacher averaged X')
    equal(d_teachers.mean(1), p['D'], 'teacher averaged D')
    probe_by = {(v['rotation'], v['teacher_seed']): v for v in p['probe_summaries']}
    assert len(p['probe_summaries']) == len(probe_by) == 24
    equal([[probe_by[r, s]['mean_error'] for s in seeds] for r in rotations],
          x_teachers, 'probe means')
    x_order = sorted(range(8), key=lambda i: (p['X'][i], rotations[i]))
    d_order = sorted(range(8), key=lambda i: (p['D'][i], rotations[i]))
    assert p['X_ascending'] == [rotations[i] for i in x_order]
    assert p['D_ascending'] == [rotations[i] for i in d_order]
    assert p['selected_low'] == rotations[x_order[0]]
    assert p['selected_high'] == rotations[x_order[-1]]
    equal(np.ptp(p['X']), p['X_span'], 'diagnostic span')
    assert p['X_min_span'] == c['X_min_span'] == 1e-5
    assert p['X_span'] > p['X_min_span']
    leave_one = [np.delete(x_teachers, k, axis=1).mean(1) for k in range(3)]
    equal(leave_one, p['leave_one_teacher_X'], 'leave one teacher values')

    assert c['bootstrap'] == {'seed': 20260924, 'draws': 20000}
    draw = np.random.default_rng(c['bootstrap']['seed']).integers(96, size=(20000, 96))

    def check_contrast(delta, obj, label):
        nonlocal contrasts
        target = delta.mean(0)
        equal(target, obj['per_target'], label + ': targets')
        equal(target.mean(), obj['mean'], label + ': mean')
        equal(delta.mean(1), obj['per_seed'], label + ': seeds')
        equal(np.quantile(target[draw].mean(1), [.025, .975]), obj['ci95'], label + ': CI')
        contrasts += 1

    exact_results = {}
    for metric in ['ca_lddt', 'residue_ca_lddt', 'tm_score_fixed_full_length']:
        m = a['metrics'][metric]
        get = lambda name: np.asarray([by[name, i][metric] for i in ids])
        native = np.stack([get(f'native_s{s}') for s in seeds])
        query = get('query')
        rotated = np.stack([[get(f'{r}_s{s}') for s in seeds] for r in rotations])
        delta = native[None] - rotated
        y = delta.mean((1, 2))
        equal(native.mean(), m['native_mean'], metric + ': Native')
        equal(query.mean(), m['query_mean'], metric + ': Query')
        equal(rotated.mean((1, 2)), m['rotated_means'], metric + ': Rotated')
        equal(y, m['Y'], metric + ': Y')
        check_contrast(native-query, m['native_minus_query'], metric + ': Native-Query')
        for j, rid in enumerate(rotations):
            check_contrast(delta[j], m['rotation_costs'][rid], metric + ': ' + rid)
            check_contrast(rotated[j]-query, m['rotated_minus_query'][rid],
                           metric + ': ' + rid + '-Query')
        if metric != 'ca_lddt':
            continue
        for diagnostic, field in [('X', 'X_test'), ('D', 'D_exploratory_test')]:
            test = exact_positive_rank_test(p[diagnostic], y)
            old = m[field]
            assert old['status'] == 'ok' and old['n_rotations'] == 8
            assert old['permutations'] == test['permutations'] == 40320
            equal(test['rho'], old['rho'], diagnostic + ': rho')
            equal(test['p_one_sided'], old['p_one_sided'], diagnostic + ': exact p')
            exact_results[diagnostic] = test
        for k, seed in enumerate(seeds):
            marginal = m['per_training_seed'][k]
            assert marginal['seed'] == seed
            equal(spearman(p['X'], delta[:, k].mean(1)), marginal['X_rho'], 'seed X')
            equal(spearman(p['D'], delta[:, k].mean(1)), marginal['D_rho'], 'seed D')
        equal([spearman(x, y) for x in leave_one], m['leave_one_teacher_X_rho'],
              'leave one teacher correlation')
        # Resample the same targets simultaneously for all eight rotations.
        target_delta = delta.mean(1)
        correlations = []
        for chunk in np.array_split(draw, 40):
            samples = target_delta[:, chunk].mean(2).T
            correlations.extend(spearman(p['X'], v) for v in samples)
        correlations = np.asarray(correlations)
        finite = np.isfinite(correlations)
        bootstrap = m['target_bootstrap_X_rho']
        assert bootstrap['draws'] == 20000
        assert bootstrap['constant_outcome_draws'] == int((~finite).sum())
        equal(np.quantile(correlations[finite], [.025, .975]), bootstrap['ci95'],
              'conditional target bootstrap of rank correlation')

    return dict(passed=True, records=len(rows), new_predictions=2304,
                historical_predictions=384, failures=0, prediction_units=8,
                main_X=exact_results['X'], exploratory_D=exact_results['D'],
                verified_contrasts=contrasts, scalar_or_array_checks=checks,
                max_abs_error=max_error, selected_low=p['selected_low'],
                selected_high=p['selected_high'], X_span=p['X_span'],
                auxiliary_dtype_mismatches=len(mismatches),
                scope=('Score-array, sealed diagnostic-summary, exact rotation-permutation '
                       'and conditional target-bootstrap reconstruction; no CIF scoring, '
                       'training rerun or proof of optimal reconstruction. E1 uses observed '
                       'Confirm96-B and is not new-target confirmation.'))


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parents[1]), indent=2))
