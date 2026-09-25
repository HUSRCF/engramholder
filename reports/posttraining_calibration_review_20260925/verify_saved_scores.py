"""Read-only reconstruction of the completed calibration comparisons.

Inputs remain in the execution archive. This does not load checkpoints, rerun
folding, rescore coordinates or add statistical endpoints to the paper.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def verify(root):
    sources, results = {}, {}
    original_comparisons = supplementary_comparisons = scalar_checks = 0
    maximum_error = 0.0
    seeds = [20260923, 20260924, 20260925]
    metrics = ['ca_lddt', 'residue_ca_lddt', 'tm_score_fixed_full_length']

    def read(name):
        content = (root / name).read_bytes()
        sources[name] = hashlib.sha256(content).hexdigest()
        return json.loads(content)

    def equal(actual, expected):
        nonlocal maximum_error, scalar_checks
        a, b = np.asarray(actual), np.asarray(expected)
        assert a.shape == b.shape and np.isfinite(a).all()
        error = float(np.max(np.abs(a-b)))
        assert error < 1e-12, error
        maximum_error = max(maximum_error, error)
        scalar_checks += 1

    verified = read('final_analysis/verified_analysis.json')['results']
    csv_path = root/'final_analysis/paired_scores.csv'
    sources['final_analysis/paired_scores.csv'] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    with csv_path.open() as stream:
        csv_rows = list(csv.DictReader(stream))
    csv_by = {(r['backbone'], r['metric'], r['direction'], int(r['seed']), r['target_id']): r
              for r in csv_rows}
    assert len(csv_by) == len(csv_rows) == 4320

    def effects(states):
        z, c, h = (states[k] for k in ['baseline', 'C_only', 'head_only'])
        out = dict(C_only_gain=c-z, head_only_gain=h-z, C_only_minus_head_only=c-h)
        if 'joint' in states:
            j = states['joint']
            out.update(joint_gain=j-z, joint_minus_head_only=j-h,
                       joint_minus_C_only=j-c, factorial_J=j-h-c+z)
        return out

    for family in ['openfold', 'protenix']:
        prefix = f'remote_{family}'
        analysis_prefix = f'{prefix}/{family}/formal_analysis'
        lock = read(f'{prefix}/execution_lock.json')
        panel = read(f'{prefix}/panels/{family}/formal_reference.json')['targets']
        ids = [t['target_id'] for t in panel]
        published = read(f'{analysis_prefix}/analysis.json')
        rows = read(f'{analysis_prefix}/metric_records.json')
        baseline_path = f'{family}_baseline_records.json'
        baselines = read(baseline_path)
        audit = read(f'{prefix}/operations/completion_audit.json')
        done = read(f'{prefix}/{family}/COMPLETE.json')
        n = 96 if family == 'openfold' else 192
        assert len(ids) == len(set(ids)) == n and ids == published['target_ids']
        assert sources[baseline_path] == lock['baselines'][family]['sha256']
        for receipt in [audit, done]:
            assert receipt['records_sha256'] == sources[f'{analysis_prefix}/metric_records.json']
            assert receipt['analysis_sha256'] == sources[f'{analysis_prefix}/analysis.json']
            assert receipt['execution_lock_sha256'] == sources[f'{prefix}/execution_lock.json']
        assert audit['passed'] and audit['failures'] == 0
        matrix = lock['matrices'][family]['formal']
        expected = {(m['name'], t) for m in matrix for t in ids}
        by = {(r['system'], r['target_id']): r for r in rows}
        old = {(r['system'], r['target_id']): r for r in baselines}
        assert set(by) == expected and len(by) == len(rows) == n*len(matrix)
        assert all(r['status'] == 'ok' for r in rows)
        assert audit['runs'] == len(matrix) and audit['predictions'] == len(rows)
        assert audit['training_updates'] == 512*len(matrix)
        assert {x['name'] for x in audit['checks']} == {m['name'] for m in matrix}
        for check in audit['checks']:
            assert check['steps'] == 512 and check['initial_head_matches_parent']
            branch = next(m['branch'] for m in matrix if m['name'] == check['name'])
            assert check['theta_frozen'] == (branch == 'C_only')
            expected_steps = {'C_only': [512.0], 'head_only': [2048.0], 'joint': [512.0, 2048.0]}
            assert sorted(check['optimizer_final_steps']) == expected_steps[branch]

        rng = np.random.default_rng(20260925)
        if family == 'openfold':
            draws = rng.integers(n, size=(20000, n))
        else:
            groups = [np.array([i for i, t in enumerate(panel) if t['length_bin'] == name])
                      for name in sorted({t['length_bin'] for t in panel})]
            assert len(groups) == 4 and all(len(g) == 48 for g in groups)
            draws = np.concatenate([rng.choice(g, size=(20000, len(g))) for g in groups], axis=1)

        def summary(array):
            target = array.mean(axis=0)
            return dict(mean=float(target.mean()),
                        ci95=np.quantile(target[draws].mean(axis=1), [.025, .975]).tolist(),
                        per_seed_mean=array.mean(axis=1).tolist(),
                        positive_targets=int(np.count_nonzero(target > 0)),
                        per_target=target.tolist())

        def compare(summary_value, reference):
            for key in ['mean', 'ci95', 'per_seed_mean', 'per_target']:
                equal(summary_value[key], reference[key])
            assert summary_value['positive_targets'] == reference['positive_targets']

        family_results = {}
        for metric in metrics:
            states = {}
            directions = ['I', 'r20270107', 'r20270104'] if family == 'openfold' else ['I']
            branches = ['C_only', 'head_only', 'joint'] if family == 'openfold' else ['C_only', 'head_only']
            for direction in directions:
                base_names = [lock['baselines'][family]['names'][f'{direction}_s{s}'] for s in seeds]
                states[direction] = {'baseline': np.array([[old[name, t][metric] for t in ids] for name in base_names])}
                for branch in branches:
                    states[direction][branch] = np.array([
                        [by[f'{family}_{direction}_s{s}_{branch}', t][metric] for t in ids] for s in seeds])
                for branch, values in states[direction].items():
                    equal(values.mean(), published['metrics'][metric]['by_direction'][direction][branch+'_mean'])
                    csv_values = np.array([[float(csv_by[family, metric, direction, s, t][branch])
                                            for t in ids] for s in seeds])
                    equal(values, csv_values)
            if family == 'openfold':
                states['rotated_mean'] = {b: (states[directions[1]][b]+states[directions[2]][b])/2
                                          for b in states['I']}
                primary = effects(states['rotated_mean'])['C_only_gain']-effects(states['I'])['C_only_gain']
            else:
                primary = effects(states['I'])['C_only_minus_head_only']
            main = summary(primary)
            compare(main, published['metrics'][metric]['main'])
            compare(main, verified[family]['metrics'][metric]['primary'])
            original_comparisons += 1
            direction_results = {}
            for direction, state in states.items():
                contrasts = {}
                for name, values in effects(state).items():
                    stat = summary(values)
                    compare(stat, verified[family]['metrics'][metric]['by_direction'][direction]['comparisons'][name])
                    existing = published['metrics'][metric]['by_direction'].get(direction, {})
                    if name in existing:
                        compare(stat, existing[name]); original_comparisons += 1
                    else:
                        supplementary_comparisons += 1
                    contrasts[name] = {k: v for k, v in stat.items() if k != 'per_target'}
                direction_results[direction] = dict(means={k: float(v.mean()) for k, v in state.items()}, contrasts=contrasts)
            family_results[metric] = dict(primary={k: v for k, v in main.items() if k != 'per_target'}, by_direction=direction_results)
        results[family] = dict(runs=len(matrix), predictions=len(rows), targets=n, metrics=family_results)
    assert original_comparisons == 69 and supplementary_comparisons == 30
    return dict(passed=True, original_comparisons=original_comparisons,
                previously_reported_supplementary_comparisons=supplementary_comparisons,
                maximum_error=maximum_error, scalar_or_array_checks=scalar_checks,
                bootstrap_replicates=20000, bootstrap_seed=20260925,
                new_endpoints=False, checkpoint_tensors_reaudited=False,
                coordinate_rescoring=False, source_sha256=sources, results=results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.source_root)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: result[k] for k in ['passed', 'original_comparisons',
                     'previously_reported_supplementary_comparisons', 'maximum_error']}, indent=2))
