"""Tables for completed controls, preserving each study's original estimands."""

METRICS = [('ca_lddt', 'pair', 'Pair-lDDT'),
           ('residue_ca_lddt', 'residue', 'Residue-lDDT'),
           ('tm_score_fixed_full_length', 'tm', 'TM-score')]


def build(data, number, contrast, tex, save_rows, interval):
    signed_rows, signed_means = [], []
    for panel, short, label in [('confirm96', 'c96', 'C96-B'), ('length48', 'l48', 'L48')]:
        for metric, ms, ml in METRICS:
            base = ['panels', panel, metric]
            for name, display in [('factor_rotation', r'$\Delta_F$'),
                                  ('gplus_rotation', r'$\Delta_{G+}$'),
                                  ('interaction', r'$\Psi$')]:
                key = f'signed_{short}_{ms}_{name}'
                contrast(key, 'signed_summary', base + [name])
                signed_rows.append([label, ml, display, tex(key), interval(key)])
            if metric == 'ca_lddt':
                keys = []
                for name in ['query', 'factor_native', 'factor_rotated', 'gplus_native', 'gplus_rotated']:
                    key = f'signed_{short}_{name}'
                    number(key, 'signed_summary', base + ['group_means', name])
                    keys.append(tex(key))
                signed_means.append([label, *keys])
    save_rows('signed_contrasts.tex', signed_rows)
    save_rows('signed_means.tex', signed_means)

    e2_rows, e2_arms = [], []
    rotations = [data['e1_prediction_lock']['selected_low'], data['e1_prediction_lock']['selected_high']]
    for metric, ms, ml in METRICS:
        base = ['metrics', metric]
        for name, display in [('native_change', r'$B_N$'), ('B_R', r'$B_R$'), ('T_C', r'$T_C$')]:
            key = f'e2_{ms}_{name}'
            contrast(key, 'e2_intervention_summary', base + [name])
            e2_rows.append([ml, 'Two-R mean' if name != 'native_change' else 'Native', display, tex(key), interval(key)])
        for i, rid in enumerate(rotations, 1):
            for name, display in [('B_R', r'$B_R$'), ('T_C', r'$T_C$')]:
                key = f'e2_{ms}_r{i}_{name}'
                contrast(key, 'e2_intervention_summary', base + ['per_rotation', rid, name])
                e2_rows.append([ml, rid, display, tex(key), interval(key)])
        if metric == 'ca_lddt':
            number('e2_native_fixed', 'e2_intervention_summary', base + ['native_fixed'])
            number('e2_native_learned', 'e2_intervention_summary', base + ['native_learned'])
            e2_arms.append(['Native', tex('e2_native_fixed'), tex('e2_native_learned'),
                            tex('e2_pair_native_change'), interval('e2_pair_native_change')])
            for i, rid in enumerate(rotations, 1):
                for name in ['fixed_mean', 'learned_mean']:
                    number(f'e2_r{i}_{name}', 'e2_intervention_summary', base + ['per_rotation', rid, name])
                e2_arms.append([rid, tex(f'e2_r{i}_fixed_mean'), tex(f'e2_r{i}_learned_mean'),
                                tex(f'e2_pair_r{i}_B_R'), interval(f'e2_pair_r{i}_B_R')])
    save_rows('e2_contrasts.tex', e2_rows)
    save_rows('e2_means.tex', e2_arms)
    # Descriptions of the already paired target effects, not new hypothesis tests.
    for name in ['B_R', 'T_C']:
        path = ['metrics', 'ca_lddt', name, 'per_target']
        for operation in ['median', 'count_positive']:
            number(f'e2_pair_{name}_{operation}', 'e2_intervention_summary', path,
                   operation=operation, signed=(operation == 'median'))

    curve_rows, curve_means = [], []
    for metric, ms, ml in METRICS:
        for size in ['96', '384']:
            for step in ['384', '768', '1536']:
                base = ['curves', metric, size, step]
                prefix = f'curve_{ms}_n{size}_s{step}'
                values = []
                for name in ['factor_rotation', 'gplus_rotation', 'interaction']:
                    key = prefix + '_' + name
                    contrast(key, 'checkpoint_curves_summary', base + [name])
                    values.append(tex(key))
                curve_rows.append([ml, size, step, *values, interval(prefix + '_interaction')])
                if metric == 'ca_lddt':
                    means = []
                    for name in ['factor_native', 'factor_rotated', 'gplus_native', 'gplus_rotated']:
                        key = prefix + '_' + name
                        number(key, 'checkpoint_curves_summary', base + ['means', name])
                        means.append(tex(key))
                    curve_means.append([size, step, *means])
    save_rows('checkpoint_curve_contrasts.tex', curve_rows)
    save_rows('checkpoint_curve_means.tex', curve_means)
