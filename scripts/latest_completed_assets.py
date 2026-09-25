"""Fresh192 and the E2 same-seed repeat; no edits to historical numeric cells."""
METRICS = [('ca_lddt', 'pair', 'Pair-lDDT'),
           ('residue_ca_lddt', 'residue', 'Residue-lDDT'),
           ('tm_score_fixed_full_length', 'tm', 'TM-score')]


def build(data, number, contrast, tex, save_rows, interval):
    fresh, marginals, strata = [], [], []
    for metric, tag, label in METRICS:
        base = ['metrics', metric]
        for name in ['query', 'factor_native', 'factor_rotated', 'gplus_native', 'gplus_rotated']:
            number(f'p192_{tag}_{name}', 'protenix_fresh192_summary',
                   base+['four_cell_means', name], decimals=4)
        for name, display in [('factor_rotation', r'$\Delta_F$'), ('gplus_rotation', r'$\Delta_{G+}$'),
                              ('interaction', r'$\Psi$'), ('native_minus_gplus', 'F--G+'),
                              ('native_minus_query', 'F--Query'), ('gplus_minus_query', 'G+--Query'),
                              ('rotated_factor_minus_query', 'RF--Query'), ('rotated_gplus_minus_query', 'RG+--Query')]:
            key = f'p192_{tag}_{name}'
            contrast(key, 'protenix_fresh192_summary', base+[name])
            fresh.append([label, display, tex(key), interval(key)])
        number(f'p192_{tag}_positive', 'protenix_fresh192_summary', base+['interaction', 'positive_targets'], decimals=0)
        for i, length in enumerate(['128-191', '192-255', '256-319', '320-384']):
            key = f'p192_{tag}_stratum{i}'
            path = base+['length_strata_secondary', length]
            number(key, 'protenix_fresh192_summary', path+['mean'], signed=True)
            for j, suffix in enumerate(['Lo', 'Hi']):
                number(key+suffix, 'protenix_fresh192_summary', path+['ci95_unadjusted', j], signed=True)
            strata.append([label, length.replace('-', '--'), tex(key), interval(key)])
    for kind in ['seed', 'rotation']:
        keys = []
        for i in range(3):
            key = f'p192_{kind}{i}'
            number(key, 'protenix_fresh192_summary', ['metrics', 'ca_lddt', 'interaction', 'per_'+kind, i], signed=True)
            keys.append(tex(key))
        marginals.append([kind.capitalize(), *keys])
    for i, rid in enumerate([20261001, 20261002, 20261003]):
        keys = []
        for j in range(3):
            key = f'p192_r{i}_s{j}'
            number(key, 'protenix_fresh192_summary', ['metrics', 'ca_lddt', 'interaction', 'rotation_by_seed', i, j], signed=True)
            keys.append(tex(key))
        marginals.append([f'r{rid}', *keys])
    save_rows('fresh192_contrasts.tex', fresh)
    save_rows('fresh192_marginals.tex', marginals)
    save_rows('fresh192_strata.tex', strata)

    repeat_rows, absolute = [], []
    rotations = [data['e1_prediction_lock']['selected_low'], data['e1_prediction_lock']['selected_high']]
    for metric, tag, label in METRICS:
        base = ['metrics', metric]
        for name, display in [('native_change', r'$B_N$'), ('B_R', r'$B_R^C$'), ('T_C', r'$T_C$')]:
            key = f'e2rep_{tag}_{name}'
            contrast(key, 'e2_retraining_summary', base+[name])
            repeat_rows.append([label, 'Native' if name == 'native_change' else 'Two-R mean', display, tex(key), interval(key)])
        for i, rid in enumerate(rotations, 1):
            for name, display in [('B_R', r'$B_R^C$'), ('T_C', r'$T_C$')]:
                key = f'e2rep_{tag}_r{i}_{name}'
                contrast(key, 'e2_retraining_summary', base+['per_rotation', rid, name])
                repeat_rows.append([label, rid, display, tex(key), interval(key)])
        number(f'e2rep_{tag}_native', 'e2_retraining_summary', base+['native_learned'], decimals=4)
        for i, rid in enumerate(rotations, 1):
            number(f'e2rep_{tag}_r{i}', 'e2_retraining_summary', base+['per_rotation', rid, 'learned_mean'], decimals=4)
    absolute.append(['Native', tex('e2_native_fixed'), tex('e2_native_learned'), tex('e2rep_pair_native')])
    for i, rid in enumerate(rotations, 1):
        absolute.append([rid, tex(f'e2_r{i}_fixed_mean'), tex(f'e2_r{i}_learned_mean'), tex(f'e2rep_pair_r{i}')])
    save_rows('e2_execution_means.tex', absolute)
    save_rows('e2_retraining_contrasts.tex', repeat_rows)
    comparison = []
    for name, display in [('B_R', r'$B_R^C$'), ('native_change', r'$B_N$'), ('T_C', r'$T_C$')]:
        old, repeat = f'e2_pair_{name}', f'e2rep_pair_{name}'
        comparison.append([display, tex(old), interval(old), tex(repeat), interval(repeat)])
    save_rows('e2_retraining_comparison.tex', comparison)
