"""Render the sealed anchor study with notation distinct from compensation."""


def build(data, number, contrast, tex, save_rows, interval):
    rotations = data['e3_anchor_execution_lock']['selected_rotations']
    source = 'e3_anchor_summary'
    pooled, individual = [], []
    metrics = [('ca_lddt', 'pair', 'Pair-lDDT'),
               ('residue_ca_lddt', 'residue', 'Residue-lDDT'),
               ('tm_score_fixed_full_length', 'tm', 'TM-score')]
    for metric, tag, label in metrics:
        base = ['metrics', metric]
        for name, display in [('T_A', r'$T_A$'),
                              ('B_R_live_minus_Q', r'$B_R^{\mathrm{anchor}}$'),
                              ('native_Q_minus_live', r'$N_Q-N_L$')]:
            key = f'anchor_{tag}_{name}'
            contrast(key, source, base+[name])
            # Count fields are integers, not four/five-decimal score estimates.
            number(key+'_positive', source, base+[name, 'per_target'], operation='count_positive')
            pooled.append([label, display, tex(key), interval(key), tex(key+'_positive')+'/96'])
            if metric == 'ca_lddt':
                for i in range(3):
                    number(key+f'_seed{i+1}', source, base+[name, 'per_seed', i], signed=True)
        if metric == 'ca_lddt':
            contrast('anchor_pair_native_Q_minus_query', source, base+['native_Q_minus_query'])
        for i, rid in enumerate(rotations, 1):
            for name, display in [('T_A', r'$T_A$'),
                                  ('B_R_live_minus_Q', r'$B_R^{\mathrm{anchor}}$')]:
                key = f'anchor_{tag}_r{i}_{name}'
                path = base+['per_rotation', rid, name]
                contrast(key, source, path)
                # Preserve the very small positive endpoint instead of rounding to zero.
                lo = data[source]['metrics'][metric]['per_rotation'][rid][name]['ci95'][0]
                if 0 < abs(lo) < .000005:
                    number(key+'Lo', source, path+['ci95', 0], signed=True, scientific=True)
                individual.append([label, rid, display, tex(key), interval(key)])
    save_rows('anchor_pooled.tex', pooled)
    save_rows('anchor_individual.tex', individual)

    absolute = ['metrics', 'ca_lddt', 'absolute']
    for name in ['query', 'native_live', 'native_Q', 'rotated_live', 'rotated_Q']:
        number('anchor_'+name, 'e3_anchor_statistical_readback', absolute+[name], decimals=4)
    for name in ['live_native_minus_rotated', 'Q_native_minus_rotated']:
        number('anchor_'+name, 'e3_anchor_statistical_readback', absolute+[name], signed=True)
    means = [['Query-only', tex('anchor_query'), tex('anchor_query')],
             ['Native', tex('anchor_native_live'), tex('anchor_native_Q')],
             ['Rotated (two-R mean)', tex('anchor_rotated_live'), tex('anchor_rotated_Q')]]
    for i, rid in enumerate(rotations, 1):
        for name in ['rotated_live', 'rotated_Q']:
            number(f'anchor_r{i}_{name}', source,
                   ['metrics', 'ca_lddt', 'per_rotation', rid, name], decimals=4)
        means.append([rid, tex(f'anchor_r{i}_rotated_live'), tex(f'anchor_r{i}_rotated_Q')])
    means.append(['Native--Rotated', tex('anchor_live_native_minus_rotated'),
                  tex('anchor_Q_native_minus_rotated')])
    save_rows('anchor_means.tex', means)
