"""Compact tables for the completed budget and schedule follow-ups."""


def build(number, score, contrast, tex, save_rows):
    pair = 'ca_lddt'
    pt = 'pt_budget_summary'
    of = 'of_schedule_summary'
    interval = lambda k: r'$[' + tex(k+'Lo') + ', ' + tex(k+'Hi') + ']$'
    labels = [('factor_native', 'Factor Native'), ('factor_rotated', 'Factor Rotated'),
              ('gplus_native', 'G+ Native'), ('gplus_rotated', 'G+ Rotated')]
    score('budget_query', pt, ['metrics', pair, '1536', 'query_mean'])
    rows = [['Query', tex('budget_query'), tex('budget_query'), '--', '--']]
    for name, label in labels:
        keys = []
        for node in ['1536', '3072']:
            key = f'budget_{name}_{node}'
            score(key, pt, ['metrics', pair, node, 'cells', name, 'mean'])
            keys.append(tex(key))
        key = f'budget_gain_{name}'
        contrast(key, pt, ['metrics', pair, 'budget_change', 'cell_gains', name])
        rows.append([label, *keys, tex(key), interval(key)])
    save_rows('budget_fourcell_rows.tex', rows)
    rows = []
    for name, label, change in [('factor_rotation', r'$\Delta_F$', 'factor_rotation_change'),
                               ('gplus_rotation', r'$\Delta_{G+}$', 'gplus_rotation_change'),
                               ('interaction', r'$\Psi$', 'K')]:
        fields = [label]
        for node in ['1536', '3072']:
            key = f'budget_{name}_{node}'
            contrast(key, pt, ['metrics', pair, node, 'contrasts', name])
            fields += [tex(key), interval(key)]
        key = f'budget_{change}'
        contrast(key, pt, ['metrics', pair, 'budget_change', change])
        fields += [tex(key), interval(key)]
        rows.append(fields)
    save_rows('budget_effect_rows.tex', rows)
    for metric, short in [('residue_ca_lddt', 'residue'), ('tm_score_fixed_full_length', 'tm')]:
        contrast('budget_psi_'+short, pt, ['metrics', metric, '3072', 'contrasts', 'interaction'])
        contrast('budget_K_'+short, pt, ['metrics', metric, 'budget_change', 'K'])
    number('budget_positive', pt, ['metrics', pair, '3072', 'contrasts', 'interaction', 'positive_targets'], decimals=0)
    rows = []
    for schedule, label in [('first', 'First-only'), ('all', 'All'), ('last', 'Last-only')]:
        base = ['statistics', pair, 'conditions', schedule]
        fields = [label]
        for arm, _ in labels:
            key = f'schedule_{schedule}_{arm}'
            score(key, of, base + ['absolute', arm, 'mean'])
            fields.append(tex(key))
        for name in ['factor_rotation', 'gplus_rotation', 'interaction', 'native_factor_minus_gplus']:
            contrast(f'schedule_{schedule}_{name}', of, base + [name])
        key = f'schedule_{schedule}_interaction'
        rows.append([*fields, tex(key), interval(key)])
    save_rows('schedule_fourcell_rows.tex', rows)
    score('schedule_query', of, ['statistics', pair, 'query', 'mean'])
    rows = []
    for name, label in [('D_write_R1', 'All $-$ First'), ('D_timing_R1', 'First $-$ Last'), ('D_last_R1', 'All $-$ Last')]:
        key = f'schedule_{name}'
        contrast(key, of, ['statistics', pair, 'differences', name, 'interaction'])
        rows.append([label, tex(key), interval(key)])
    save_rows('schedule_difference_rows.tex', rows)
    for metric, short in [('residue_ca_lddt', 'residue'), ('tm_score_fixed_full_length', 'tm')]:
        contrast('schedule_write_'+short, of, ['statistics', metric, 'differences', 'D_write_R1', 'interaction'])
