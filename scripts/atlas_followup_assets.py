"""Compact Atlas boundary table; keep full propagation records in the artifact."""


def build(number, tex, save_rows):
    rows = []
    for site, label in [('after_injection', 'After injection'),
                        ('lm_block4', 'LM stack output'),
                        ('projection_out', 'After projection'),
                        ('trunk_input', 'After fusion'),
                        ('trunk_end', 'Trunk output')]:
        keys = []
        for statistic, digits in [('relative_delta_rms', 4), ('delta_rms', 2), ('query_rms', 2)]:
            key = f'atlas_path_{site}_{statistic}'
            number(key, 'atlas_propagation_summary',
                   ['summary', 'parent', 'z', site, statistic, 'median'], decimals=digits)
            keys.append(tex(key))
        rows.append([label, *keys])
    save_rows('atlas_propagation_rows.tex', rows)
    for name in ['C_only_minus_head_only', 'C_only_minus_query']:
        for suffix, field in [('', ['mean']), ('Lo', ['ci95', 0]), ('Hi', ['ci95', 1])]:
            number('atlas_cal_' + name + suffix, 'atlas_posttraining_summary',
                   ['metrics', 'ca_lddt', 'contrasts', name] + field,
                   signed=True, decimals=6)
