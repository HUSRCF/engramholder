"""Generate the complete factorial tables without changing historical cells."""


def build(data, number, contrast, tex, save_rows, interval):
    specs = [('Protenix', '96', 'C96-B', 'ESM2', 'p96',
              'pt96_fourcell_summary', ['metrics', 'ca_pair_lddt'])]
    for backbone, label, train in [('protenix', 'Protenix', '384'), ('atlas', 'AtlasFold', '96')]:
        for panel, short, panel_label in [('confirm96', 'c96', 'C96-B'), ('length48', 'l48', 'L48')]:
            for feature in ['E', 'C']:
                specs.append((label, train, panel_label, 'ESM2' if feature == 'E' else 'ESMC',
                              f'dh_{backbone}_{short}_{feature.lower()}', 'dh_A66_summary',
                              ['panels', backbone, panel, 'ca_lddt', 'cells', feature]))
    means, supplements, marginals = [], {}, []
    for backbone, train, panel, feature, prefix, source, path in specs:
        names = ['query', 'native', 'rotated_factor', 'gplus', 'rotated_gplus']
        for name in names:
            number(prefix+'_'+name, source, path+['means', name])
        means.append([backbone+'/'+train, panel, feature, *[tex(prefix+'_'+n) for n in names]])
        for metric, tag in [('ca_lddt', 'pair'), ('residue_ca_lddt', 'residue'), ('tm_score_fixed_full_length', 'tm')]:
            metric_path = (['metrics', 'ca_pair_lddt' if metric == 'ca_lddt' else metric]
                           if prefix == 'p96' else [*path[:3], metric, *path[4:]])
            for name, label in [('factor_rotation', r'$\Delta_F$'), ('gplus_rotation', r'$\Delta_{G+}$'),
                                ('interaction', r'$\Psi$'), ('native_minus_gplus', 'F--G+'),
                                ('native_minus_query', 'F--Query'), ('gplus_minus_query', 'G+--Query')]:
                key = prefix+'_'+tag+'_'+name
                contrast(key, source, metric_path+[name])
                if name not in ['native_minus_query', 'gplus_minus_query']:
                    supplements.setdefault((backbone, tag), []).append(
                        [train+'/'+feature, panel, label, tex(key), interval(key)])
        obj = data[source]
        for part in path+['interaction']:
            obj = obj[part]
        # Marginals are descriptive; they are not three or nine independent targets.
        marginals.append([backbone+'/'+train, panel, feature,
                          ', '.join(f'{x:+.5f}' for x in obj['per_seed']),
                          ', '.join(f'{x:+.5f}' for x in obj['per_rotation'])])

    of_means = []
    for short, panel in [('c96', 'C96-B'), ('l48', 'L48')]:
        of_means.append(['OpenFold/96', panel, 'ESM2', tex(f'of96_{short}_query'),
                         *[tex(f'inter_{short}_{k}') for k in ['factor_native', 'factor_rotated', 'gplus_native', 'gplus_rotated']]])
        for feature, tag in [('ESM2', 'e'), ('ESMC', 'c')]:
            of_means.append(['OpenFold/384', panel, feature,
                             *[tex(f'a_{short}_{tag}_{n}') for n in names]])
    # Group recipe labels rather than repeating long predictor names in every row.
    ordered = sorted(means[:5]+of_means+means[5:], key=lambda row: (
        ['Protenix', 'OpenFold', 'AtlasFold'].index(row[0].split('/')[0]),
        int(row[0].split('/')[1]), ['ESM2', 'ESMC'].index(row[2]),
        ['C96-B', 'L48'].index(row[1])))
    compact, previous = [], None
    for model, panel, feature, *scores in ordered:
        predictor, train = model.split('/')
        recipe = (predictor, train, feature)
        same_predictor = previous is not None and predictor == previous[0]
        same_train = same_predictor and train == previous[1]
        same_feature = same_train and feature == previous[2]
        label = '' if same_predictor else predictor
        if previous is not None and not same_predictor:
            label = r'\midrule ' + label
        compact.append([label, '' if same_train else train,
                        '' if same_feature else feature, panel, *scores])
        previous = recipe
    save_rows('complete_fourcell_means.tex', compact)

    interactions = []
    for label, keys in [
        ('OpenFold 96 / ESM2', ['inter_c96_interaction', 'inter_l48_interaction']),
        ('OpenFold 384 / ESM2', ['a_c96_e_ca_lddt_interaction', 'a_l48_e_ca_lddt_interaction']),
        ('OpenFold 384 / ESMC', ['a_c96_c_ca_lddt_interaction', 'a_l48_c_ca_lddt_interaction']),
        ('Protenix 96 / ESM2', ['p96_pair_interaction', None]),
        ('Protenix 384 / ESM2', ['dh_protenix_c96_e_pair_interaction', 'dh_protenix_l48_e_pair_interaction']),
        ('Protenix 384 / ESMC', ['dh_protenix_c96_c_pair_interaction', 'dh_protenix_l48_c_pair_interaction']),
        ('AtlasFold 96 / ESM2', ['dh_atlas_c96_e_pair_interaction', 'dh_atlas_l48_e_pair_interaction']),
        ('AtlasFold 96 / ESMC', ['dh_atlas_c96_c_pair_interaction', 'dh_atlas_l48_c_pair_interaction'])]:
        interactions.append([label, *['---' if k is None else tex(k)+' '+interval(k) for k in keys]])
    save_rows('complete_interaction_scope.tex', interactions)
    for (backbone, tag), rows in supplements.items():
        save_rows(f'dh_{backbone.lower()}_{tag}_contrasts.tex', rows)
    save_rows('dh_interaction_marginals.tex', marginals)
    changes = []
    for backbone, label in [('protenix', 'Protenix'), ('atlas', 'AtlasFold')]:
        for panel, short, panel_label in [('confirm96', 'c96', 'C96-B'), ('length48', 'l48', 'L48')]:
            for name, display in [('direction_change', r'$\Delta_{F,C}-\Delta_{F,E}$'),
                                  ('gplus_direction_change', r'$\Delta_{G+,C}-\Delta_{G+,E}$'),
                                  ('psi_change', r'$\Psi_C-\Psi_E$'), ('native_gain', 'Native gain'),
                                  ('gplus_gain', 'G+ gain'), ('native_gplus_gap_change', 'Change in F--G+')]:
                key = f'dh_{backbone}_{short}_{name}'
                contrast(key, 'dh_A66_summary', ['panels', backbone, panel, 'ca_lddt', 'plm_interactions', name])
                changes.append([label, panel_label, display, tex(key), interval(key)])
    save_rows('dh_feature_changes.tex', changes)
