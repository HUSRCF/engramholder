"""Vector layouts for the adapter and the complete interaction evidence."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle


def draw_adapter(directory):
    fig, ax = plt.subplots(figsize=(6.6, 3.65))
    fig.subplots_adjust(left=.01, right=.99, bottom=.02, top=.99)
    ax.set(xlim=(0, 13), ylim=(0, 7.3))
    ax.axis('off')
    frozen, learned, intervention = '#f0f2f4', '#deebf7', '#fff0d7'

    def box(x, y, w, h, text, color=frozen, fs=8):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=.04',
                                   facecolor=color, edgecolor='.35', linewidth=.65))
        ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=fs)

    def arrow(x, y, xx, yy, style='arc3', color='.3'):
        ax.add_patch(FancyArrowPatch((x, y), (xx, yy), arrowstyle='-|>',
                                    mutation_scale=9, connectionstyle=style,
                                    linewidth=.75, color=color))

    ax.text(.1, 7.04, 'A  Factor: shared residue increments, frozen composition',
            fontsize=9, weight='bold')
    ax.text(.1, 3.65, 'B  G+: explicit anchors, free pair-output layer',
            fontsize=9, weight='bold')
    for y, kind in [(4.85, 'factor'), (1.45, 'generic')]:
        box(.1, y, 1.65, 1.1, 'Full sequence\nExtra PLM\n(frozen)\n'+r'$e_{1:L}$')
        box(2.1, y, 2.05, 1.1,
            'Sequence encoder\n'+('Increment head\n'+r'$\delta a,\delta b$'+'\nZero final layer'
                                  if kind == 'factor' else r'$h_{1:L}=f_\phi(e_{1:L})$'+'\nPair features '+r'$h_i,h_j$'), learned, 7.8)
        if kind == 'factor':
            box(4.55, y, 3.05, 1.1,
                r'$G(a+\delta a,b+\delta b)$'+'\n'+r'$-\ G(a,b)$'+'\nFrozen native weights', frozen, 8)
            box(4.85, y+1.55, 2.45, .45, r'Anchor factors $a_i,b_j$', frozen)
            arrow(6.07, y+1.55, 6.07, y+1.1)
        else:
            box(4.55, y, 3.05, 1.1,
                'Pair MLP + free affine\n'+r'$\Delta U_G=W h_{\rm pair}+b$'+'\nZero final W, b', learned, 8)
            box(4.45, y+1.5, 3.25, .48, r'$a_i,b_j,\ \mathrm{clip}(i-j)$', frozen)
            arrow(6.07, y+1.5, 6.07, y+1.1)
        box(8.02, y, 1.02, 1.1, r'$R\Delta U$'+'\nR fixed\nor I', intervention, 8)
        ax.add_patch(Circle((9.75, y+.55), .20, facecolor='white', edgecolor='.3', lw=.7))
        ax.text(9.75, y+.55, '+', ha='center', va='center', fontsize=10)
        box(9.1, y+1.5, 1.3, .5, r'Original $U_0$', frozen, 8)
        arrow(9.75, y+1.5, 9.75, y+.76)
        box(10.45, y, 2.35, 1.1, 'Frozen folding\ndownstream\nTask loss', frozen, 8)
        for left, right in [(1.75, 2.1), (4.15, 4.55), (7.6, 8.02), (9.04, 9.55), (9.95, 10.45)]:
            arrow(left, y+.55, right, y+.55)
    ax.text(.1, .82, 'Blue: trainable parameters     Grey: frozen parameters (gradients still flow)',
            fontsize=8)
    ax.text(.1, .34, 'Separate fitted encoders; same added features and zero-residual initial function.',
            fontsize=8)
    fig.savefig(directory/'construction.pdf')
    fig.savefig(directory/'construction.png', dpi=200)
    plt.close(fig)


def draw_interactions(directory, cells, data):
    val = lambda k: cells[k]['value']
    # Preserve both new-target studies; do not replace the OpenFold boundary.
    specs = []
    for prefix, label in [('inter', 'OpenFold 96 / ESM2'),
                          ('a_e', 'OpenFold 384 / ESM2'), ('a_c', 'OpenFold 384 / ESMC')]:
        for short, panel in [('c96', 'C96-B'), ('l48', 'L48')]:
            key = f'inter_{short}_interaction' if prefix == 'inter' else f'a_{short}_{prefix[-1]}_ca_lddt_interaction'
            specs.append((label+' / '+panel, key, '#8da0cb', False))
    specs.append(('Protenix 96 / ESM2 / C96-B', 'p96_pair_interaction', '#66c2a5', False))
    for feature in ['e', 'c']:
        for short, panel in [('c96', 'C96-B'), ('l48', 'L48')]:
            specs.append(('Protenix 384 / '+('ESM2' if feature == 'e' else 'ESMC')+' / '+panel,
                          f'dh_protenix_{short}_{feature}_pair_interaction', '#66c2a5', False))
    for feature in ['e', 'c']:
        for short, panel in [('c96', 'C96-B'), ('l48', 'L48')]:
            specs.append(('AtlasFold 96 / '+('ESM2' if feature == 'e' else 'ESMC')+' / '+panel,
                          f'dh_atlas_{short}_{feature}_pair_interaction', '#fc8d62', False))
    specs.append(('OpenFold 96 / ESM2 / Fresh96', 'fresh_ca_lddt_interaction', '#8da0cb', True))
    specs.append(('Protenix 384 / ESMC / Fresh192', 'p192_pair_interaction', '#66c2a5', True))
    assert len(specs) == 17
    # Match the manuscript's 5.5-inch line width: these font sizes survive
    # inclusion unchanged. Distribution and inferential intervals need separate
    # horizontal scales, with exactly aligned rows and shared labels.
    fig = plt.figure(figsize=(5.5, 4.15))
    left = fig.add_axes([.075, .44, .27, .42])
    right = fig.add_axes([.585, .17, .19, .76])
    intervals = fig.add_axes([.825, .17, .165, .76], sharey=right)
    means = [val('dh_protenix_c96_c_'+n) for n in ['native', 'rotated_factor', 'gplus', 'rotated_gplus']]
    # Seaborn Set2 teal/lavender, with lighter partners for trained rotations.
    # Keep the exact palette here so artifact generation needs no new dependency.
    factor_color, generic_color = '#66c2a5', '#8da0cb'
    lighter = lambda color: tuple(.6 * c + .4 for c in to_rgb(color))
    positions = [0, 1.05, 2.3, 3.35]
    left.set_axisbelow(True)
    left.yaxis.grid(True, color='#e8ecf0', linewidth=.55)
    left.bar(positions, means,
             color=[factor_color, lighter(factor_color),
                    generic_color, lighter(generic_color)],
             width=.66, edgecolor='white', linewidth=.55, zorder=3)
    for position, x in zip(positions, means):
        left.text(position, x+.019, f'{x:.4f}', ha='center', va='bottom',
                  fontsize=6.7, color='#374151')
    left.set(xticks=positions, xticklabels=['F', 'RF', 'G+', 'RG+'], ylim=(0, 1),
             ylabel='Mean pair-lDDT')
    left.set_ylabel('Mean pair-lDDT', fontsize=7.2, labelpad=2)
    left.tick_params(labelsize=7.2, length=0)
    left.tick_params(axis='y', labelcolor='#66717e', pad=4)
    left.spines['left'].set_visible(False)
    left.spines['bottom'].set_color('#b9c2cc')
    left.spines['bottom'].set_linewidth(.65)
    left.set_title('Four-cell example\nProtenix 384 / ESMC / C96-B', fontsize=7.2)
    fig.text(.04, .30, r'$\Psi=(F-RF)-(G^+-RG^+)$'+'\n'+r'$\quad=(F-G^+)-(RF-RG^+)$', fontsize=7.5)
    fig.text(.04, .20, 'The cross-head gap remains\nafter rotation; its change is '+r'$\Psi$'+'.', fontsize=7)
    distributions = []
    for i, (label, key, color, fresh) in enumerate(specs):
        y = len(specs)-1-i
        if fresh:
            right.axhspan(y-.46, y+.46, color='#eef2f5', zorder=0)
            intervals.axhspan(y-.46, y+.46, color='#eef2f5', zorder=0)
        # Each sample is one target after averaging its paired seeds/rotations.
        # Never use bootstrap replicates or individual fits as violin samples.
        cell = cells[key]
        obj = data[cell['source'].rsplit('/', 1)[-1].removesuffix('.json')]
        assert cell['field_path'][-1] == 'mean'
        for part in cell['field_path'][:-1]:
            obj = obj[part]
        values = np.asarray(obj['per_target'], dtype=float)
        assert values.shape == (192 if label.endswith('/ Fresh192') else 48 if label.endswith('/ L48') else 96,)
        assert np.isfinite(values).all() and abs(values.mean()-val(key)) < 1e-12
        distributions.append(values)
        if np.ptp(values) > 0:
            violin = right.violinplot(values, positions=[y], vert=False, widths=.68,
                                     showmeans=False, showmedians=False, showextrema=False,
                                     points=160, bw_method=.35)
            for body in violin['bodies']:
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(.65)
                body.set_linewidth(.55)
                body.set_zorder(2)
        else:
            right.vlines(values[0], y-.34, y+.34, color=color, lw=.7, zorder=2)
        m, lo, hi = val(key), val(key+'Lo'), val(key+'Hi')
        intervals.errorbar(m, y, xerr=[[m-lo], [hi-m]], fmt='D' if fresh else 'o',
                           color=color, markeredgecolor='#354052', markeredgewidth=.45,
                           markersize=3, capsize=1.5, linewidth=1, zorder=4)
        assert -.02 <= lo <= m <= hi <= .065
    for axis in [right, intervals]:
        axis.axvline(0, color='#8f99a5', linewidth=.65, linestyle='--', zorder=1)
        for boundary in [10.5, 5.5, 1.5]:
            axis.axhline(boundary, color='#d4dae1', linewidth=.55, linestyle=':', zorder=1)
        axis.tick_params(axis='x', labelsize=7, colors='#66717e', length=3)
        axis.spines['left'].set_visible(False)
        axis.spines['bottom'].set_color('#b9c2cc')
        axis.spines['bottom'].set_linewidth(.65)
    all_values = np.concatenate(distributions)
    lower = np.floor(all_values.min()/.05)*.05-.01
    upper = np.ceil(all_values.max()/.05)*.05+.01
    labels = [s[0].replace('OpenFold', 'OF').replace('Protenix', 'P').replace('AtlasFold', 'A')
              for s in reversed(specs)]
    right.set(yticks=range(len(specs)), yticklabels=labels,
              ylim=(-.65, len(specs)-.35), xlim=(lower, upper),
              xticks=[-.2, 0, .2])
    right.set_xlabel(r'Target effects $\Psi_i$', fontsize=7.2, labelpad=2)
    right.tick_params(axis='y', labelsize=7.2, length=0, pad=4)
    intervals.set(xlim=(-.02, .065), xticks=[0, .03, .06])
    intervals.set_xlabel(r'Mean $\Psi$', fontsize=7.2, labelpad=2)
    intervals.tick_params(axis='y', left=False, labelleft=False)
    right.set_title('Target distribution', fontsize=7.2, pad=7)
    intervals.set_title('Mean + 95% CI', fontsize=7.2, pad=7)
    fig.text(.99, .047, 'OF: OpenFold; P: Protenix; A: AtlasFold.', ha='right', fontsize=6.5)
    fig.text(.99, .017, 'Shaded diamonds: new targets. Distribution and mean axes use different scales.',
             ha='right', fontsize=6.3)
    fig.savefig(directory/'interaction.pdf')
    fig.savefig(directory/'interaction.png', dpi=200)
    plt.close(fig)
