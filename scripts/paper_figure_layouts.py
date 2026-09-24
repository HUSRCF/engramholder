"""Vector layouts for the adapter and the complete interaction evidence."""
import matplotlib.pyplot as plt
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


def draw_interactions(directory, cells):
    val = lambda k: cells[k]['value']
    # Keep every completed panel/configuration, including the new-target null result.
    specs = []
    for prefix, label in [('inter', 'OpenFold 96 / ESM2'),
                          ('a_e', 'OpenFold 384 / ESM2'), ('a_c', 'OpenFold 384 / ESMC')]:
        for short, panel in [('c96', 'C96-B'), ('l48', 'L48')]:
            key = f'inter_{short}_interaction' if prefix == 'inter' else f'a_{short}_{prefix[-1]}_ca_lddt_interaction'
            specs.append((label+' / '+panel, key, '#9b5d16', False))
    specs.append(('Protenix 96 / ESM2 / C96-B', 'p96_pair_interaction', '#245a81', False))
    for feature in ['e', 'c']:
        for short, panel in [('c96', 'C96-B'), ('l48', 'L48')]:
            specs.append(('Protenix 384 / '+('ESM2' if feature == 'e' else 'ESMC')+' / '+panel,
                          f'dh_protenix_{short}_{feature}_pair_interaction', '#245a81', False))
    for feature in ['e', 'c']:
        for short, panel in [('c96', 'C96-B'), ('l48', 'L48')]:
            specs.append(('AtlasFold 96 / '+('ESM2' if feature == 'e' else 'ESMC')+' / '+panel,
                          f'dh_atlas_{short}_{feature}_pair_interaction', '#587b46', False))
    specs.append(('OpenFold 96 / ESM2 / Fresh96', 'fresh_ca_lddt_interaction', '#9b5d16', True))
    assert len(specs) == 16
    fig = plt.figure(figsize=(6.6, 4.85))
    left = fig.add_axes([.085, .44, .205, .43])
    right = fig.add_axes([.65, .14, .325, .79])
    means = [val('dh_protenix_c96_c_'+n) for n in ['native', 'rotated_factor', 'gplus', 'rotated_gplus']]
    left.bar(range(4), means, color=['#245a81', '#799cb5', '#9b5d16', '#d4af7b'], width=.7)
    for i, x in enumerate(means):
        left.text(i, x+.018, f'{x:.3f}', ha='center', fontsize=7.5)
    left.set(xticks=range(4), xticklabels=['F', 'RF', 'G+', 'RG+'], ylim=(0, 1),
             ylabel='Mean pair-lDDT')
    left.set_ylabel('Mean pair-lDDT', fontsize=8, labelpad=2)
    left.tick_params(labelsize=8)
    left.set_title('Four-cell example\nProtenix 384 / ESMC / C96-B', fontsize=8)
    fig.text(.06, .30, r'$\Psi=(F-RF)-(G^+-RG^+)$'+'\n'+r'$\quad=(F-G^+)-(RF-RG^+)$', fontsize=9)
    fig.text(.06, .20, 'The cross-head gap remains\nafter rotation; its change is '+r'$\Psi$'+'.', fontsize=8)
    for i, (label, key, color, fresh) in enumerate(specs):
        y = len(specs)-1-i
        if fresh:
            right.axhspan(y-.46, y+.46, color='#eef2f5', zorder=0)
        m, lo, hi = val(key), val(key+'Lo'), val(key+'Hi')
        right.errorbar(m, y, xerr=[[m-lo], [hi-m]], fmt='D' if fresh else 'o',
                       color=color, markersize=4, capsize=2)
    right.axvline(0, color='.5', linewidth=.7)
    right.axhline(.5, color='.6', linewidth=.6, linestyle=':')
    right.set(yticks=range(len(specs)), yticklabels=[s[0] for s in reversed(specs)],
              ylim=(-.65, len(specs)-.35), xlim=(-.02, .065), xticks=[-.02, 0, .02, .04, .06],
              xlabel=r'$\Psi$ (95% target interval)')
    right.tick_params(axis='y', labelsize=7.4, length=0, pad=4)
    right.tick_params(axis='x', labelsize=8)
    right.set_title('Dense-rotation interactions', fontsize=9)
    fig.text(.975, .025, 'Shaded diamond: new-target primary test; other rows: observed panels.',
             ha='right', fontsize=7.5)
    fig.savefig(directory/'interaction.pdf')
    fig.savefig(directory/'interaction.png', dpi=200)
    plt.close(fig)
