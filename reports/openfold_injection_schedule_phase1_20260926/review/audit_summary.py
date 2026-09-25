"""Independent aggregation of completed scores; no new model evaluation."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

r = Path(__file__).resolve().parents[1]
out = r / 'review'
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
done = read(r / 'analysis/complete.json')
assert sha(r / 'analysis/analysis.json') == done['analysis_sha256']
assert sha(r / 'analysis/metric_records.json') == done['scores_sha256']
assert sha(r / 'execution_lock.json') == done['execution_lock_sha256']
rows = read(r / 'analysis/metric_records.json')
old = read(r / 'analysis/analysis.json')
ids = [t['target_id'] for t in read(r / 'inference_manifest.json')]
assert len(ids) == len(set(ids)) == 96 and len(rows) == 14112
assert all(x['status'] == 'ok' for x in rows)
seeds = [20260923, 20260924, 20260925]
rots = [20261001, 20261002, 20261003]
conds = ['single1', 'all2', 'all4', 'first2', 'first4', 'last4']
metrics = ['ca_lddt', 'residue_ca_lddt', 'tm_score_fixed_full_length']
keys = [(x['target_id'], x['condition'], x['system']) for x in rows]
assert len(set(keys)) == len(keys)
lookup = dict(zip(keys, rows))
draw = np.random.default_rng(20260926).integers(96, size=(20000, 96))
errors = []
verified = 0
def check(cube, reference):
    global verified
    # This script builds seed x rotation x target arrays; the original builds
    # rotation x seed x target arrays. All comparisons are independently formed.
    cube = np.broadcast_to(cube, (3, 3, 96))
    targets = cube.mean(axis=1).mean(axis=0)
    values = dict(mean=targets.mean(), ci95=np.percentile(targets[draw].mean(axis=1), [2.5, 97.5]),
                  per_target=targets, per_seed=cube.mean(axis=(1, 2)), per_rotation=cube.mean(axis=(0, 2)),
                  rotation_by_seed=cube.mean(axis=2).T, positive_targets=int((targets > 0).sum()))
    assert reference['target_ids'] == ids
    for key, value in values.items():
        delta = float(np.max(np.abs(np.asarray(value) - np.asarray(reference[key]))))
        errors.append(delta)
        assert delta < 1e-12, (key, delta)
    verified += 1
    return targets

for metric in metrics:
    effects = {}
    for cond in conds:
        def get(system, condition=cond):
            return np.array([lookup[i, condition, system][metric] for i in ids])
        fn = np.array([get(f'native_s{s}') for s in seeds])[:, None, :]
        gn = np.array([get(f'gplus_s{s}') for s in seeds])[:, None, :]
        fr = np.array([[get(f'r{rot}_s{s}') for rot in rots] for s in seeds])
        gr = np.array([[get(f'gplus_r{rot}_s{s}') for rot in rots] for s in seeds])
        query = get('query', 'query' + cond[-1])
        prior = old[metric]['conditions'][cond]
        for key, cube in dict(factor_native=fn, factor_rotated=fr, gplus_native=gn, gplus_rotated=gr).items():
            check(cube, prior['absolute'][key])
            check(cube - query, prior['gain_over_query'][key])
        check(query, prior['query'])
        check(fn - fr, prior['factor_rotation'])
        check(gn - gr, prior['gplus_rotation'])
        check((fn - fr) - (gn - gr), prior['interaction'])
        effects[cond] = (fn-fr, gn-gr)
    for name, a, b in [('D_write', 'all4', 'first4'), ('D_last', 'all4', 'last4'),
                       ('D_timing', 'first4', 'last4'), ('D_prop', 'first4', 'single1'), ('D_depth', 'all4', 'single1')]:
        df = effects[a][0] - effects[b][0]
        dg = effects[a][1] - effects[b][1]
        prior = old[metric]['differences'][name]
        check(df - dg, prior['interaction'])
        check(df, prior['factor_contribution'])
        check(dg, prior['gplus_contribution'])

d = old['ca_lddt']['differences']['D_write']['interaction']
v = np.array(d['per_target'])
audit = dict(verified_summaries=verified, max_absolute_reaggregation_error=max(errors),
             records=len(rows), trajectories=done['trajectories'], failures=done['failed_trajectories'],
             max_independent_pair_score_error=max(x['independent_pair_maxabs'] for x in rows),
             analysis_sha256=done['analysis_sha256'], score_sha256=done['scores_sha256'],
             D_write_descriptive=dict(median=float(np.median(v)), sd=float(v.std(ddof=1)),
                  quantiles_0_25_50_75_100=np.quantile(v, [0, .25, .5, .75, 1]).tolist(),
                  positive=int((v>0).sum()), negative=int((v<0).sum()), zero=int((v==0).sum())),
             scope='Reaggregation of existing CIF-derived scores; not a new CIF rescore or independent target confirmation')
(out / 'independent_audit.json').write_text(json.dumps(audit, indent=2) + '\n')

fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
colors = ['#235789', '#6a9cc3', '#963b39', '#d79180']
names = ['Native Factor', 'Rotated Factor', 'Native G+', 'Rotated G+']
group_keys = ['factor_native', 'factor_rotated', 'gplus_native', 'gplus_rotated']
conditions = old['ca_lddt']['conditions']
x = np.arange(3)
for j, (key, label, color) in enumerate(zip(group_keys, names, colors)):
    y = [conditions[c]['absolute'][key]['mean'] for c in ('all4', 'first4', 'last4')]
    axs[0,0].bar(x+(j-1.5)*.18, y, .18, label=label, color=color)
axs[0,0].axhline(conditions['all4']['query']['mean'], color='black', linestyle='--', label='Query-only')
axs[0,0].set(xticks=x, xticklabels=['All', 'First only', 'Last only'], ylabel='Mean CA pair-lDDT', ylim=(0,.56), title='Fixed trained heads: absolute quality at pass 4')
axs[0,0].legend(fontsize=8, ncol=2)
for j,c in enumerate(conds):
    s=conditions[c]['interaction']; mean=s['mean']; lo,hi=s['ci95']
    axs[0,1].errorbar(j,mean,yerr=[[mean-lo],[hi-mean]],fmt='o',capsize=4,color='#235789')
axs[0,1].axhline(0,color='black',linewidth=.7)
axs[0,1].set(xticks=np.arange(6),xticklabels=['Single1','All2','All4','First2','First4','Last4'],ylabel='Interaction Psi',title='Target-bootstrap 95% intervals')
axs[1,0].hist(v,bins=18,color='#6a9cc3',edgecolor='white')
axs[1,0].axvline(0,color='black',linewidth=1)
axs[1,0].axvline(v.mean(),color='#963b39',linestyle='--',label=f'Mean = {v.mean():+.5f}')
axs[1,0].set(xlabel='Target-level D_write (All4 - First4)',ylabel='Targets',title=f'Primary contrast: {sum(v>0)}/96 positive')
axs[1,0].legend(fontsize=9)
matrix=np.array(d['rotation_by_seed']); limit=abs(matrix).max()
axs[1,1].imshow(matrix,cmap='RdBu',vmin=-limit,vmax=limit)
for i in range(3):
    for j in range(3): axs[1,1].text(j,i,f'{matrix[i,j]:+.4f}',ha='center',va='center',color='white' if abs(matrix[i,j])>.02 else 'black')
axs[1,1].set(xticks=np.arange(3),xticklabels=['Seed 23','Seed 24','Seed 25'],yticks=np.arange(3),yticklabels=['R1','R2','R3'],title='D_write by fitted seed and rotation')
fig.savefig(out/'phase1_summary.pdf')
fig.savefig(out/'phase1_summary.png',dpi=170)
print(json.dumps(audit,indent=2))
