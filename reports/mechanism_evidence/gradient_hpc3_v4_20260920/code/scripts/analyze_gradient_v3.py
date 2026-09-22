"""Chain-level analysis of fixed cohorts; failed conditions remain explicit."""
import argparse,json
from pathlib import Path
import numpy as np


def interval(values):
    x=np.asarray(values,dtype=float)
    if not len(x):return {'n':0,'mean':None,'ci95':None}
    rng=np.random.default_rng(20261204)
    means=x[rng.integers(0,len(x),(20000,len(x)))].mean(axis=1)
    return {'n':len(x),'mean':float(x.mean()),'ci95':np.quantile(means,[.025,.975]).tolist(),'positive':int((x>0).sum())}


def collect(root):
    records={}
    for f in sorted(root.glob('**/*.json')):
        d=json.loads(f.read_text())
        if 'target_id' not in d or 'noise_seed' not in d or 'status' not in d:continue
        key=(d['target_id'],d['noise_seed'])
        if key in records:raise ValueError(f'duplicate case {key}')
        records[key]=d
    return records


def analyze(records,ids,seeds):
    chain=[];fail=[];checks={'conditions':0,'branch_changed_interventions':0,'interventions':0,'dynamic_fd_within_5pct':0,'fixed_fd_within_5pct':0}
    by_beta={str(b):{'interventions':0,'branch_changes':0,'dynamic_fd_within_5pct':0,'fixed_fd_within_5pct':0} for b in [1e-4,1e-3,1e-2]}
    for name in ids:
        noises=[]
        for seed in seeds:
            row=records.get((name,seed))
            if row is None or not row.get('passed'):
                fail.append({'target_id':name,'noise_seed':seed,'status':'missing' if row is None else row.get('status')});continue
            p={r['rotation']:r for r in row['projections']}
            if set(p)!={None,20261001,20261002,20261003}:raise ValueError('incomplete paired spaces')
            checks['conditions']+=1
            for item in p.values():
                for effect in item['interventions']:
                    checks['interventions']+=1
                    checks['branch_changed_interventions']+=int(effect['plus']['dynamic_permutation_changed'] or effect['minus']['dynamic_permutation_changed'])
                    checks['dynamic_fd_within_5pct']+=int(effect['dynamic_fd_relative_error']<=.05)
                    checks['fixed_fd_within_5pct']+=int(effect['fixed_fd_relative_error']<=.05)
                    bucket=by_beta[str(effect['relative_norm'])]
                    bucket['interventions']+=1
                    bucket['branch_changes']+=int(effect['plus']['dynamic_permutation_changed'] or effect['minus']['dynamic_permutation_changed'])
                    bucket['dynamic_fd_within_5pct']+=int(effect['dynamic_fd_relative_error']<=.05)
                    bucket['fixed_fd_within_5pct']+=int(effect['fixed_fd_relative_error']<=.05)
            entry={'noise_seed':seed,'sigma':row['baseline']['sigma']}
            for kind,field in [('a','shrinkage_alignment'),('b','projected_energy_fraction')]:
                entry[kind+'_native']=p[None][field]
                entry[kind+'_rotated']=float(np.mean([p[k][field] for k in [20261001,20261002,20261003]]))
                entry[kind+'_difference']=entry[kind+'_native']-entry[kind+'_rotated']
            for beta in [1e-4,1e-3,1e-2]:
                effects={k:next(e for e in item['interventions'] if e['relative_norm']==beta) for k,item in p.items()}
                for mode in ['fixed','dynamic']:
                    field=mode+'_normalized_decrease';prefix=f'{mode}_D_{beta:g}'
                    entry[prefix+'_native']=effects[None][field]
                    entry[prefix+'_rotated']=float(np.mean([effects[k][field] for k in [20261001,20261002,20261003]]))
                    entry[prefix+'_difference']=entry[prefix+'_native']-entry[prefix+'_rotated']
            noises.append(entry)
        if len(noises)==len(seeds):
            item={'target_id':name,'noise_records':noises}
            item.update({k:float(np.mean([r[k] for r in noises])) for k in noises[0] if k not in ['noise_seed','sigma']})
            chain.append(item)
    metrics={k:interval([r[k] for r in chain]) for k in chain[0] if k not in ['target_id','noise_records']} if chain else {}
    return {'expected_chains':len(ids),'complete_chains':len(chain),'complete_panel':len(chain)==len(ids),'failures':fail,'checks':checks,'checks_by_beta':by_beta,'metrics':metrics,'chains':chain,
            'inference_scope':'fixed-cohort offline diagnostic; partial-panel estimates describe complete subset only' if fail else 'fixed-cohort offline diagnostic'}


def correlations(chains,records_file):
    from scipy.stats import rankdata
    d=json.loads(records_file.read_text());seeds=[20260923,20260924,20260925];rotations=[20261001,20261002,20261003]
    names=[f'n24_tangent_native_s{s}_u384' for s in seeds]+[f'n24_tangent_r{r}_s{s}_u384' for s in seeds for r in rotations]
    table={k:{v['target_id']:v['original_ca_lddt'] for v in d[k]} for k in names}
    ids=[c['target_id'] for c in chains]
    y=np.array([np.mean([table[f'n24_tangent_native_s{s}_u384'][i]-np.mean([table[f'n24_tangent_r{r}_s{s}_u384'][i] for r in rotations]) for s in seeds]) for i in ids])
    def rho(a,b):
        ar=rankdata(a,axis=-1);br=rankdata(b,axis=-1);ar-=ar.mean(axis=-1,keepdims=True);br-=br.mean(axis=-1,keepdims=True)
        numerator=(ar*br).sum(axis=-1);denominator=np.sqrt((ar*ar).sum(axis=-1)*(br*br).sum(axis=-1))
        return np.divide(numerator,denominator,out=np.full_like(numerator,np.nan),where=denominator>0)
    out={};rng=np.random.default_rng(20261204);indices=rng.integers(0,len(ids),(20000,len(ids)))
    for field in ['a_difference','dynamic_D_0.001_difference']:
        x=np.array([c[field] for c in chains]);draws=rho(x[indices],y[indices]);valid=draws[np.isfinite(draws)]
        out[field]={'spearman':float(rho(x,y)),'ci95':np.quantile(valid,[.025,.975]).tolist(),'finite_bootstrap_draws':len(valid),'n':len(ids),'scope':'observed-panel association, not independent predictive validation'}
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--lock',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--cohort',choices=['old16','observed96'],required=True);p.add_argument('--structure-records',type=Path)
    a=p.parse_args();lock=json.loads(a.lock.read_text());records=collect(a.input)
    if a.cohort=='old16':groups={'Train8':lock['old16']['training'],'Dev8':lock['old16']['development'],'Old16':lock['old16']['training']+lock['old16']['development']}
    else:groups={'Observed96':lock['observed_ids']}
    result={'schema':'engramfold.gradient_analysis.v3','groups':{k:analyze(records,v,lock['noise_seeds']) for k,v in groups.items()},'primary_ridge':1e-4,'primary_relative_norm':1e-3,'bootstrap_draws':20000,'bootstrap_unit':'chain','noise_and_rotations_are_not_independent_samples':True}
    if a.structure_records and a.cohort=='observed96' and result['groups']['Observed96']['complete_panel']:
        result['correlations']=correlations(result['groups']['Observed96']['chains'],a.structure_records)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for k,g in result['groups'].items():print(k,g['complete_chains'],{m:g['metrics'].get(m) for m in ['a_difference','dynamic_D_0.001_difference']})
if __name__=='__main__':main()
