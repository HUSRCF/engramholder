"""Prespecified within-chain averaging; no noise/model pseudo-replication."""
import argparse,json,re
from pathlib import Path
import numpy as np
from analyze_gradient_v3 import collect,interval,correlations


def summarize(records,lock):
 chains=[];fail=[];seed_means={s:[] for s in [20260923,20260924,20260925]};fd={}
 for target in lock['observed_ids']:
  noises=[]
  for noise in lock['noise_seeds']:
   r=records.get((target,noise))
   if not r or not r.get('passed'):
    fail.append({'target_id':target,'noise_seed':noise,'status':r.get('status') if r else 'missing'});continue
   p={x['rotation']:x for x in r['projections']};e={k:next(x for x in v['interventions'] if x['relative_norm']==.001) for k,v in p.items()};entry={'noise_seed':noise}
   for name,vals in [('a',{k:v['shrinkage_alignment'] for k,v in p.items()}),('oracle_D',{k:v['dynamic_normalized_decrease'] for k,v in e.items()})]:
    entry[name+'_native']=vals[None];entry[name+'_rotated']=float(np.mean([vals[k] for k in lock['rotation_seeds'] if k is not None]));entry[name+'_difference']=entry[name+'_native']-entry[name+'_rotated']
   for pr in p.values():
    for it in pr['interventions']:
     key=str(it['relative_norm']);b=fd.setdefault(key,{'n':0,'fixed_pass':0,'dynamic_pass':0,'branch_changes':0});b['n']+=1;b['fixed_pass']+=it['fixed_fd_relative_error']<=.05;b['dynamic_pass']+=it['dynamic_fd_relative_error']<=.05;b['branch_changes']+=it['plus']['dynamic_permutation_changed'] or it['minus']['dynamic_permutation_changed']
   for comp in ['denoising','distogram']:
    vals={x['rotation']:x['shrinkage_alignment'] for x in r['components'] if x['component']==comp};entry[comp+'_a_difference']=vals[None]-float(np.mean([vals[k] for k in lock['rotation_seeds'] if k is not None]))
   for field in ['dynamic_equal_decrease','fixed_equal_decrease','dynamic_actual_loss_decrease','q_learned','norm_ratio']:
    diffs=[];natives=[];rotated=[]
    for seed in seed_means:
     rows=[x for x in r['students'] if f'_s{seed}_' in x['model']];assert len(rows)==4
     nv=next(x[field] for x in rows if x['rotation'] is None);rv=float(np.mean([x[field] for x in rows if x['rotation'] is not None]));diffs.append(nv-rv);natives.append(nv);rotated.append(rv)
     if field=='dynamic_equal_decrease':entry[f'seed_{seed}_difference']=nv-rv
    entry[field+'_difference']=float(np.mean(diffs));entry[field+'_native']=float(np.mean(natives));entry[field+'_rotated']=float(np.mean(rotated))
   noises.append(entry)
  if len(noises)==len(lock['noise_seeds']):
   row={'target_id':target,**{k:float(np.mean([v[k] for v in noises])) for k in noises[0] if k!='noise_seed'}};chains.append(row)
 for row in chains:
  for seed in seed_means:seed_means[seed].append(row[f'seed_{seed}_difference'])
 metrics={k:interval([r[k] for r in chains]) for k in chains[0] if k!='target_id'} if chains else {}
 return {'complete':len(chains)==len(lock['observed_ids']),'expected_chains':len(lock['observed_ids']),'complete_chains':len(chains),'failures':fail,'metrics':metrics,'seed_marginals':{str(k):interval(v) for k,v in seed_means.items()},'fd':fd,'chains':chains}


def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--lock',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--structure-records',type=Path);a=p.parse_args();lock=json.loads(a.lock.read_text());d=summarize(collect(a.input),lock)
 if d['complete'] and a.structure_records:
  # The legacy correlation helper uses fixed field names, not selected metrics.
  rows=[{**r,'dynamic_D_0.001_difference':r['dynamic_equal_decrease_difference']} for r in d['chains']]
  d['correlations']=correlations(rows,a.structure_records)
  d['correlations']['learned_equal_norm_difference']=d['correlations'].pop('dynamic_D_0.001_difference')
 d['scope']='observed-panel offline diagnostic; seeds/noises/rotations averaged within chain'
 a.output.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');print(d['complete_chains'],d['metrics'].get('dynamic_equal_decrease_difference'))
if __name__=='__main__':main()
