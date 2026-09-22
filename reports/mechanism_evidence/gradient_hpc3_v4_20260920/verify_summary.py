"""CPU-only reconstruction from compact archived scalar records; no new inference."""
from pathlib import Path
import json
import numpy as np
R=Path(__file__).resolve().parent
load=lambda n:json.loads((R/n).read_text())
lock=load('execution_lock.json');core=load('core_conditions.json');e3=load('e3_conditions.json')
a=load('analysis.json');b=load('e3_analysis.json')
seeds=[20260923,20260924,20260925]
rotations=set(lock['rotation_seeds'])
idx={(d['target_id'],d['noise_seed']):d for d in core}
assert len(core)==len(idx)==864
assert set(idx)=={(t,n) for t in lock['observed_ids'] for n in lock['noise_seeds']}
fields=['dynamic_equal_decrease','fixed_equal_decrease','dynamic_actual_loss_decrease','q_learned','norm_ratio']
chains=[];residual_error=0.
for target in lock['observed_ids']:
 ns=[]
 for noise in lock['noise_seeds']:
  d=idx[target,noise];assert d['passed'] and len(d['students'])==12
  row={}
  for st in d['students']:
   assert st['applicable']
   actual=d['baseline']-st['actual_loss']
   equal=(d['baseline']-st['equal_loss'])/(.001*d['query_norm']*d['gradient_norm'])
   residual_error=max(residual_error,abs(actual-st['dynamic_actual_loss_decrease']),abs(equal-st['dynamic_equal_decrease']))
  for f in fields:
   n=[];rr=[]
   for seed in seeds:
    group=[x for x in d['students'] if f'_s{seed}_' in x['model']]
    assert len(group)==4 and {x['rotation'] for x in group}==rotations
    n.append(next(x[f] for x in group if x['rotation'] is None))
    rr.append(np.mean([x[f] for x in group if x['rotation'] is not None]))
   for tag,val in [('native',np.mean(n)),('rotated',np.mean(rr)),('difference',np.mean(np.array(n)-rr))]:row[f+'_'+tag]=float(val)
  for key,f in [('a','a'),('oracle_D','D')]:
   p=d['projections'];assert len(p)==4 and {x['rotation'] for x in p}==rotations
   row[key+'_difference']=next(x[f] for x in p if x['rotation'] is None)-float(np.mean([x[f] for x in p if x['rotation'] is not None]))
  ns.append(row)
 chains.append({'target_id':target,**{f:float(np.mean([n[f] for n in ns])) for f in ns[0]}})
assert residual_error<1e-12

def verify(chains,summary):
 saved={x['target_id']:x for x in summary['chains']};out={}
 for f in chains[0]:
  if f=='target_id':continue
  x=np.array([v[f] for v in chains]);expected=np.array([saved[v['target_id']][f] for v in chains])
  assert np.max(abs(x-expected))<1e-12
  samples=np.random.default_rng(lock['bootstrap_seed']).integers(0,len(x),(lock['bootstrap_draws'],len(x)))
  ci=np.quantile(x[samples].mean(1),[.025,.975]);m=summary['metrics'][f]
  assert abs(x.mean()-m['mean'])<1e-12 and np.max(abs(ci-m['ci95']))<1e-12
  assert int((x>0).sum())==m['positive']
  out[f]={'mean':float(x.mean()),'ci95':ci.tolist(),'positive':int((x>0).sum())}
 return out
core_result=verify(chains,a)
keys={(d['target_id'],d['noise_seed'],d['model']) for d in e3};assert len(keys)==len(e3)==864
inventory=load('checkpoint_inventory.json');e3models={k for k,v in inventory.items() if v['experiment']=='E3'}
assert keys=={(t,n,m) for t in lock['e3_targets'] for n in lock['noise_seeds'][:3] for m in e3models}
ec=[]
for t in lock['e3_targets']:
 rows=[d for d in e3 if d['target_id']==t];assert len(rows)==36 and all(d['passed'] for d in rows)
 assert all(abs(d['removed_loss']-d['baseline_loss']-d['retaining_effect'])<1e-12 for d in rows)
 row={'target_id':t}
 for f in ['retaining_effect','norm_ratio']:
  n=np.mean([d[f] for d in rows if d['rotation'] is None]);r=np.mean([d[f] for d in rows if d['rotation'] is not None]);row.update({f+'_native':float(n),f+'_rotated':float(r),f+'_difference':float(n-r)})
 ec.append(row)
e3_result=verify(ec,b)
out={'passed':True,'core_conditions':len(core),'student_cases':sum(len(x['students']) for x in core),'e3_conditions':len(e3),'raw_scalar_reconstruction_error_max':residual_error,'absolute_tolerance':1e-12,'bootstrap_seed':lock['bootstrap_seed'],'bootstrap_draws':lock['bootstrap_draws'],'core_metrics':core_result,'e3_metrics':e3_result,'scope':'Archived scalar arithmetic and target-bootstrap reconstruction only; no GPU execution, checkpoint-file rehash, or CIF scoring; correlation intervals not recomputed.'}
(R/'independent_verification.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ['core_metrics','e3_metrics']},indent=2))
