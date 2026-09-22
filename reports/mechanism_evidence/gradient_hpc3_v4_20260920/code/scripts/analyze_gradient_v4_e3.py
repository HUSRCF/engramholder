"""Paired trained-context removal effects; previously observed targets."""
import argparse,json
from pathlib import Path
import numpy as np
from analyze_gradient_v3 import interval
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--lock',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();lock=json.loads(a.lock.read_text());records={}
for path in a.input.glob('**/*.json'):
 d=json.loads(path.read_text())
 if 'target_id' not in d:continue
 key=(d['target_id'],d['noise_seed'],d['model']);assert key not in records;records[key]=d
chains=[];fail=[]
for target in lock['e3_targets']:
 rows=[v for k,v in records.items() if k[0]==target]
 if len(rows)!=36 or not all(x.get('passed') for x in rows):fail.append({'target_id':target,'records':len(rows),'errors':[x for x in rows if not x.get('passed')]});continue
 metrics={}
 for field in ['retaining_effect','norm_ratio']:
  native=[v[field] for v in rows if v['rotation'] is None];rot=[v[field] for v in rows if v['rotation'] is not None]
  metrics[field+'_native']=float(np.mean(native));metrics[field+'_rotated']=float(np.mean(rot));metrics[field+'_difference']=metrics[field+'_native']-metrics[field+'_rotated']
 for kind in ['baseline','removed']:
  native=[v[kind]['dynamic_loss'] for v in rows if v['rotation'] is None];rot=[v[kind]['dynamic_loss'] for v in rows if v['rotation'] is not None]
  metrics[kind+'_native_loss']=float(np.mean(native));metrics[kind+'_rotated_loss']=float(np.mean(rot))
 chains.append({'target_id':target,**metrics})
d={'complete':len(chains)==len(lock['e3_targets']),'expected_cases':864,'completed_cases':len(records),'failures':fail,'chains':chains,'metrics':{k:interval([c[k] for c in chains]) for k in chains[0] if k!='target_id'} if chains else {},'scope':'models have different learned preceding states; not a pure direction intervention'}
a.output.write_text(json.dumps(d,indent=2)+'\n');print(d['completed_cases'],d['complete'])
