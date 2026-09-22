"""Independent target-paired recomputation from the 720 scored record rows."""
from pathlib import Path
import hashlib,json
import numpy as np
r=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows=read(r/'analysis/metric_records.json');reported=read(r/'analysis/analysis.json');lock=read(r/'remote_v3/execution_lock.json');targets=read(r/'panels/scoring24.json')['targets'];remote=read(r/'remote_v3/completion_review.json')
assert len(rows)==720 and all(x['status']=='ok' for x in rows)
assert sha(r/'analysis/metric_records.json')==remote['scored_records_sha256'] and sha(r/'analysis/analysis.json')==remote['analysis_sha256']
ids=[x['target_id'] for x in targets];rng=np.random.default_rng(20260926)
draw=np.concatenate([rng.choice([i for i,t in enumerate(targets) if t['length_bin']==b],size=(20000,6)) for b in ['128-191','192-255','256-319','320-384']],axis=1)
max_error=0.;checked=0
for metric in ['ca_lddt','residue_ca_lddt','tm_score']:
 lookup={(x['system'],x['target_id']):x[metric] for x in rows};assert len(lookup)==720
 arrays={k:np.empty((3,3,24)) for k in ['NN','NR','RN','RR','D_mN','D_mR','A_dN','A_dR','Edir','Eamp','I','NN_minus_RR']}
 for si,s in enumerate(lock['seeds']):
  for ri,rot in enumerate(lock['rotations']):
   for ti,tid in enumerate(ids):
    nn=lookup[f'NN_s{s}',tid];nr=lookup[f'NR_r{rot}_s{s}',tid];rn=lookup[f'RN_r{rot}_s{s}',tid];rr=lookup[f'RR_r{rot}_s{s}',tid]
    v=dict(NN=nn,NR=nr,RN=rn,RR=rr,D_mN=nn-rn,D_mR=nr-rr,A_dN=nn-nr,A_dR=rn-rr,Edir=(nn-rn+nr-rr)/2,Eamp=(nn-nr+rn-rr)/2,I=nn-rn-nr+rr,NN_minus_RR=nn-rr)
    for k,value in v.items():arrays[k][si,ri,ti]=value
 for k,v in arrays.items():
  got=reported['metrics'][metric]['cells' if k in ['NN','NR','RN','RR'] else 'contrasts'][k]
  target=v.mean((0,1));calculated=dict(mean=target.mean(),per_target=target,per_seed=v.mean((1,2)),per_rotation=v.mean((0,2)),ci95=np.quantile(target[draw].mean(1),[.025,.975]))
  for key,value in calculated.items():
   err=float(np.max(np.abs(np.array(value)-np.array(got[key]))));max_error=max(max_error,err);assert err<1e-12,(metric,k,key,err)
  assert int((target>0).sum())==got['positive_targets'];checked+=1
 np.testing.assert_allclose(arrays['Edir']+arrays['Eamp'],arrays['NN_minus_RR'],atol=1e-14)
result=dict(passed=True,rows=720,contrasts_and_cells_checked=checked,max_absolute_error=max_error,bootstrap='20000 draws, 4 strata x6 targets',conditions='3seeds and3rotations aggregated within24 targets, not independent replicates',remote_completion_audit_passed=remote['passed'])
(r/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
