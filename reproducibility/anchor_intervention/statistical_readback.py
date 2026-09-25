"""Independent readback of all predeclared E3 target-level contrasts."""
import csv,datetime,hashlib,json
from pathlib import Path
import numpy as np

r=Path(__file__).resolve().parents[1]
read=lambda p:json.loads(p.read_text())
reported=read(r/'analysis/analysis.json');rows=read(r/'analysis/combined_metric_records.json')
lock=read(r/'execution_lock.json');ids=reported['targets'];seeds=[20260923,20260924,20260925]
rotations=['r20270107','r20270104'];assert lock['selected_rotations']==rotations
assert len(ids)==len(set(ids))==96
v={(x['system'],x['target_id']):x for x in rows};assert len(v)==len(rows)==1824
assert all(x['status']=='ok' for x in rows)
sample=np.random.default_rng(20260924).integers(0,96,size=(20000,96))
errors=[];contrasts=[];out_metrics={};target_rows=[]

def score(name,metric):return np.array([v[name,t][metric]for t in ids])
def verify(arr,saved,label):
    target=arr.sum(axis=0)/3
    boot=target[sample].mean(axis=1)
    result=dict(mean=float(target.mean()),ci95=np.percentile(boot,[2.5,97.5]).tolist(),per_target=target.tolist(),per_seed=arr.mean(axis=1).tolist(),positive_targets=int(np.count_nonzero(target>0)))
    for k in ['mean','ci95','per_target','per_seed']:
        err=float(np.max(np.abs(np.asarray(result[k])-np.asarray(saved[k]))));errors.append(err)
        assert err<1e-14,(label,k,err)
    assert result['positive_targets']==saved['positive_targets']
    contrasts.append(label)
    return result

for metric in ['ca_lddt','residue_ca_lddt','tm_score_fixed_full_length']:
    m=reported['metrics'][metric]
    N=np.vstack([score('native_s'+str(s),metric)for s in seeds])
    Nq=np.vstack([score('I_Q_s'+str(s),metric)for s in seeds])
    R=np.stack([np.vstack([score(rid+'_s'+str(s),metric)for s in seeds])for rid in rotations])
    Rq=np.stack([np.vstack([score(rid+'_Q_s'+str(s),metric)for s in seeds])for rid in rotations])
    Q=score('query',metric)
    live_gap=N[None]-R;q_gap=Nq[None]-Rq
    TA=q_gap-live_gap;BR=R-Rq;NC=Nq-N
    assert np.max(np.abs(TA-(BR+NC[None])))<3e-16
    result={}
    for name,array in [('T_A',TA.mean(axis=0)),('B_R_live_minus_Q',BR.mean(axis=0)),('native_Q_minus_live',NC),('native_Q_minus_query',Nq-Q)]:
        result[name]=verify(array,m[name],metric+'/'+name)
    result['absolute']=dict(query=float(Q.mean()),native_live=float(N.mean()),native_Q=float(Nq.mean()),rotated_live=float(R.mean()),rotated_Q=float(Rq.mean()),live_native_minus_rotated=float(live_gap.mean()),Q_native_minus_rotated=float(q_gap.mean()))
    for k in ['query','native_live','native_Q']:assert abs(result['absolute'][k]-m[k])<1e-14
    result['per_rotation']={}
    for j,rid in enumerate(rotations):
        detail={}
        for name,array in [('T_A',TA[j]),('B_R_live_minus_Q',BR[j]),('live_gap',live_gap[j]),('Q_gap',q_gap[j]),('rotated_Q_minus_query',Rq[j]-Q)]:
            detail[name]=verify(array,m['per_rotation'][rid][name],metric+'/'+rid+'/'+name)
        detail.update(rotated_live=float(R[j].mean()),rotated_Q=float(Rq[j].mean()))
        for k in ['rotated_live','rotated_Q']:assert abs(detail[k]-m['per_rotation'][rid][k])<1e-14
        result['per_rotation'][rid]=detail
    out_metrics[metric]=result
    for i,tid in enumerate(ids):
        target_rows.append(dict(metric=metric,target_id=tid,native_live=float(N[:,i].mean()),native_Q=float(Nq[:,i].mean()),rotated_live=float(R[:,:,i].mean()),rotated_Q=float(Rq[:,:,i].mean()),query=float(Q[i]),T_A=float(TA[:,:,i].mean()),B_R_live_minus_Q=float(BR[:,:,i].mean()),native_Q_minus_live=float(NC[:,i].mean())))

out=r/'audit_20260925';out.mkdir(exist_ok=True)
with(out/'per_target.csv').open('w')as f:
    w=csv.DictWriter(f,fieldnames=list(target_rows[0]));w.writeheader();w.writerows(target_rows)
audit=dict(passed=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),input_records=len(rows),unique_targets=96,seed_count=3,rotation_count=2,bootstrap_draws=20000,bootstrap_seed=20260924,verified_contrasts=len(contrasts),maxabs=max(errors),contrasts=contrasts,metrics=out_metrics,input_sha256={name:hashlib.sha256((r/'analysis'/name).read_bytes()).hexdigest()for name in ['analysis.json','combined_metric_records.json']},scientific_changes=False)
(out/'statistical_readback.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in audit.items()if k not in ['metrics','contrasts']},indent=2))
