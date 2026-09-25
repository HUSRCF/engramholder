import argparse,datetime
from pathlib import Path
import numpy as np
from e3_common import *
import scoring
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--mode',choices=['engineering','references','score'],required=True);a=p.parse_args();r=a.root
c,old,parent,lk=contract(r);stamp=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
if a.mode=='engineering':
 items={}
 for i in range(2):
  path=r/'engineering'/f'smoke_{i}.json';x=read(path);assert x['passed']and x['execution_lock_sha256']==lk and len(x['checks'])==3
  items[str(i)]=sha(path)
 write(r/'engineering/complete.json',dict(passed=True,execution_lock_sha256=lk,inputs=items,utc=stamp()));raise SystemExit
if a.mode=='references':
 rows=[]
 for i in range(16):
  x=read(r/'references'/f'shard_{i}.json');assert x['complete']and x['execution_lock_sha256']==lk;rows+=x['records']
 keys={(x['phase'],x['target_id'])for x in rows};expected={(phase,t['target_id'])for phase in ['train','eval']for t in target_list(parent,phase)}
 assert keys==expected and len(rows)==192
 for rec in rows:
  assert rec['two_seed_exact']and rec['rng_preserved']
  path=r/'references'/rec['phase']/(rec['target_id']+'.pt');assert sha(path)==rec['sha256']
 write(r/'references/complete.json',dict(complete=True,execution_lock_sha256=lk,utc=stamp(),unique_caches=192,reference_forward_trajectories=384,records=rows));raise SystemExit
out=r/'analysis';out.mkdir(exist_ok=True)
if(out/'complete.json').exists():
 done=read(out/'complete.json');assert done['complete']and done['execution_lock_sha256']==lk and sha(out/'analysis.json')==done['analysis_sha256'];raise SystemExit('ALREADY_COMPLETE')
targets=target_list(parent,'eval');ids=[t['target_id']for t in targets];jobs=[];audits=[]
for run in c['runs']:
 folder=r/'formal'/run['name'];done=read(folder/'complete.json');tr=read(folder/'training_complete.json')
 assert done['complete']and done['execution_lock_sha256']==lk and tr['steps']==1536
 assert sha(folder/'checkpoint_1536.pt')==tr['checkpoint_sha256']
 logs=[__import__('json').loads(line)for line in(folder/'training.jsonl').read_text().splitlines()]
 assert [x['step']for x in logs]==list(range(1,1537))
 assert all(x['forward_hook_calls']==4 and x['backward_hook_replays']==1 and np.isfinite(x['loss'])for x in logs)
 assert all(x['reference_replay_exact']for x in logs if x['step']in[384,768,1536])
 assert [x['target_id']for x in done['records']]==ids
 jobs.extend((run['name'],t,rec,folder/'predictions'/f"{t['target_id']}.npz")for t,rec in zip(targets,done['records'],strict=True))
 audits.append(dict(run=run,checkpoint_sha256=tr['checkpoint_sha256'],steps=1536,reference_replays=3,training_seconds=sum(x['seconds']for x in logs),zero_gradient_steps=sum(bool(x['zero_gradient_parameters'])for x in logs),training_residual_norm_by_round=np.mean([[y['residual_norm']for y in x['injections']]for x in logs],axis=0).tolist()))
scoring.r=parent;rows=scoring.score_jobs(jobs,out,True);assert len(rows)==864
original=read(parent/'e1/analysis/metric_records.json');names={'query'}|{f'native_s{s}'for s in old['formal_seeds']}|{f'{rid}_s{s}'for rid in c['selected_rotations']for s in old['formal_seeds']}
oldrows=[x for x in original if x['system']in names];assert len(oldrows)==960
combined=rows+oldrows;values={(x['system'],x['target_id']):x for x in combined};assert len(values)==1824
write(out/'combined_metric_records.json',combined)
draw=np.random.default_rng(20260924).integers(96,size=(20000,96))
def vec(name,metric):return np.array([values[name,tid][metric]for tid in ids])
def summary(x):
 di=np.asarray(x).mean(0);return dict(mean=float(di.mean()),ci95=np.quantile(di[draw].mean(1),[.025,.975]).tolist(),per_target=di.tolist(),per_seed=np.asarray(x).mean(-1).tolist(),positive_targets=int((di>0).sum()))
result=dict(execution_lock_sha256=lk,utc=stamp(),stage='E3',scope='Independent anchor-history intervention on two previously X-selected rotations and observed Confirm96-B; not new-target confirmation',targets=ids,runs=c['runs'],new_failures=sum(x['status']!='ok'for x in rows),metrics={},runtime_audit=audits)
for metric in ['ca_lddt','residue_ca_lddt','tm_score_fixed_full_length']:
 N=np.stack([vec(f'native_s{s}',metric)for s in old['formal_seeds']]);Q=vec('query',metric)
 Nq=np.stack([vec(f'I_Q_s{s}',metric)for s in old['formal_seeds']])
 R=np.stack([[vec(f'{rid}_s{s}',metric)for s in old['formal_seeds']]for rid in c['selected_rotations']])
 Rq=np.stack([[vec(f'{rid}_Q_s{s}',metric)for s in old['formal_seeds']]for rid in c['selected_rotations']])
 ta=(Nq[None]-Rq)-(N[None]-R);br=R-Rq;nc=Nq-N
 assert np.allclose(ta,br+nc[None],rtol=0,atol=3e-16)
 result['metrics'][metric]=dict(T_A=summary(ta.mean(0)),B_R_live_minus_Q=summary(br.mean(0)),native_Q_minus_live=summary(nc),native_live=float(N.mean()),native_Q=float(Nq.mean()),query=float(Q.mean()),native_Q_minus_query=summary(Nq-Q),per_rotation={rid:dict(T_A=summary(ta[j]),B_R_live_minus_Q=summary(br[j]),rotated_live=float(R[j].mean()),rotated_Q=float(Rq[j].mean()),live_gap=summary(N-R[j]),Q_gap=summary(Nq-Rq[j]),rotated_Q_minus_query=summary(Rq[j]-Q))for j,rid in enumerate(c['selected_rotations'])})
write(out/'analysis.json',result);m=result['metrics']['ca_lddt']
lines=['# E3 query-only anchor 干预结果','','全9模型、864正式尝试完成后统一评分；无E2补偿器C；现场基线与完整recycle保留。','','|比较|均值|95%目标条件区间|','|---|---:|---|']
for name in ['T_A','B_R_live_minus_Q','native_Q_minus_live']:
 x=m[name];lines.append(f"|{name}|{x['mean']:+.8f}|{x['ci95']}|")
lines+=['',f"Query={m['query']:.8f}，Native现场={m['native_live']:.8f}，Native Q anchor={m['native_Q']:.8f}。"]
for rid,x in m['per_rotation'].items():lines.append(f"- {rid}：Rotated现场={x['rotated_live']:.8f}，Q anchor={x['rotated_Q']:.8f}；T_A={x['T_A']['mean']:+.8f}；B_R={x['B_R_live_minus_Q']['mean']:+.8f}。")
lines+=['','T_A正且B_R正并获区间支持，才支持现场anchor缓和旋转代价且帮助旋转臂；仅Native变化不够。方向相反原样报告。','已观察Confirm96-B、两个预选旋转的有限预算干预；不恢复E1预测、不替代Fresh96四格确认。','三指标、逐目标/种子、完整绝对分数见analysis.json；所有失败保留，未扩目标、旋转或训练。']
(out/'results.md').write_text('\n'.join(lines)+'\n')
write(out/'complete.json',dict(complete=True,execution_lock_sha256=lk,utc=stamp(),new_predictions=864,new_failures=result['new_failures'],analysis_sha256=sha(out/'analysis.json'),records_sha256=sha(out/'metric_records.json'),combined_sha256=sha(out/'combined_metric_records.json')))
