"""All-terminal, fixed-reference scoring and target-stratified four-cell inference."""
import concurrent.futures,csv,time,subprocess
from pathlib import Path
import numpy as np
from .common import *
from engramfold.evaluation.structure import read_atom_site_positions
from engramfold.evaluation.independent import fixed_mask_metrics
from engramfold.evaluation.post_validation import ca_pdb,parse_tm_score,load_reference_lddt
R=Path('/media/PM982/engramfold/runs/protenix_fresh192_20260925')
METRICS=['ca_lddt','residue_ca_lddt','tm_score_fixed_full_length']

def validate_records(rows,targets):
 expected={(m,t['target_id']) for m in models() for t in targets}
 actual=[(x['system'],x['target_id']) for x in rows]
 assert len(actual)==len(set(actual))==4800 and set(actual)==expected
 assert all(x['status'] in ('ok','failed') for x in rows)
 for m in models():assert any(x['status']=='ok' for x in rows if x['system']==m)

def summarize(rows,targets):
 ids=[t['target_id'] for t in targets];bins=['128-191','192-255','256-319','320-384'];rng=np.random.default_rng(20260925)
 groups=[np.array([i for i,t in enumerate(targets) if t['length_bin']==b]) for b in bins];assert [len(g) for g in groups]==[48]*4
 draws=np.concatenate([g[rng.integers(48,size=(20000,48))] for g in groups],axis=1)
 def summary(a):
  # R x seed x target; collapse only after retaining marginal results.
  a=np.asarray(a);v=a.mean(axis=(0,1));boot=v[draws].mean(1)
  return dict(mean=float(v.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),per_target=v.tolist(),positive_targets=int((v>0).sum()),per_rotation=a.mean((1,2)).tolist(),per_seed=a.mean((0,2)).tolist(),rotation_by_seed=a.mean(2).tolist())
 result={}
 for metric in METRICS:
  score={(x['system'],x['target_id']):x[metric] for x in rows}
  def vec(m):return np.array([score[m,i] for i in ids])
  n=np.stack([vec(f'C_factor_native_s{s}') for s in SEEDS]);g=np.stack([vec(f'C_generic_plus_native_s{s}') for s in SEEDS]);q=vec('query')
  fr=np.stack([np.stack([vec(f'C_factor_r{r}_s{s}') for s in SEEDS]) for r in ROTS]);gr=np.stack([np.stack([vec(f'C_generic_plus_r{r}_s{s}') for s in SEEDS]) for r in ROTS])
  df=n[None]-fr;dg=g[None]-gr;psi=df-dg
  result[metric]=dict(system_means={m:float(vec(m).mean()) for m in models()},four_cell_means=dict(query=float(q.mean()),factor_native=float(n.mean()),factor_rotated=float(fr.mean()),gplus_native=float(g.mean()),gplus_rotated=float(gr.mean())),factor_rotation=summary(df),gplus_rotation=summary(dg),interaction=summary(psi),native_minus_gplus=summary((n-g)[None]),native_minus_query=summary((n-q)[None]),gplus_minus_query=summary((g-q)[None]),rotated_factor_minus_query=summary(fr-q),rotated_gplus_minus_query=summary(gr-q))
  # Direct independent assembly of the unique primary vector, not separate p-values.
  independent=np.array([sum((score[f'C_factor_native_s{s}',i]-sum(score[f'C_factor_r{r}_s{s}',i] for r in ROTS)/3)-(score[f'C_generic_plus_native_s{s}',i]-sum(score[f'C_generic_plus_r{r}_s{s}',i] for r in ROTS)/3) for s in SEEDS)/3 for i in ids])
  assert np.max(np.abs(independent-psi.mean((0,1))))<1e-12
  v=psi.mean((0,1));secondary={}
  for name,ix in zip(bins,groups):
   rr=np.random.default_rng(20260925);bb=ix[rr.integers(48,size=(20000,48))]
   secondary[name]=dict(n=48,mean=float(v[ix].mean()),ci95_unadjusted=np.quantile(v[bb].mean(1),[.025,.975]).tolist())
  result[metric]['length_strata_secondary']=secondary
 return result

def main():
 ex=read(R/'execution_lock.json');sl=read(R/'score_lock.json')
 assert sha(R/'execution_lock.json')==sl['execution_lock_sha256']
 for f,h in ex['files'].items():assert sha(f)==h,f
 for f,h in sl['files'].items():assert sha(f)==h,f
 done=read(R/'prediction_completion.json');assert done['complete'] and done['execution_lock_sha256']==sha(R/'execution_lock.json')
 targets=read(R/'panel/reference_manifest.json')['targets'];rows=done['records'];validate_records(rows,targets)
 out=R/'analysis';out.mkdir(exist_ok=True)
 if (out/'complete.json').exists():
  old=read(out/'complete.json');assert sha(out/'analysis.json')==old['analysis_sha256'];return
 fn=load_reference_lddt(sl['lddt_path']);refs={};by={t['target_id']:t for t in targets}
 for t in targets:
  assert sha(t['raw_mmcif_path'])==t['raw_mmcif_sha256']
  _,atoms=read_atom_site_positions(Path(t['raw_mmcif_path']),label_asym_id=t['source_label_asym_id'])
  ref={i:np.asarray(v,dtype=np.float64) for (i,a),v in atoms.items() if a=='CA'};assert sorted(ref)==t['reference_ca_indices'];refs[t['target_id']]=ref
  f=out/'coordinates'/t['target_id'];f.mkdir(parents=True,exist_ok=True);(f/'reference.pdb').write_text(ca_pdb(ref))
 def score(x):
  tid=x['target_id'];t=by[tid];base=dict(system=x['system'],target_id=tid,status=x['status'])
  if x['status']=='failed':return dict(base,**dict.fromkeys(METRICS,0.),error=x.get('error'))
  assert sha(x['prediction_path'])==x['prediction_sha256']
  _,a=read_atom_site_positions(Path(x['prediction_path']));pred={i:np.asarray(v,dtype=np.float64) for (i,atom),v in a.items() if atom=='CA'}
  assert sorted(pred)==list(range(1,t['sequence_length']+1)) and np.isfinite(list(pred.values())).all()
  ref=refs[tid];ids=sorted(ref);px=np.stack([pred[i] for i in ids]);rx=np.stack([ref[i] for i in ids]);mask=np.ones((1,len(ids),1))
  pair=fixed_mask_metrics(ref,pred)['ca_lddt'];assert abs(pair-float(fn(px[None],rx[None],mask)[0]))<1e-6
  residue=float(fn(px[None],rx[None],mask,per_residue=True).mean());folder=out/'coordinates'/tid;pp=folder/(x['system']+'.pdb');pp.write_text(ca_pdb({i:pred[i] for i in ids}))
  tm=subprocess.run([sl['tm_path'],str(pp),str(folder/'reference.pdb'),'-l',str(t['sequence_length'])],capture_output=True,text=True,check=True)
  (folder/(x['system']+'.tm.txt')).write_text(tm.stdout+tm.stderr)
  return dict(base,ca_lddt=pair,residue_ca_lddt=residue,tm_score_fixed_full_length=parse_tm_score(tm.stdout),prediction_sha256=x['prediction_sha256'])
 scored=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
  for x in pool.map(score,rows):
   scored.append(x)
   if len(scored)%192==0:print('SCORED',len(scored),flush=True)
 write(out/'metric_records.json',scored)
 analysis=dict(primary='ca_lddt interaction Psi',targets=[t['target_id'] for t in targets],n_targets=192,predictions=4800,failures=sum(x['status']!='ok' for x in scored),new_training=0,conditional_on_fixed_models=True,bootstrap=dict(draws=20000,seed=20260925,unit='target within four predefined length strata'),metrics=summarize(scored,targets),execution_lock_sha256=sha(R/'execution_lock.json'),scoring_started_after_prediction_completion=True)
 write(out/'analysis.json',analysis)
 with (out/'per_target.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['target_id','length_bin','psi','factor_rotation','gplus_rotation'])
  for i,t in enumerate(targets):w.writerow([t['target_id'],t['length_bin'],*[analysis['metrics']['ca_lddt'][k]['per_target'][i] for k in ['interaction','factor_rotation','gplus_rotation']]])
 primary=analysis['metrics']['ca_lddt'];psi=primary['interaction'];text=f"# Protenix Fresh192 结果\n\n192目标/4800正式预测；失败{analysis['failures']}，保留分母。\n\n主交互Psi={psi['mean']:+.8f}，95%CI {psi['ci95']}，正向{psi['positive_targets']}/192。\n\n四格与Query均值：{primary['four_cell_means']}。\n\n唯一主要终点正向支持：{psi['ci95'][0]>0}。条件于既有24拟合模型；不保证家族/基础模型训练隔离。历史配方选择已披露。其他比较、长度层和指标为次要，不替换Psi；不追加样本。\n"
 (out/'results.md').write_text(text)
 write(out/'complete.json',dict(complete=True,analysis_sha256=sha(out/'analysis.json'),records_sha256=sha(out/'metric_records.json'),execution_lock_sha256=sha(R/'execution_lock.json'),completed_time=time.time()))
if __name__=='__main__':main()
