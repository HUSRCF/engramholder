"""Score all fixed predictions; direct target-paired head x rotation interaction."""
import argparse,json,subprocess,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from engramfold.evaluation.structure import read_atom_site_positions
from engramfold.evaluation.independent import fixed_mask_metrics
from engramfold.evaluation.post_validation import ca_pdb,parse_tm_score,load_reference_lddt
from engramfold.experiments.cross_backbone_protocol import atomic_json,sha
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
lock=r/'formal_execution_lock.json';c=json.loads(lock.read_text());lk=sha(lock);old=Path(c['historical_root']);sl=json.loads((r/'scoring_execution_lock.json').read_text())
assert sha(__file__)==sl['scorer_sha256'];assert sha(old/'formal_execution_lock.json')==c['historical_execution_lock_sha256'];assert sha(old/'analysis/metric_records.json')==sl['historical_metrics_sha256']
for name,key in [('TMscore','tm_sha256'),('alphafold_lddt.py','lddt_sha256')]:assert sha(r/'data'/name)==sl[key]
targets=json.loads((r/'data/evaluation144.json').read_text())['targets'];assert len(targets)==144
out=r/'analysis';out.mkdir(exist_ok=True);refs={};fn=load_reference_lddt(r/'data/alphafold_lddt.py')
for t in targets:
 tid=t['target_id'];path=r/'data'/f'{tid}.cif.gz';assert sha(path)==c['data_hashes'][f'{tid}.cif.gz'];_,atoms=read_atom_site_positions(path,label_asym_id=t['source_label_asym_id']);ref={i:v for(i,n),v in atoms.items()if n=='CA'};assert sorted(ref)==t['reference_ca_indices'];refs[tid]=ref
 folder=out/'coordinates'/tid;folder.mkdir(parents=True,exist_ok=True);(folder/'reference.pdb').write_text(ca_pdb(ref))
jobs=[]
for run in c['new_runs']:
 folder=r/'openfold/formal'/run['name'];done=json.loads((folder/'complete.json').read_text());assert done['complete']and done['execution_lock_sha256']==lk
 rep=json.loads((folder/'evaluation.json').read_text());assert rep['complete']and rep['execution_lock_sha256']==lk;assert [x['target_id']for x in rep['records']]==[x['target_id']for x in targets]
 jobs.extend((run['name'],t,x,folder/'predictions'/f"{t['target_id']}.npz")for t,x in zip(targets,rep['records']))
def score(job):
 name,t,x,path=job;tid=t['target_id'];base=dict(system=name,target_id=tid,panel=t['panel'],status=x['status'])
 if x['status']!='ok':return dict(**base,ca_lddt=0.,residue_ca_lddt=0.,tm_score_fixed_full_length=0.,error=x.get('error'))
 assert sha(path)==x['prediction_sha256'];arr=np.load(path);assert str(arr['sequence'])==t['sequence'];xyz=arr['coordinates'].astype(np.float64);mask=arr['mask'];assert xyz.shape==(len(t['sequence']),37,3)and np.isfinite(xyz).all()
 pred={i+1:v for i,v in enumerate(xyz[:,1])if mask[i,1]};ref=refs[tid];ids=sorted(ref);assert all(i in pred for i in ids)
 px=np.stack([pred[i]for i in ids]);rx=np.stack([ref[i]for i in ids]);m=np.ones((1,len(ids),1));pair=fixed_mask_metrics(ref,pred)['ca_lddt'];assert abs(pair-float(fn(px[None],rx[None],m)[0]))<1e-6
 resid=float(fn(px[None],rx[None],m,per_residue=True).mean());folder=out/'coordinates'/tid;pp=folder/f'{name}.pdb';pp.write_text(ca_pdb({i:pred[i]for i in ids}))
 tm=subprocess.run([str(r/'data/TMscore'),str(pp),str(folder/'reference.pdb'),'-l',str(len(t['sequence']))],capture_output=True,text=True,check=True);(folder/f'{name}.tm.txt').write_text(tm.stdout+tm.stderr)
 return dict(**base,ca_lddt=pair,residue_ca_lddt=resid,tm_score_fixed_full_length=parse_tm_score(tm.stdout),prediction_sha256=sha(path))
with ThreadPoolExecutor(max_workers=8)as pool:new=list(pool.map(score,jobs))

assert len(new)==4752
legacy=json.loads((old/'analysis/metric_records.json').read_text())
for row in legacy:
 name=row['system']
 if name.startswith('native_s'):row['system']='E_last_factor_'+name
 elif name.startswith('r2026'):row['system']='E_last_factor_'+name
 elif name.startswith('gplus_s'):row['system']='E_last_generic_plus_native_'+name.removeprefix('gplus_')
for row in new:
 if row['system'].startswith('E_last_gplus_r'):row['system']=row['system'].replace('E_last_gplus_r','E_last_generic_plus_r',1)
rows=legacy+new
atomic_json(out/'new_metric_records.json',new);atomic_json(out/'combined_metric_records.json',rows)
result=dict(execution_lock_sha256=lk,n_new_predictions=len(new),new_failures=sum(x['status']!='ok' for x in new),primary='confirm96 ca_lddt C-last factor_rotation',scope='observed follow-up; fitted-model conditional intervals',panels={})
for panel in ['confirm96','length48']:
 ids=[t['target_id'] for t in targets if t['panel']==panel]
 draw=np.random.default_rng(20260926).integers(len(ids),size=(20000,len(ids)));summary={}
 for metric in ['ca_lddt','residue_ca_lddt','tm_score_fixed_full_length']:
  values={}
  for row in rows:
   if row['panel']==panel:
    key=row['system'],row['target_id'];assert key not in values;values[key]=row[metric]
  def vector(name):return np.array([values[name,i] for i in ids])
  def contrast(delta):
   di=delta.mean((0,1));boot=di[draw].mean(1);mean=float(di.mean())
   return dict(mean=mean,ci95=np.quantile(boot,[.025,.975]).tolist(),target_ids=ids,per_target=di.tolist(),positive_targets=int((di>0).sum()),per_seed=delta.mean((0,2)).tolist(),per_rotation=delta.mean((1,2)).tolist(),centered_bootstrap_p=float((1+(np.abs(boot-mean)>=abs(mean)).sum())/20001))
  cells={};deltas={}
  for feature in ['E_last','C_last']:
   native=np.stack([vector(f'{feature}_factor_native_s{s}') for s in c['formal_seeds']])
   generic=np.stack([vector(f'{feature}_generic_plus_native_s{s}') for s in c['formal_seeds']])
   fr=np.stack([np.stack([vector(f'{feature}_factor_r{rr}_s{s}') for s in c['formal_seeds']]) for rr in c['rotation_seeds']])
   gr=np.stack([np.stack([vector(f'{feature}_generic_plus_r{rr}_s{s}') for s in c['formal_seeds']]) for rr in c['rotation_seeds']])
   df=native[None]-fr;dg=generic[None]-gr;psi=df-dg
   deltas[feature]=dict(df=df,dg=dg,psi=psi,native=native,generic=generic)
   cells[feature]=dict(factor_rotation=contrast(df),gplus_rotation=contrast(dg),interaction=contrast(psi),native_minus_gplus=contrast((native-generic)[None]),means=dict(native=float(native.mean()),rotated_factor=float(fr.mean()),gplus=float(generic.mean()),rotated_gplus=float(gr.mean()),query=float(vector('query').mean())))
  e,d=deltas['E_last'],deltas['C_last']
  cross=dict(direction_change=contrast(d['df']-e['df']),psi_change=contrast(d['psi']-e['psi']),native_gain=contrast((d['native']-e['native'])[None]),gplus_gain=contrast((d['generic']-e['generic'])[None]),native_gplus_gap_change=contrast(((d['native']-d['generic'])-(e['native']-e['generic']))[None]))
  summary[metric]=dict(cells=cells,plm_interactions=cross)
 result['panels'][panel]=summary
family=[('Psi_C',result['panels']['confirm96']['ca_lddt']['cells']['C_last']['interaction']),('K',result['panels']['confirm96']['ca_lddt']['plm_interactions']['direction_change']),('J',result['panels']['confirm96']['ca_lddt']['plm_interactions']['psi_change'])]
ranked=sorted(family,key=lambda x:x[1]['centered_bootstrap_p']);adjusted={};running=0.
for rank,(name,obj) in enumerate(ranked):
 running=max(running,min(1.,(3-rank)*obj['centered_bootstrap_p']));adjusted[name]=running
result['key_secondary_holm_p']=adjusted
atomic_json(out/'analysis.json',result)
lines=['# OpenFold ESMC Stage A','','| Panel | Feature | F | FR | G+ | G+R | F−FR | Psi |','|---|---|---:|---:|---:|---:|---:|---:|']
for panel,d in result['panels'].items():
 for feat,row in d['ca_lddt']['cells'].items():
  m=row['means'];lines.append(f"|{panel}|{feat}|{m['native']:.5f}|{m['rotated_factor']:.5f}|{m['gplus']:.5f}|{m['rotated_gplus']:.5f}|{row['factor_rotation']['mean']:+.5f}|{row['interaction']['mean']:+.5f}|")
(out/'results.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(complete=True,predictions=4752,failures=result['new_failures'])),flush=True)
