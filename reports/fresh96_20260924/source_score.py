"""Uniform scoring only after all 2400 fixed model-target attempts terminate."""
import argparse,json,pathlib,subprocess
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from engramfold.experiments.fresh_interaction import sha,atomic_json,validate_matrix,require_complete,contrast_arrays
from engramfold.evaluation.structure import read_atom_site_positions
from engramfold.evaluation.independent import fixed_mask_metrics
from engramfold.evaluation.post_validation import ca_pdb,parse_tm_score,load_reference_lddt

def score_prediction(x, t, ref, path, out, tm_binary, fn):
 m=x['model'];tid=x['target_id'];base=dict(system=m,target_id=tid,panel='fresh96',status=x['status'])
 if x['status']=='failed':return dict(**base,ca_lddt=0.,residue_ca_lddt=0.,tm_score_fixed_full_length=0.,error=x.get('error'))
 assert sha(path)==x['prediction_sha256'];arr=np.load(path);assert str(arr['sequence'])==t['sequence']
 xyz=arr['coordinates'].astype(np.float64);mask=arr['mask'];pred={i+1:v for i,v in enumerate(xyz[:,1]) if mask[i,1]};ii=sorted(ref);assert all(i in pred for i in ii)
 px=np.stack([pred[i]for i in ii]);rx=np.stack([ref[i]for i in ii]);mm=np.ones((1,len(ii),1));pair=fixed_mask_metrics(ref,pred)['ca_lddt'];assert abs(pair-float(fn(px[None],rx[None],mm)[0]))<1e-6
 residue=float(fn(px[None],rx[None],mm,per_residue=True).mean());folder=out/'coordinates'/tid;pp=folder/(m+'.pdb');pp.write_text(ca_pdb({i:pred[i]for i in ii}))
 tm=subprocess.run([str(tm_binary),str(pp),str(folder/'reference.pdb'),'-l',str(len(t['sequence']))],text=True,capture_output=True,check=True);(folder/(m+'.tm.txt')).write_text(tm.stdout+tm.stderr)
 return dict(**base,ca_lddt=pair,residue_ca_lddt=residue,tm_score_fixed_full_length=parse_tm_score(tm.stdout),prediction_sha256=x['prediction_sha256'])

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=pathlib.Path);a=p.parse_args();r=a.root
 sl=json.loads((r/'scoring_lock.json').read_text());el=json.loads((r/'execution_lock.json').read_text());lk=sha(r/'execution_lock.json');assert sl['execution_lock_sha256']==lk
 for name,h in sl['files'].items():assert sha(name)==h,name
 models=json.loads((r/'model_lock.json').read_text())['models'];validate_matrix(models)
 refs=json.loads((r/'panel/reference_manifest.json').read_text())['targets'];ids=[t['target_id'] for t in refs];assert len(ids)==96
 records=[]
 for m in models:
  done=json.loads((r/'formal'/m['name']/'complete.json').read_text());assert done['complete'] and done['execution_lock_sha256']==lk and done['expected']==96
  records+=done['records']
 require_complete(records,models,ids)
 assert all(any(x['status']=='ok' for x in records if x['model']==m['name']) for m in models), 'wholly failed system is an incomplete engineering run; do not manufacture a scientific contrast'
 assert all(x['execution_lock_sha256']==lk for x in records)
 out=r/'analysis';out.mkdir(exist_ok=True);fn=load_reference_lddt(r/'scoring/alphafold_lddt.py');byid={t['target_id']:t for t in refs};refcoords={}
 for t in refs:
  tid=t['target_id'];path=r/'reference'/(tid+'.cif.gz');assert sha(path)==t['raw_mmcif_sha256']
  _,atoms=read_atom_site_positions(path,label_asym_id=t['source_label_asym_id']);ref={i:v for (i,n),v in atoms.items() if n=='CA'};assert sorted(ref)==t['reference_ca_indices'];refcoords[tid]=ref
  folder=out/'coordinates'/tid;folder.mkdir(parents=True,exist_ok=True);(folder/'reference.pdb').write_text(ca_pdb(ref))
 def score(x):
  return score_prediction(x,byid[x['target_id']],refcoords[x['target_id']],r/'formal'/x['model']/(x['target_id']+'.npz'),out,r/'scoring/TMscore',fn)
 with ThreadPoolExecutor(max_workers=8) as pool:rows=list(pool.map(score,records))
 atomic_json(out/'metric_records.json',rows);rng=np.random.default_rng(20260921);draw=rng.integers(96,size=(20000,96));result=dict(execution_lock_sha256=lk,targets=ids,metrics={},n_predictions=2400,failed=sum(x['status']!='ok' for x in rows),conditional_on_fitted_models=True,new_training=0,primary='fresh96 ca_lddt interaction')
 def summary(x):return dict(mean=float(x.mean()),ci95=np.quantile(x[draw].mean(axis=1),[.025,.975]).tolist(),per_target=x.tolist(),positive_targets=int((x>0).sum()))
 for metric in ['ca_lddt','residue_ca_lddt','tm_score_fixed_full_length']:
  scores={(x['system'],x['target_id']):x[metric]for x in rows};arr=contrast_arrays(scores,models,ids)
  d={k:summary(v) if k in ['factor_rotation','gplus_rotation','interaction','native_minus_gplus'] else v.tolist()for k,v in arr.items()};d['system_means']={m['name']:float(np.mean([scores[m['name'],i] for i in ids])) for m in models}
  d['group_means']={}
  for k in ['factor','generic_plus','query']:
   for rotated in ([False] if k=='query' else [False,True]):
    names=[m['name']for m in models if m['kind']==k and ((m['rotation']is not None)==rotated)];d['group_means'][k+('_rotated' if rotated else '_native')]=float(np.mean([scores[n,i] for n in names for i in ids]))
  q=np.array([scores['query',i] for i in ids])
  for kind in ['factor','generic_plus']:
   names=[m['name'] for m in models if m['kind']==kind and m['rotation'] is None]
   d[kind+'_minus_query']=summary(np.mean([[scores[n,i]for i in ids]for n in names],axis=0)-q)
  d['length_strata']={}
  for b in sorted({t['length_bin'] for t in refs}):
   ix=np.array([k for k,t in enumerate(refs)if t['length_bin']==b]);v=arr['interaction'][ix];bdraw=np.random.default_rng(20260921).integers(len(ix),size=(20000,len(ix)))
   d['length_strata'][b]={'n':len(ix),'mean':float(v.mean()),'ci95_unadjusted':np.quantile(v[bdraw].mean(axis=1),[.025,.975]).tolist()}
  result['metrics'][metric]=d
 atomic_json(out/'analysis.json',result);atomic_json(out/'complete.json',dict(complete=True,execution_lock_sha256=lk,records_sha256=sha(out/'metric_records.json'),analysis_sha256=sha(out/'analysis.json')))
 print(json.dumps({'complete':True,'primary':result['metrics']['ca_lddt']['interaction']['mean']}))
if __name__=='__main__':main()
