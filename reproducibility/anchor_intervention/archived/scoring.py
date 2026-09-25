# Scoring function copied verbatim from accepted E1 scoring amendment.
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from parent_core import sha,write
from engramfold.evaluation.independent import fixed_mask_metrics
from engramfold.evaluation.structure import read_atom_site_positions
from engramfold.evaluation.post_validation import ca_pdb,parse_tm_score,load_reference_lddt
r=None
def score_jobs(jobs,out,secondary):
 out.mkdir(parents=True,exist_ok=True);refs={};fn=load_reference_lddt(r/'data/alphafold_lddt.py')
 for _,t,_,_ in jobs:
  tid=t['target_id']
  if tid not in refs:
   _,atoms=read_atom_site_positions(r/'data'/f'{tid}.cif.gz',label_asym_id=t['source_label_asym_id']);ref={i:v for(i,n),v in atoms.items()if n=='CA'}
   if'reference_ca_indices'in t:assert sorted(ref)==t['reference_ca_indices']
   refs[tid]=ref
 def calc(job):
  name,t,rec,path=job;tid=t['target_id'];base=dict(system=name,target_id=tid,status=rec['status'])
  if rec['status']!='ok':return dict(**base,ca_lddt=0.,residue_ca_lddt=0.,tm_score_fixed_full_length=0.,error=rec.get('error'))
  assert sha(path)==rec['prediction_sha256'];d=np.load(path);assert str(d['sequence'])==t['sequence'];xyz=d['coordinates'];mask=d['mask'];assert np.isfinite(xyz).all()and xyz.shape==(len(t['sequence']),37,3)
  pred={i+1:v for i,v in enumerate(xyz[:,1])if mask[i,1]};ref=refs[tid];ids=sorted(ref);assert all(i in pred for i in ids)
  pair=fixed_mask_metrics(ref,pred)['ca_lddt'];px=np.stack([pred[i]for i in ids]).astype('float64');rx=np.stack([ref[i]for i in ids]).astype('float64');m=np.ones((1,len(ids),1))
  check_px=np.stack([pred[i]for i in ids]);check_rx=np.stack([ref[i]for i in ids])
  check_value=float(fn(check_px[None],check_rx[None],m)[0])
  assert abs(pair-check_value)<1e-6,(name,tid,pair,check_value,str(check_px.dtype),str(check_rx.dtype))
  result=dict(**base,ca_lddt=pair,prediction_sha256=sha(path),reference_check_same_input_dtypes=check_value,reference_check_fp64=float(fn(px[None],rx[None],m)[0]))
  if secondary:
   folder=out/'coordinates'/name/tid;folder.mkdir(parents=True,exist_ok=True);rp=folder/'ref.pdb';pp=folder/'pred.pdb';rp.write_text(ca_pdb(ref));pp.write_text(ca_pdb({i:pred[i]for i in ids}))
   tm=subprocess.run([str(r/'data/TMscore'),str(pp),str(rp),'-l',str(len(t['sequence']))],capture_output=True,text=True,check=True);(folder/'tm.txt').write_text(tm.stdout+tm.stderr)
   result.update(residue_ca_lddt=float(fn(px[None],rx[None],m,per_residue=True).mean()),tm_score_fixed_full_length=parse_tm_score(tm.stdout))
  return result
 with ThreadPoolExecutor(max_workers=8)as pool:rows=list(pool.map(calc,jobs))
 write(out/'metric_records.json',rows);return rows
