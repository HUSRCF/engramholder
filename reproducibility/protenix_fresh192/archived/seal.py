"""Seal deployment only after all three independent gates, before new folding."""
import time
from pathlib import Path
import torch
from .common import *
R=Path('/media/PM982/engramfold/runs/protenix_fresh192_20260925')

def main():
 assert not (R/'execution_lock.json').exists(),'do not overwrite a scientific lock'
 model=read(R/'model_lock.json');assert set(model['systems'])==set(models()) and len(model['adapters'])==24
 engineering=read(R/'engineering/complete.json');assert engineering['passed'] and engineering['predictions']==50 and engineering['model_lock_sha256']==sha(R/'model_lock.json')
 assert read(R/'selection_audit.json')['passed']
 selection=read(R/'panel/complete.json');assert selection['passed'] and selection['targets']==192
 assert selection['inference_sha256']==sha(R/'panel/inference_manifest.json') and selection['reference_sha256']==sha(R/'panel/reference_manifest.json')
 validate_manifest(read(R/'panel/inference_manifest.json'))
 feature=R/'features/esmc';idx=read(feature/'index.json');ready=read(R/'features_ready.json');assert ready['passed'] and ready['index_sha256']==sha(feature/'index.json')
 assert ready['manifest_sha256']==sha(R/'panel/inference_manifest.json')==idx['manifest_sha256']
 oldidx=read(Path(model['feature_root'])/'index.json');assert idx['weights_hashes']==oldidx['weights_hashes'] and idx['feature_spec']==oldidx['feature_spec']
 assert idx['layer']==36 and idx['input_dim']==1152 and set(idx['records'])=={t['target_id'] for t in read(R/'panel/inference_manifest.json')['targets']}
 for t in read(R/'panel/inference_manifest.json')['targets']:
  rec=idx['records'][t['target_id']];p=feature/rec['file'];assert sha(p)==rec['sha256']
  d=torch.load(p,map_location='cpu',weights_only=True)
  assert d['sequence_sha256']==t['sequence_sha256']==rec['sequence_sha256'] and d['features'].shape==(t['sequence_length'],1152) and d['features'].dtype==torch.bfloat16 and torch.isfinite(d['features']).all()
 for name,item in model['adapters'].items():assert sha(item['path'])==item['sha256']
 lock={**model,'manifest_sha256':sha(R/'panel/inference_manifest.json'),'model_lock_sha256':sha(R/'model_lock.json')};write(R/'prediction_lock.json',lock)
 tasks=[]
 for name in models():
  for t in read(R/'panel/inference_manifest.json')['targets']:
   tasks.append(dict(system=name,target_id=t['target_id'],sequence_sha256=t['sequence_sha256'],checkpoint_sha256=(model['adapters'][name]['sha256'] if name!='query' else next(iter(model['base'].values()))['sha256']),seed=101,cycles=4,steps=5,samples=1))
 assert len(tasks)==4800
 (R/'prediction_manifest.jsonl').write_text(''.join(__import__('json').dumps(x,sort_keys=True)+'\n' for x in tasks))
 paths=[R/x for x in ['prediction_manifest.jsonl','model_lock.json','protocol.md','scoring_exposure_audit.json','engineering/complete.json','engineering_manifest.json','selection_audit.json','panel/selection_lock.json','panel/inference_manifest.json','panel/reference_manifest.json','panel/complete.json','prediction_lock.json','features_ready.json','features/esmc/index.json']]
 paths+=sorted(p for p in (R/'source').rglob('*') if p.is_file() and '__pycache__' not in str(p) and p.suffix in ('.py','.md'))
 # The numerical prediction implementation must remain the one tested on old chains.
 code=R/'source/src/engramfold/experiments/protenix_fresh_fourcell/predict.py'
 for name in models():assert read(R/'engineering'/name/'report.json')['source_sha256']==sha(code)
 paths += [Path(x['path']) for x in model['base'].values()]
 files={str(p):sha(p) for p in paths}
 env=read(R/'environment_snapshot.json');assert env['torch_version']==torch.__version__ and env['hip_version']==torch.version.hip
 for path,h in env['package_python_hashes'].items():assert sha(path)==h
 files.update(env['package_python_hashes']);files[str(R/'environment_snapshot.json')]=sha(R/'environment_snapshot.json')
 write(R/'execution_lock.json',dict(schema='engramfold.protenix_fresh192.execution.v1',files=files,created_time=time.time(),models=25,targets=192,predictions=4800,new_training=0,inference_manifest_sha256=sha(R/'panel/inference_manifest.json'),prediction_lock_sha256=sha(R/'prediction_lock.json'),feature_index_sha256=sha(feature/'index.json'),primary='ca_lddt interaction Psi',bootstrap=dict(draws=20000,seed=20260925,strata=['128-191','192-255','256-319','320-384']),retry_delay=120,max_retries_per_target=2,backend='DiamondHill MI250 ROCm; features historical H100 path',selected_from_observed_recipe=True))
 oldscore=read('/media/PM982/engramfold/folding_e2e_20260921/formal_panel/scoring_lock.json')
 for k in ['tm','lddt']:assert sha(oldscore[k+'_path'])==oldscore[k+'_sha256']
 write(R/'score_lock.json',dict(execution_lock_sha256=sha(R/'execution_lock.json'),tm_path=oldscore['tm_path'],lddt_path=oldscore['lddt_path'],files={oldscore[k+'_path']:oldscore[k+'_sha256'] for k in ['tm','lddt']},primary='ca_lddt interaction Psi',release='all 4800 predictions terminal and source/output identities valid'))
 write(R/'preflight_complete.json',dict(passed=True,execution_lock_sha256=sha(R/'execution_lock.json'),models=25,targets=192,engineering=50,features=192,all_scientific_parameters_fixed_before_new_folding=True,time=time.time()))
 print('SEALED',sha(R/'execution_lock.json'),flush=True)
if __name__=='__main__':main()
