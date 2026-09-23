"""Fixed-checkpoint inference. This file never reads reference coordinates/masks."""
import argparse,json,pathlib,time,hashlib,fcntl,traceback,os
import numpy as np
import torch
from engramfold.experiments.fresh_interaction import sha,atomic_json,validate_sequence_manifest,validate_matrix
from engramfold.experiments.openfold_adapter_runtime import runtime_config,enable_nonreentrant_checkpointing,sequence_features
from engramfold.models.live_opm import LiveOPMAdapter,LiveOPMHook
from smoke_openfold_adapter import state_hash

def verify(r,stage):
 c=json.loads((r/'preparation_lock.json').read_text())
 for path,h in c['files'].items():assert sha(path)==h,path
 ml=json.loads((r/'model_lock.json').read_text());validate_matrix(ml['models'])
 for path,h in ml['source_hashes'].items():assert sha(path)==h,path
 assert sha(r/'weights/params_model_3_ptm.npz')==ml['weight_sha256']
 if stage=='formal':
  e=json.loads((r/'execution_lock.json').read_text());assert e['preparation_lock_sha256']==sha(r/'preparation_lock.json')
  for path,h in e['files'].items():assert sha(path)==h,path
  assert json.loads((r/'engineering/complete.json').read_text())['passed']
 return ml

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--stage',choices=['engineering','formal'],required=True);p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1);a=p.parse_args();r=a.root
 torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 assert 'H100' in torch.cuda.get_device_name()
 ml=verify(r,a.stage);manifest=r/('smoke_sequences.json' if a.stage=='engineering' else 'panel/inference_manifest.json');ts=validate_sequence_manifest(json.loads(manifest.read_text()),2 if a.stage=='engineering' else 96)
 cfg=runtime_config(activation_checkpointing=True);enable_nonreentrant_checkpointing()
 from openfold.model.model import AlphaFold
 from openfold.utils.import_weights import assign,generate_translation_dict,process_translation_dict
 model=AlphaFold(cfg).eval().requires_grad_(False);weights=np.load(r/'weights/params_model_3_ptm.npz');trans=process_translation_dict(generate_translation_dict(model,'model_3_ptm',is_multimer=False));assert set(trans)==set(weights.files);assign(trans,weights);del trans,weights
 model.cuda();frozen=state_hash(model);assert frozen==ml['frozen_backbone_sha256'];models=ml['models'][a.worker::a.workers];allrows=[]
 prepared={}
 for t in ts:
  tid=t['target_id'];torch.manual_seed(20260921);f=sequence_features(t['sequence'],cfg)
  fp=r/('historical_data' if a.stage=='engineering' else 'features')/(tid+'.pt');cache=torch.load(fp,map_location='cpu',weights_only=True)
  assert cache['sequence_sha256']==t['sequence_sha256'];e=cache['features'].float();assert e.shape==(len(t['sequence']),480) and torch.isfinite(e).all()
  prepared[tid]=(f,e)
 for m in models:
  out=r/a.stage/m['name'];out.mkdir(parents=True,exist_ok=True);handle=(out/'run.lock').open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
  lockhash=sha(r/('preparation_lock.json' if a.stage=='engineering' else 'execution_lock.json'))
  writer=None
  if m['kind']!='query':
   assert sha(m['checkpoint'])==m['checkpoint_sha256'];ck=torch.load(m['checkpoint'],map_location='cpu',weights_only=True);assert ck['step']==1536 and ck['execution_lock_sha256']==m['source_execution_lock_sha256']
   assert hashlib.sha256(ck['writer']['rotation'].contiguous().numpy().tobytes()).hexdigest()==m['rotation_sha256']
   torch.manual_seed(m['seed']);writer=LiveOPMAdapter(backbone='openfold',kind=m['kind'],rotation_seed=m['rotation']).cuda().eval().requires_grad_(False);writer.load_state_dict(ck['writer']);del ck
  rows=[]
  for t in ts:
   tid=t['target_id'];predpath=out/(tid+'.npz');recordpath=out/(tid+'.json');row=None
   if recordpath.exists():
    row=json.loads(recordpath.read_text());assert row['execution_lock_sha256']==lockhash and row['sequence_sha256']==t['sequence_sha256'] and row['checkpoint_sha256']==m.get('checkpoint_sha256')
    if row['status']=='ok':assert sha(predpath)==row['prediction_sha256']
   if row is None:
    started=time.monotonic();errors=[]
    for attempt in range(3):
     hook=None;trace=[]
     try:
      f,e=prepared[tid];f={k:v.cuda() for k,v in f.items()};e=e.cuda();torch.manual_seed(20260921)
      hook=LiveOPMHook(model.evoformer.blocks[0].outer_product_mean,writer,e) if writer is not None else None
      counter=model.evoformer.register_forward_hook(lambda *args:trace.append(1))
      try:
       with torch.no_grad():result=model(f)
      finally:counter.remove()
      assert len(trace)==4 and (hook is None or len(hook.calls)==4)
      xyz=result['final_atom_positions'].cpu().numpy();mask=result['final_atom_mask'].cpu().numpy();assert xyz.shape==(t['sequence_length'],37,3) and np.isfinite(xyz).all() and mask[:,1].sum()==t['sequence_length']
      with predpath.with_suffix('.npz.tmp').open('wb') as file:np.savez(file,coordinates=xyz,mask=mask,sequence=t['sequence'])
      predpath.with_suffix('.npz.tmp').replace(predpath)
      row=dict(status='ok',prediction_sha256=sha(predpath),attempts=attempt+1,trunk_calls=len(trace))
      if a.stage=='engineering':
       old=np.load(pathlib.Path(m['historical_predictions'])/(tid+'.npz'));assert str(old['sequence'])==t['sequence'] and np.array_equal(old['mask'],mask)
       delta=xyz.astype('float64')-old['coordinates'];rel=float(np.linalg.norm(delta)/max(np.linalg.norm(old['coordinates']),1e-30));rms=float(np.sqrt(np.mean(delta**2)))
       assert rel<=1e-3 and rms<=.02,(m['name'],tid,rel,rms);row.update(replay_relative_l2=rel,replay_coordinate_rms=rms)
      del result,f,e;break
     except Exception:
      err=traceback.format_exc();errors.append(err);atomic_json(out/f'{tid}.attempt{attempt}.error.json',dict(error=err,attempt=attempt))
      if a.stage=='engineering':raise
      # Only resource/runtime failures are retried; contract/assertion failures stop the job.
      if not any(s in err for s in ('CUDA out of memory','CUDA error','HIP error','hipError')):raise
      if attempt==2:row=dict(status='failed',attempts=3,error=err)
      else:torch.cuda.empty_cache();time.sleep(120)
     finally:
      if hook is not None:hook.remove()
    row.update(model=m['name'],target_id=tid,sequence_sha256=t['sequence_sha256'],checkpoint_sha256=m.get('checkpoint_sha256'),execution_lock_sha256=lockhash,seconds=time.monotonic()-started)
    atomic_json(recordpath,row)
   rows.append(row);print(json.dumps(row),flush=True);atomic_json(out/'progress.json',dict(expected=len(ts),records=rows))
  assert state_hash(model)==frozen and all(p.grad is None for p in model.parameters())
  atomic_json(out/'complete.json',dict(complete=True,execution_lock_sha256=lockhash,records=rows,expected=len(ts),frozen_backbone_sha256=frozen))
  allrows+=rows;del writer;handle.close()
 if a.stage=='engineering':
  assert len(allrows)==50 and all(x['status']=='ok' for x in allrows)
  atomic_json(r/'engineering/complete.json',dict(passed=True,count=50,preparation_lock_sha256=sha(r/'preparation_lock.json'),max_relative_l2=max(x['replay_relative_l2'] for x in allrows),max_coordinate_rms=max(x['replay_coordinate_rms'] for x in allrows)))
if __name__=='__main__':main()
