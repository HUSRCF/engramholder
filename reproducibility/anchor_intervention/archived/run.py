import argparse,fcntl,time,traceback,json
from pathlib import Path
import numpy as np
from e3_common import *
from query_anchor import QueryAnchorHook
from engramfold.experiments.cross_backbone_protocol import schedule
from engramfold.experiments.openfold_adapter_runtime import native_loss
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--index',type=int,required=True);a=p.parse_args();r=a.root
c,old,parent,lk=contract(r,True)
gate=read(r/'references/complete.json');assert gate['complete']and gate['execution_lock_sha256']==lk;reference_receipt_sha=sha(r/'references/complete.json')
run=c['runs'][a.index];out=r/'formal'/run['name'];out.mkdir(parents=True,exist_ok=True)
lease=(out/'lease').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
if(out/'complete.json').exists():assert read(out/'complete.json')['execution_lock_sha256']==lk;raise SystemExit('ALREADY_COMPLETE')
write(out/'run.json',dict(run=run,execution_lock_sha256=lk,reference_receipt_sha256=reference_receipt_sha))
model,config=pc.load_model(parent,old);prep=pc.Features(parent,config);w=pc.writer(parent,run['rotation_id'],run['seed']);assert w.channels is None
opt=pc.optimizer(w,False);train=target_list(parent,'train');order=schedule(run['seed'],96,1536);write(out/'schedule.json',[train[i]['target_id']for i in order]);start=0
if(out/'latest.pt').exists():
 ck=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
 assert ck['execution_lock_sha256']==lk and ck['run']==run and ck['reference_receipt_sha256']==reference_receipt_sha
 w.load_state_dict(ck['writer']);opt.load_state_dict(ck['optimizer']);start=ck['step']
 assert ck['data_position']==start and ck['lr']==[g['lr']for g in opt.param_groups]
 if(out/'training.jsonl').exists():
  raw=(out/'training.jsonl').read_text();(out/f'resume_{time.time_ns()}.jsonl').write_text(raw)
  (out/'training.jsonl').write_text(''.join(line+'\n'for line in raw.splitlines()if json.loads(line)['step']<=start))
 torch.set_rng_state(ck['torch_rng'].cpu());torch.cuda.set_rng_state_all([x.cpu()for x in ck['cuda_rng']])
def checkpoint(step):
 ck=dict(writer=w.state_dict(),optimizer=opt.state_dict(),step=step,run=run,execution_lock_sha256=lk,reference_receipt_sha256=reference_receipt_sha,frozen_backbone_sha256=old['frozen_backbone_sha256'],torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),data_position=step,lr=[g['lr']for g in opt.param_groups])
 save(out/'latest.pt',ck)
 if step in [0,384,768,1536]:save(out/f'checkpoint_{step}.pt',ck)
if start==0:checkpoint(0)
reference_cache={}
def reference(phase,t,f):
 key=(phase,t['target_id'])
 if key not in reference_cache:
  q,rec=load_reference(r,phase,t,f,lk,device='cpu');reference_cache[key]=(q,rec)
 q,rec=reference_cache[key]
 return [{k:v.cuda()if torch.is_tensor(v)else v for k,v in anchor.items()}for anchor in q],rec
for step in range(start,1536):
 t=train[order[step]];f,e,labels=prep.get(t,True);q,ref=reference('train',t,f)
 torch.manual_seed(run['seed']+step);opt.zero_grad(set_to_none=True)
 hook=QueryAnchorHook(model.evoformer.blocks[0].outer_product_mean,w,e,q,diagnostics=True);tick=time.monotonic()
 try:
  pred=model(f);hook.finish_forward();forward_calls=list(hook.calls)
  loss,comp=native_loss(pred,labels,config);assert torch.isfinite(loss);loss.backward()
  assert [x['grad_enabled']for x in forward_calls]==[False,False,False,True]
  assert [x['anchor_index']for x in hook.calls]==[0,1,2,3,3]
  assert all(p.grad is None for p in model.parameters())
  assert all(p.grad is not None and torch.isfinite(p.grad).all()for p in w.parameters())
  zero_names=[n for n,p in w.named_parameters()if not torch.count_nonzero(p.grad)]
  norm=torch.nn.utils.clip_grad_norm_(w.parameters(),1.,error_if_nonfinite=True);opt.step()
  assert all(torch.isfinite(p).all()for p in w.parameters())
 finally:hook.remove()
 torch.cuda.synchronize()
 row=dict(step=step+1,target_id=t['target_id'],loss=float(loss.detach()),loss_terms={k:float(v.detach())for k,v in comp.items()},gradient_norm=float(norm),forward_hook_calls=4,backward_hook_replays=len(hook.calls)-4,zero_gradient_parameters=zero_names,seconds=time.monotonic()-tick,peak_gib=torch.cuda.max_memory_allocated()/2**30,reference_sha256=ref['sha256'],injections=forward_calls)
 if(step+1)in[384,768,1536]:
  fresh,xyz=query_reference(model,f,gradient_path=True,seed=run['seed']+step)
  assert all(torch.equal(x[k],y[k])for x,y in zip(q,fresh,strict=True)for k in ['a','b','mask'])
  row['reference_replay_exact']=True;del fresh,xyz
 with(out/'training.jsonl').open('a')as fp:fp.write(json.dumps(row)+'\n')
 if(step+1)%32==0:write(out/'progress.json',row);print('STEP',run['name'],step+1,flush=True)
 if(step+1)%96==0:checkpoint(step+1)
 del pred,loss,comp,f,e,labels,q
assert state_hash(model)==old['frozen_backbone_sha256']
write(out/'training_complete.json',dict(complete=True,steps=1536,execution_lock_sha256=lk,checkpoint_sha256=sha(out/'checkpoint_1536.pt'),reference_receipt_sha256=reference_receipt_sha))
w.requires_grad_(False);folder=out/'predictions';folder.mkdir(exist_ok=True);records=[]
for t in target_list(parent,'eval'):
 tid=t['target_id'];path=folder/f'{tid}.npz';recfile=folder/f'{tid}.json';tick=time.monotonic()
 if recfile.exists():
  rec=read(recfile);assert rec['execution_lock_sha256']==lk
  if rec['status']=='ok':assert sha(path)==rec['prediction_sha256']
  records.append(rec);continue
 try:
  f,e,_=prep.get(t);q,ref=reference('eval',t,f);torch.manual_seed(20260921)
  hook=QueryAnchorHook(model.evoformer.blocks[0].outer_product_mean,w,e,q,diagnostics=True)
  try:
   with torch.no_grad():pred=model(f)
   hook.finish_forward();assert len(hook.calls)==4
   xyz=pred['final_atom_positions'].cpu().numpy();mask=pred['final_atom_mask'].cpu().numpy()
  finally:hook.remove()
  assert xyz.shape==(len(t['sequence']),37,3)and np.isfinite(xyz).all()and mask[:,1].sum()==len(t['sequence'])
  with path.with_suffix('.tmp').open('wb')as fp:np.savez(fp,coordinates=xyz,mask=mask,sequence=t['sequence'])
  path.with_suffix('.tmp').replace(path)
  rec=dict(target_id=tid,status='ok',prediction_sha256=sha(path),seconds=time.monotonic()-tick,execution_lock_sha256=lk,reference_sha256=ref['sha256'],injections=hook.calls);del pred
 except Exception:
  rec=dict(target_id=tid,status='failed',error=traceback.format_exc(),seconds=time.monotonic()-tick,execution_lock_sha256=lk);torch.cuda.empty_cache()
 write(recfile,rec);records.append(rec);print('PREDICTED',run['name'],tid,rec['status'],flush=True)
 write(out/'prediction_progress.json',dict(expected=96,completed=len(records)))
assert state_hash(model)==old['frozen_backbone_sha256']
write(out/'complete.json',dict(complete=True,execution_lock_sha256=lk,run=run,records=records,n_predictions=len(records),failures=sum(x['status']!='ok'for x in records),no_evaluation_labels_parsed=True))
