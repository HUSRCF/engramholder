"""Preserve the original per-global-step training function and AdamW state."""
import argparse,copy,fcntl,json,os,time
from pathlib import Path
from types import SimpleNamespace
import torch
from .common import ROOT,CACHE,RUNTIME,A,locked,read,write,sha,stamp
from .state import rng,restore_rng,exact,state_hash
from engramfold.experiments.train_interface_control import save_atomic,schedule
from engramfold.experiments.train_structure_control import frozen_digest,make_runner
from engramfold.experiments.interface_runtime import native_task_loss
from engramfold.experiments.plm_extension_runtime import load_esmc
from engramfold.models.interface_heads import FrozenOPMDecoder
from engramfold.models.plm_extension import make_writer


def validate_optimizer(opt,expected_step):
 for g in opt['param_groups']:
  assert g['lr']==5e-5 and g['weight_decay']==1e-4 and tuple(g['betas'])==(.9,.999) and g['eps']==1e-8
 ids=[i for g in opt['param_groups'] for i in g['params']]
 assert len(ids)==len(set(ids))==len(opt['state']) and set(ids)==set(opt['state'])
 for v in opt['state'].values():
  assert float(v['step'])==expected_step
  assert v['exp_avg'].shape==v['exp_avg_sq'].shape
  assert torch.isfinite(v['exp_avg']).all() and torch.isfinite(v['exp_avg_sq']).all()


def run(args):
 c,lockhash=locked();row=c['parents'][args.system]
 if sha(row['resume_path'])!=row['resume_sha256']:raise ValueError('parent resume changed')
 parent=torch.load(row['resume_path'],map_location='cpu',weights_only=False)
 validate_optimizer(parent['optimizer'],1536)
 index=read(CACHE/'index.json');ids=index['train384_ids'];order=schedule(ids,parent['seed'],3072)
 assert len(ids)==384 and order[:1536]==[x['target_id'] for x in parent['logs']]
 if args.mode!='formal':
  engineering_ids=[t['target_id'] for t in read(ROOT/'inputs/engineering.json')['targets']]
  order[1536:1544]=engineering_ids*4
 out=ROOT/('formal' if args.mode=='formal' else 'engineering/training')/args.system
 out.mkdir(parents=True,exist_ok=True)
 lease=(out/'lease.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
 receipt=out/('complete.json' if args.mode=='formal' else args.mode+'_complete.json')
 if receipt.exists():
  old=read(receipt);assert old['execution_lock_sha256']==lockhash
  for p,h in old['files'].items():assert sha(p)==h
  return
 if args.mode=='formal':
  gate=read(ROOT/'engineering/complete.json');assert gate['execution_lock_sha256']==lockhash and gate['passed']
 state=parent
 if args.mode=='smoke_replay':state=torch.load(out/'split_1540.pt',map_location='cpu',weights_only=False)
 elif (out/'resume.pt').exists():state=torch.load(out/'resume.pt',map_location='cpu',weights_only=False)
 if state['step']!=1536:
  assert state['continuation']['execution_lock_sha256']==lockhash
  assert state['continuation']['parent_resume_sha256']==row['resume_sha256']
  assert state['global_order']==order
 validate_optimizer(state['optimizer'],state['step'])
 os.environ['PROTENIX_ROOT_DIR']=str(RUNTIME)
 from protenix.model.loss import ProtenixLoss
 from protenix.utils.permutation.permutation import SymmetricPermutation
 from protenix.utils.seed import seed_everything
 from protenix.utils.torch_utils import to_device
 runner=make_runner(SimpleNamespace(protenix_root=RUNTIME,seed=parent['seed']))
 model=runner.model.requires_grad_(False).train();original=model.msa_module;before=frozen_digest(model)
 common=torch.load(CACHE/'common.pt',map_location=runner.device,weights_only=False)
 decoder=FrozenOPMDecoder(common['weight'],common['bias'],factor_dim=32,depth=common['depth'],eps=common['eps']).to(runner.device)
 seed_everything(seed=parent['seed'],deterministic=True)
 writer=make_writer('protenix',parent['geometry']['kind'],parent['geometry']['rotation_seed'],parent['seed'],1152,runner.device)
 writer.load_state_dict(state['writer'],strict=True)
 optimizer=torch.optim.AdamW(writer.parameters(),lr=5e-5,weight_decay=1e-4,betas=(.9,.999),eps=1e-8)
 optimizer.load_state_dict(state['optimizer'])
 assert exact(optimizer.state_dict(),state['optimizer']) and exact(writer.state_dict(),state['writer'])
 configs=runner.configs;configs.loss.weight.alpha_bond=1.;configs.loss.weight.alpha_confidence=0.;configs.atom_permutation.train.diffusion_sample=True
 lossfn=ProtenixLoss(configs);permutation=SymmetricPermutation(configs,error_dir=str(out/'permutation_errors'))
 logs=list(state['logs']);start=state['step']+1;end=3072 if args.mode=='formal' else 1544
 if 'rng' in state:restore_rng(state['rng'])
 contract={k:v for k,v in parent.items() if k not in ('writer','optimizer','logs','step','rng','global_order')}
 contract.update(budget=3072,trainer_sha256=sha(Path(__file__)),continuation=dict(execution_lock_sha256=lockhash,parent_resume_sha256=row['resume_sha256'],parent_task_sha256=row['task_sha256'],historical_trainer_sha256=parent['trainer_sha256'],historical_rng_unavailable=True,step_seed='training_seed + global_step'))
 expected=read(out/'smoke_trace.json') if args.mode=='smoke_replay' else {}
 trace=read(out/'smoke_trace.json') if (out/'smoke_trace.json').exists() else {}
 for step in range(start,end+1):
  tid=order[step-1];ts=time.monotonic()
  assert sha(CACHE/(tid+'.pt'))==index['records'][tid]['sha256'],'target cache changed'
  data=torch.load(CACHE/(tid+'.pt'),map_location='cpu',weights_only=False,mmap=True);rec=index['records'][tid]
  data['plm']=load_esmc(A/'features/esmc',tid,rec['sequence_sha256'],rec['length']);data=to_device(data,runner.device)
  torch.cuda.synchronize();transfer=time.monotonic()-ts
  optimizer.zero_grad(set_to_none=True);seed_everything(seed=parent['seed']+step,deterministic=True);torch.cuda.reset_peak_memory_stats();ts=time.monotonic()
  update=writer(data['plm'],decoder=decoder,**data['query'])
  loss,info=native_task_loss(model,original,copy.deepcopy(data['features']),copy.deepcopy(data['labels']),update,lossfn,permutation,audit_gradient=False)
  loss.backward()
  assert all(p.grad is None for p in model.parameters()),'frozen model has gradients'
  disconnected=[n for n,p in writer.named_parameters() if p.requires_grad and p.grad is None]
  if disconnected:raise RuntimeError('unexpected disconnected parameters: '+str(disconnected))
  diag={n:dict(nonzero=int(torch.count_nonzero(p.grad)),norm64=float(p.grad.double().norm())) for n,p in writer.named_parameters() if 'output_head' in n}
  norm=float(torch.nn.utils.clip_grad_norm_(writer.parameters(),1.,error_if_nonfinite=True));optimizer.step()
  assert all(torch.isfinite(p).all() for p in writer.parameters())
  torch.cuda.synchronize();model.msa_module=original
  entry=dict(stage='task',step=step,target_id=tid,loss=float(loss.detach()),seconds=time.monotonic()-ts,transfer_seconds=transfer,peak_gib=torch.cuda.max_memory_allocated()/2**30,unclipped_grad_norm=norm,output_gradients=diag,**info)
  assert info['pair_stack_forward_calls']==4
  logs.append(entry);print(json.dumps(entry),flush=True);write(out/('replay_progress.json' if args.mode=='smoke_replay' else 'progress.json'),entry)
  if args.mode!='formal':
   observed=dict(loss=entry['loss'],target_id=tid,writer_sha256=state_hash(writer),unclipped_grad_norm=norm)
   if args.mode=='smoke_replay':assert exact(observed,expected[str(step)]),('restore trajectory differs',step,observed,expected[str(step)])
   else:trace[str(step)]=observed;write(out/'smoke_trace.json',trace)
  if step%128==0 or step in (1540,1544,2304,3072):
   payload={**contract,'step':step,'writer':writer.state_dict(),'optimizer':optimizer.state_dict(),'logs':logs,'rng':rng(),'global_order':order}
   if args.mode=='smoke_replay':save_atomic(payload,out/'replay_final.pt')
   else:
    save_atomic(payload,out/'resume.pt')
    if args.mode=='smoke' and step==1540:save_atomic(payload,out/'split_1540.pt')
    if args.mode=='formal' and step in (2304,3072):save_atomic({k:v for k,v in payload.items() if k not in ('optimizer','rng')},out/f'task_{step}.pt')
  del data,update,loss
 assert frozen_digest(model)==before,'frozen parameters changed'
 validate_optimizer(optimizer.state_dict(),end)
 if args.mode=='smoke_replay':
  reference=torch.load(out/'resume.pt',map_location='cpu',weights_only=False)
  assert exact(writer.state_dict(),reference['writer']) and exact(optimizer.state_dict(),reference['optimizer']) and exact(rng(),reference['rng'])
 files=[out/('replay_final.pt' if args.mode=='smoke_replay' else 'resume.pt')]
 if args.mode=='formal':files.extend(out/f'task_{s}.pt' for s in (2304,3072))
 write(receipt,dict(complete=True,passed=True,mode=args.mode,system=args.system,step=end,new_updates=end-1536 if args.mode!='smoke_replay' else 4,execution_lock_sha256=lockhash,files={str(p):sha(p) for p in files},frozen_unchanged=True,exact_restore=args.mode=='smoke_replay',time=stamp()))

def main():
 p=argparse.ArgumentParser();p.add_argument('--system',required=True);p.add_argument('--mode',choices=['smoke','smoke_replay','formal'],required=True);run(p.parse_args())
if __name__=='__main__':main()
