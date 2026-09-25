import argparse,copy,time,subprocess,sys
from pathlib import Path
import torch
from e3_common import *
from query_anchor import QueryAnchorHook,QueryAnchorCapture
from engramfold.models.live_opm import LiveOPMHook
from engramfold.experiments.openfold_adapter_runtime import native_loss
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--index',type=int,required=True);a=p.parse_args();r=a.root
c,old,parent,lk=contract(r,True);folder=r/'engineering';folder.mkdir(exist_ok=True)
unit=subprocess.run([sys.executable,'-m','pytest','-q',str(r/'staging/test_query_anchor.py')],capture_output=True,text=True)
(folder/f'unit_{a.index}.txt').write_text(unit.stdout+unit.stderr);assert unit.returncode==0
train=target_list(parent,'train');t=train[0] if a.index==0 else max(train,key=lambda x:len(x['sequence']))
model,config=pc.load_model(parent,old);prep=pc.Features(parent,config);f,e,labels=prep.get(t,True);fh=tensor_hash(f);refs={};certs={}
for phase in ['train','eval']:
 refs[phase],certs[phase]=certify_reference(model,f,phase)
 refs[phase]=[{k:v.cuda()if torch.is_tensor(v)else v for k,v in q.items()}for q in refs[phase]]
module=model.evoformer.blocks[0].outer_product_mean
checks=[]
def forward(w=None,phase='eval',kind='q',diagnostics=False):
 torch.manual_seed(20260923)
 hook=(QueryAnchorHook(module,w,e,refs[phase],diagnostics=diagnostics) if kind=='q' else LiveOPMHook(module,w,e))if w is not None else None
 try:
  with torch.set_grad_enabled(phase=='train'):pred=model(f)
  if hook and kind=='q':hook.finish_forward()
 except BaseException:
  if hook:hook.remove()
  raise
 return pred,hook
base,h=forward();baseline=base['final_atom_positions'].detach().clone();del base
for rid in ['I',*c['selected_rotations']]:
 w=pc.writer(parent,rid,old['formal_seeds'][0]);assert w.channels is None
 q,h=forward(w);assert torch.equal(q['final_atom_positions'],baseline);h.remove();del q
 opt=pc.optimizer(w,False)
 # Zero-writer full native-loss gradients agree with the preexisting live path.
 q,h=forward(w,'train');loss,_=native_loss(q,labels,config);loss.backward();assert len(h.calls)==5 and [v['anchor_index']for v in h.calls]==[0,1,2,3,3];h.remove()
 gq=[p.grad.detach().clone()for p in w.parameters()];lossq=float(loss);del q,loss
 opt.zero_grad(set_to_none=True);q,h=forward(w,'train','live');loss,_=native_loss(q,labels,config);loss.backward();h.remove()
 assert lossq==float(loss)
 assert all(torch.equal(g,p.grad)for g,p in zip(gq,w.parameters(),strict=True))
 assert all(p.grad is None for p in model.parameters())
 gn=torch.nn.utils.clip_grad_norm_(w.parameters(),1.,error_if_nonfinite=True);assert gn>0;opt.step();del q,loss
 # Atomically save nonzero state, then compare uninterrupted and resumed next step.
 save(folder/f'checkpoint_{a.index}_{rid}.pt',dict(writer=w.state_dict(),optimizer=opt.state_dict()))
 clone=pc.writer(parent,rid,old['formal_seeds'][0]);co=pc.optimizer(clone,False)
 ck=torch.load(folder/f'checkpoint_{a.index}_{rid}.pt',map_location='cpu',weights_only=False);clone.load_state_dict(ck['writer']);co.load_state_dict(ck['optimizer'])
 costs=[]
 for head,optimizer in [(w,opt),(clone,co)]:
  optimizer.zero_grad(set_to_none=True);tick=time.monotonic();q,h=forward(head,'train',diagnostics=True)
  loss,_=native_loss(q,labels,config);loss.backward();assert len(h.calls)==5;h.remove()
  assert all(p.grad is not None and torch.isfinite(p.grad).all()for p in head.parameters());torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True);optimizer.step()
  assert all(torch.isfinite(p).all()for p in head.parameters());torch.cuda.synchronize();costs.append(time.monotonic()-tick);del q,loss
 for k,v in w.state_dict().items():assert torch.equal(v,clone.state_dict()[k]),k
 for s1,s2 in zip(opt.state.values(),co.state.values(),strict=True):
  for k,v in s1.items():assert torch.equal(v.cpu(),s2[k].cpu())if torch.is_tensor(v)else v==s2[k]
 checks.append(dict(rotation=rid,zero_exact=True,initial_gradient_exact=True,resumed_next_update_exact=True,seconds_per_update=costs))
 del w,clone,opt,co
# Existing nonzero teacher: baseline comes from this actual adapted trajectory,
# while residuals use separate reference factors. Inspect the injection directly.
w=pc.teacher(parent,old,'I',old['formal_seeds'][0]);before=[];after=[];capt=QueryAnchorCapture(module)
def pre(mod,args,kwargs,out):before.append(out.detach().clone())
audit_before=module.register_forward_hook(pre,with_kwargs=True)
hook=QueryAnchorHook(module,w,e,refs['eval'],diagnostics=True)
def post(mod,args,kwargs,out):after.append(out.detach().clone())
audit_after=module.register_forward_hook(post,with_kwargs=True)
block_counts=[0]*48;handles=[]
for i,block in enumerate(model.evoformer.blocks):
 def count(mod,args,out,index=i):block_counts[index]+=1
 handles.append(block.register_forward_hook(count))
try:
 with torch.no_grad():pred=model(f)
 hook.finish_forward();assert block_counts==[4]*48
 for j,q in enumerate(refs['eval']):
  expected=before[j]+w.residual(e,q['a'],q['b'],module.linear_out.weight,q['mask'],q['eps'])
  assert torch.equal(expected,after[j])
 anchor_changes=[float((x['a']-q['a']).double().norm())for x,q in zip(capt.anchors,refs['eval'],strict=True)]
 assert anchor_changes[0]==0 and max(anchor_changes[1:])>0
 injection_rows=hook.calls
finally:
 for h in [capt,audit_before,hook,audit_after,*handles]:h.remove()
assert tensor_hash(f)==fh and state_hash(model)==old['frozen_backbone_sha256']
write(folder/f'smoke_{a.index}.json',dict(passed=True,execution_lock_sha256=lk,target=t['target_id'],length=len(t['sequence']),reference_certificates=certs,checks=checks,nonzero_teacher_anchor_changes=anchor_changes,live_baseline_plus_reference_residual_exact=True,all_evoformer_blocks_calls=block_counts,injections=injection_rows,frozen=True,feature_unchanged=True,dropout_off=all(not m.training for m in model.modules()),peak_gib=torch.cuda.max_memory_allocated()/2**30))
print('SMOKE_PASSED',t['target_id'],flush=True)
