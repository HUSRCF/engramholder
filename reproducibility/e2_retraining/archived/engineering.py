"""Six real-loss, full-backbone 32-step/resume checks; no scientific tuning."""
import argparse
import time
from pathlib import Path
import torch
from common import *
from support import check, exact, rng, restore_rng
from engramfold.models.live_opm import LiveOPMHook
from engramfold.experiments.cross_backbone_protocol import schedule
from engramfold.experiments.openfold_adapter_runtime import native_loss

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
p.add_argument('--index',type=int,required=True)
p.add_argument('--phase',choices=['continuous','prefix','resume'],required=True)
a=p.parse_args();r=a.root
lock,c,lk=check(r,True);rep_sha=sha(r/'repetition_lock.json')
rid=lock['directions'][a.index//2];channels=bool(a.index%2);seed=c['formal_seeds'][0]
folder=r/'repetition_engineering'/str(a.index);folder.mkdir(parents=True,exist_ok=True)
dest=folder/(a.phase+'.json')
if dest.exists():
    done=read(dest);assert done['passed'] and done['repetition_lock_sha256']==rep_sha
    raise SystemExit('ALREADY_COMPLETE')
model,config=load_model(r,c);prep=Features(r,config)
train=read(r/'data/train96.json')['targets'];order=schedule(seed,96,1536)
w=writer(r,rid,seed,channels);opt=optimizer(w,channels)
module=model.evoformer.blocks[0].outer_product_mean
start=0;end=16 if a.phase=='prefix' else 32

def preview(step):
    state=rng()
    try:
        t=train[order[step]];f,e,labels=prep.get(t,True)
        torch.manual_seed(seed+step);h=LiveOPMHook(module,w,e)
        try:
            # Preserve the real training forward graph / last-recycle path.
            pred=model(f);loss,_=native_loss(pred,labels,config)
            assert [x['grad_enabled']for x in h.calls]==[False,False,False,True]
            result=dict(x=pred['final_atom_positions'].detach().cpu(),loss=float(loss.detach()),target=t['target_id'])
            del pred,loss
            return result
        finally:h.remove()
    finally:restore_rng(state)

resume_audit=None
if a.phase=='resume':
    ck=torch.load(folder/'prefix.pt',map_location='cpu',weights_only=False)
    assert ck['repetition_lock_sha256']==rep_sha and ck['step']==16 and ck['direction']==rid and ck['channels']==channels
    w.load_state_dict(ck['writer']);opt.load_state_dict(ck['optimizer']);restore_rng(ck['rng'])
    assert exact(w.state_dict(),ck['writer']) and exact(opt.state_dict(),ck['optimizer']) and exact(rng(),ck['rng'])
    assert ck['next_target']==train[order[16]]['target_id'] and ck['lr']==[g['lr'] for g in opt.param_groups]
    actual=preview(16);expected=ck['next_forward']
    diff=float((actual['x'].double()-expected['x'].double()).norm()/expected['x'].double().norm().clamp_min(1e-30))
    dl=abs(actual['loss']-expected['loss'])
    assert diff<=1e-6 and dl<=1e-6+1e-6*abs(expected['loss']),(diff,dl)
    resume_audit=dict(state_exact=True,optimizer_exact=True,rng_exact=True,data_position_exact=True,lr_exact=True,forward_relative=diff,loss_abs=dl)
    start=16
rows=[]
for step in range(start,end):
    t=train[order[step]];f,e,labels=prep.get(t,True);torch.manual_seed(seed+step)
    opt.zero_grad(set_to_none=True);hook=LiveOPMHook(module,w,e);tick=time.monotonic()
    try:
        pred=model(f);forward_calls=list(hook.calls);loss,comp=native_loss(pred,labels,config)
        assert torch.isfinite(loss);loss.backward()
        assert [x['grad_enabled']for x in forward_calls]==[False,False,False,True]
        assert len(hook.calls)-len(forward_calls)==1
        assert all(p.grad is None for p in model.parameters())
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in w.parameters())
        zeros=[n for n,p in w.named_parameters()if not torch.count_nonzero(p.grad)]
        grad=torch.nn.utils.clip_grad_norm_(w.parameters(),1.,error_if_nonfinite=True)
        opt.step();assert all(torch.isfinite(p).all()for p in w.parameters())
    finally:hook.remove()
    torch.cuda.synchronize()
    rows.append(dict(step=step+1,target_id=t['target_id'],loss=float(loss.detach()),gradient_norm=float(grad),zero_gradient_parameters=zeros,forward_calls=4,backward_replays=1,seconds=time.monotonic()-tick))
    print('ENGINEERING',a.index,a.phase,step+1,flush=True);del pred,loss,f,e,labels
assert state_hash(model)==c['frozen_backbone_sha256']
state=dict(writer=w.state_dict(),optimizer=opt.state_dict(),rng=rng(),step=end,direction=rid,channels=channels,repetition_lock_sha256=rep_sha,lr=[g['lr']for g in opt.param_groups])
if a.phase=='prefix':
    state.update(next_target=train[order[16]]['target_id'],next_forward=preview(16))
save(folder/(a.phase+'.pt'),state)
write(dest,dict(passed=True,repetition_lock_sha256=rep_sha,phase=a.phase,direction=rid,channels=channels,steps=len(rows),rows=rows,checkpoint_sha256=sha(folder/(a.phase+'.pt')),resume_audit=resume_audit,backbone_frozen=True))
