"""Bounded Train-only numerical diagnosis using frozen, hash-checked historical assets.

The default orchestrator runs two independent workers for 3 updates on one allocated
GPU. It extends both from their own step-3 states to step 8 only if all saved
forward tensors, gradients, parameters, and optimizer tensors were identical.
It never reads evaluation targets, predictions, or scores.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import torch

def cpu(x):
    if torch.is_tensor(x): return x.detach().cpu().clone()
    if isinstance(x,dict): return {k:cpu(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return type(x)(cpu(v) for v in x)
    return copy.deepcopy(x)

def digest(t):
    return hashlib.sha256(t.reshape(-1).contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

def summary(t):
    z=t.double()
    return dict(shape=list(t.shape),dtype=str(t.dtype),sha256=digest(t),
                norm64=float(z.norm()),maxabs=float(z.abs().max()) if z.numel() else 0.)

def save_json(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def flatten(x,prefix=''):
    if torch.is_tensor(x): return {prefix:x}
    if isinstance(x,dict):
        return {k:v for name,value in x.items() for k,v in flatten(value,prefix+'/'+str(name)).items()}
    if isinstance(x,(list,tuple)):
        return {k:v for name,value in enumerate(x) for k,v in flatten(value,prefix+'/'+str(name)).items()}
    return {}

def compare(a,b):
    aa,bb=flatten(a),flatten(b)
    result=[]
    for name in sorted(set(aa)|set(bb)):
        x,y=aa.get(name),bb.get(name)
        if x is None or y is None:
            result.append(dict(name=name,exact=False,missing='a' if x is None else 'b'));continue
        eq=x.dtype==y.dtype and x.shape==y.shape and torch.equal(x,y)
        delta=x.double()-y.double()
        result.append(dict(name=name,exact=eq,a=summary(x),b=summary(y),
                           relative64=float(delta.norm()/x.double().norm().clamp_min(1e-30)),
                           maxabs64=float(delta.abs().max()) if delta.numel() else 0.))
    return dict(all_exact=all(x['exact'] for x in result),tensors=result)

def rng():
    return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),
                numpy=np.random.get_state(),python=random.getstate())

def restore_rng(x):
    torch.set_rng_state(x['torch']);torch.cuda.set_rng_state_all(x['cuda'])
    np.random.set_state(x['numpy']);random.setstate(x['python'])

def worker(a):
    r=a.root
    sys.path[:0]=[str(r/'staging'),str(r/'vendor/openfold'),str(r/'source/src'),str(r/'source/scripts')]
    from common import contract,load_model,Features,writer,optimizer,read,state_hash
    from engramfold.models.live_opm import LiveOPMHook
    from engramfold.experiments.cross_backbone_protocol import schedule
    from engramfold.experiments.openfold_adapter_runtime import native_loss
    c,lk=contract(r,gpu=True)
    if a.condition=='deterministic_conv':
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False
    elif a.condition=='no_cudnn':
        torch.backends.cudnn.enabled=False
    # No deterministic-algorithms or cuBLAS override is silently applied here.
    run=dict(name='I_C_s20260923',rotation_id='I',seed=20260923)
    seed=run['seed']
    ck0=torch.load(r/'e2/formal'/run['name']/'checkpoint_0.pt',map_location='cpu',weights_only=False)
    assert ck0['step']==0 and ck0['execution_lock_sha256']==lk
    out=a.out/a.condition/f'trial_{a.worker}'
    out.mkdir(parents=True,exist_ok=True)
    assert not (out/f'step_{a.end:02d}.pt').exists(),'Refuse to overwrite a diagnosis trial'
    model,config=load_model(r,c);prep=Features(r,config)
    w=writer(r,'I',seed,True);opt=optimizer(w,True)
    init_cmp=compare(ck0['writer'],cpu(w.state_dict()))
    assert init_cmp['all_exact'],'Fresh adapter differs from historical step zero'
    w.load_state_dict(ck0['writer']);opt.load_state_dict(ck0['optimizer'])
    torch.set_rng_state(ck0['torch_rng']);torch.cuda.set_rng_state_all(ck0['cuda_rng'])
    train=read(r/'data/train96.json')['targets'];order=schedule(seed,96,1536)
    historical_schedule=read(r/'e2/formal'/run['name']/'schedule.json')
    assert [train[i]['target_id'] for i in order]==historical_schedule
    start=0
    if a.end==8:
        previous=torch.load(out/'step_03.pt',map_location='cpu',weights_only=False)
        w.load_state_dict(previous['post']);opt.load_state_dict(previous['optimizer']);restore_rng(previous['rng'])
        start=3
    environment=dict(torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
        device=torch.cuda.get_device_name(),device_properties=str(torch.cuda.get_device_properties(0)),
        node=os.uname().nodename,slurm_job=os.environ.get('SLURM_JOB_ID'),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_enabled=torch.backends.cudnn.enabled,cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,
        environment={k:os.environ.get(k) for k in ['CUBLAS_WORKSPACE_CONFIG','CUDA_VISIBLE_DEVICES','PYTHONHASHSEED','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','CUDA_HOME']})
    save_json(out/f'environment_{a.end}.json',environment)
    torch.save(dict(writer=cpu(w.state_dict()),optimizer=cpu(opt.state_dict()),rng=cpu(rng())),out/f'start_{start}.pt')
    for step in range(start,a.end):
        t=train[order[step]];f,e,labels=prep.get(t,True)
        torch.manual_seed(seed+step);opt.zero_grad(set_to_none=True)
        before=cpu(w.state_dict());trace=[];handles=[]
        # Copying detached small module outputs synchronizes kernels but changes no
        # values, RNG or loss. The trace is an instrumented reproduction, not a claim
        # to recover the exact historical GPU launch schedule.
        for name,module in w.named_modules():
            if name in ['encoder.embedding','encoder.global_projection','output_head'] or name.startswith('encoder.convolutions.'):
                def hook(module,args,output,name=name):
                    trace.append(dict(name=name,grad_enabled=torch.is_grad_enabled(),input=cpu(args),output=cpu(output)))
                handles.append(module.register_forward_hook(hook))
        original_matrix=w.channels.matrix
        def matrix_trace():
            value=original_matrix()
            trace.append(dict(name='channels.matrix',grad_enabled=torch.is_grad_enabled(),output=cpu(value)))
            return value
        w.channels.matrix=matrix_trace
        live=LiveOPMHook(model.evoformer.blocks[0].outer_product_mean,w,e)
        try:
            pred=model(f);calls=list(live.calls);loss,terms=native_loss(pred,labels,config)
            assert torch.isfinite(loss)
            coordinates=cpu(pred['final_atom_positions'])
            losses={k:float(v.detach()) for k,v in terms.items()}
            loss.backward()
            assert [x['grad_enabled'] for x in calls]==[False,False,False,True]
            assert all(p.grad is None for p in model.parameters())
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in w.parameters())
            gradients={n:cpu(p.grad) for n,p in w.named_parameters()}
            norm=torch.nn.utils.clip_grad_norm_(w.parameters(),1.,error_if_nonfinite=True)
            clipped={n:cpu(p.grad) for n,p in w.named_parameters()}
            opt.step()
            assert all(torch.isfinite(p).all() for p in w.parameters())
        finally:
            live.remove()
            for h in handles:h.remove()
            w.channels.matrix=original_matrix
        state=dict(step=step+1,target=t['target_id'],pre=before,post=cpu(w.state_dict()),
                   gradients=gradients,clipped_gradients=clipped,optimizer=cpu(opt.state_dict()),
                   rng=cpu(rng()),trace=trace,coordinates=coordinates,
                   inputs={name:summary(value) for name,value in flatten(cpu(dict(features=f,esm=e,labels=labels))).items()},
                   loss=float(loss.detach()),terms=losses,gradient_norm=float(norm),
                   calls=calls,backward_replays=len(live.calls)-len(calls))
        state['updates64']={n:state['post'][n].double()-before[n].double() for n,_ in w.named_parameters()}
        torch.save(state,out/f'step_{step+1:02d}.pt')
        save_json(out/f'step_{step+1:02d}.json',{k:v for k,v in state.items() if k in ['step','target','inputs','loss','terms','gradient_norm','calls','backward_replays']})
        print('TRACE',a.condition,a.worker,step+1,t['target_id'],flush=True)
        del pred,loss,terms,coordinates,state,gradients,clipped,before,trace,f,e,labels
    assert state_hash(model)==c['frozen_backbone_sha256']
    save_json(out/f'complete_{a.end}.json',dict(complete=True,steps=a.end-start,backbone_frozen=True,condition=a.condition))

def compare_trials(a,end):
    folder=a.out/a.condition
    result={'end':end,'all_exact':True,'steps':[]}
    for step in range(1,end+1):
        x=torch.load(folder/'trial_0'/f'step_{step:02d}.pt',map_location='cpu',weights_only=False)
        y=torch.load(folder/'trial_1'/f'step_{step:02d}.pt',map_location='cpu',weights_only=False)
        item={'step':step,'target':x['target'],'inputs_exact':x['inputs']==y['inputs'],
              'loss_exact':x['loss']==y['loss'],'terms_exact':x['terms']==y['terms'],
              'trace_metadata_exact':[(v['name'],v['grad_enabled']) for v in x['trace']]==[(v['name'],v['grad_enabled']) for v in y['trace']],
              'parts':{k:compare(x[k],y[k]) for k in ['pre','post','updates64','gradients','clipped_gradients','optimizer','trace','coordinates']}}
        item['parts']['torch_cuda_rng']=compare({k:x['rng'][k] for k in ['torch','cuda']},{k:y['rng'][k] for k in ['torch','cuda']})
        item['all_exact']=item['inputs_exact'] and item['loss_exact'] and item['terms_exact'] and item['trace_metadata_exact'] and all(z['all_exact'] for z in item['parts'].values())
        result['all_exact'] &= item['all_exact'];result['steps'].append(item)
    save_json(folder/f'comparison_{end}.json',result)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--condition',choices=['historical','deterministic_conv','no_cudnn'],default='historical')
    p.add_argument('--worker',type=int,choices=[0,1]);p.add_argument('--end',type=int,choices=[3,8],default=3)
    a=p.parse_args();torch.set_num_threads(8)
    assert a.out.resolve()!=a.root.resolve() and a.root.resolve() not in a.out.resolve().parents
    if a.worker is not None:worker(a);return
    for end in [3,8]:
        for trial in [0,1]:
            subprocess.run([sys.executable,'-u',__file__,'--root',str(a.root),'--out',str(a.out),
                '--condition',a.condition,'--worker',str(trial),'--end',str(end)],check=True)
        result=compare_trials(a,end)
        print('COMPARISON',a.condition,end,result['all_exact'],flush=True)
        if not result['all_exact']:break

if __name__=='__main__':main()
