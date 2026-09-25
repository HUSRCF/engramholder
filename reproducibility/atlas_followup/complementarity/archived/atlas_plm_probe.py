"""Bounded native/ESMC conditional distance probes; folding model stays frozen."""
from __future__ import annotations
import argparse
import copy
import gzip
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from engramfold.experiments.atlas_propagation_localization import read, write, sha

ARMS = ('native_view', 'esmc', 'esmc_permuted')
SEEDS = (20260923, 20260924, 20260925)


def atomic_torch(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp'); torch.save(value, tmp); tmp.replace(path)


def compressed_view(x, seed=20260926):
    """Fixed, label-free projection; same 128-slot auxiliary reader in all arms."""
    x = F.layer_norm(x.float(), (x.shape[-1],))
    g = torch.Generator().manual_seed(seed + x.shape[-1])
    p = torch.randn(x.shape[-1], 128, generator=g) / x.shape[-1] ** .5
    return F.layer_norm(x @ p.to(x), (128,))


def permutation(tid, length):
    seed = int(hashlib.sha256(('atlas_probe_permutation_v1:' + tid).encode()).hexdigest()[:15], 16)
    return torch.randperm(length, generator=torch.Generator().manual_seed(seed))


class Probe(nn.Module):
    def __init__(self):
        super().__init__()
        self.native = nn.Sequential(nn.Linear(768, 64), nn.GELU())
        self.aux = nn.Sequential(nn.Linear(128, 64), nn.GELU())
        self.pair = nn.Sequential(nn.Linear(513, 128), nn.GELU(), nn.Linear(128, 64))
        nn.init.zeros_(self.pair[-1].weight); nn.init.zeros_(self.pair[-1].bias)

    def forward(self, data, pairs, arm):
        s, z = data['s'], data['z']
        aux = data['native_view'] if arm == 'native_view' else data['esmc_view']
        if arm == 'esmc_permuted': aux = aux[data['permutation']]
        h = self.native(F.layer_norm(s.float(), (768,)))
        v = self.aux(aux.float())
        i, j = pairs.T
        zz = (z[i, j].float() + z[j, i].float()) * .5
        parts = [F.layer_norm(zz, (128,))]
        for x in (h, v):
            parts.extend((x[i]+x[j], (x[i]-x[j]).abs(), x[i]*x[j]))
        parts.append(((i-j).abs().float().clamp(max=128)/128)[:, None])
        return self.pair(torch.cat(parts, -1))


def make_labels(xyz, mask, beta, length, boundaries):
    xyz, mask = xyz[:length], mask[:length]
    idx = torch.arange(length)
    x, valid = xyz[idx, beta], mask[idx, beta]
    # Match pinned native strict '>' bin assignment, including exact boundaries.
    d = (x[:, None] - x[None, :]).norm(dim=-1)
    y = (d[..., None] > boundaries).sum(-1).long()
    pairs = torch.triu(valid[:, None] & valid[None, :], diagonal=1).nonzero()
    assert len(pairs) > 0
    return dict(pairs=pairs, labels=y[pairs[:, 0], pairs[:, 1]],
                long=(pairs[:, 1]-pairs[:, 0]) >= 24,
                close=d[pairs[:, 0], pairs[:, 1]] < 15.)


def init(root):
    c=read(root/'lock.json')
    for p,h in c['code_files'].items(): assert sha(p)==h,p
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    return c,sha(root/'lock.json')


def cache_features(root, index):
    c,lk=init(root)
    from engramfold.experiments.atlas_adapter_runtime import load_model,state_hash
    from engramfold.experiments.atlas_posttraining_calibration.predict import install_label_guard
    from engramfold.experiments.plm_extension_runtime import load_esmc
    from atlasfold.common import featurize
    from atlasfold.runner import seed_context,autocast_context,get_sampling_config
    from atlasfold.pretrained import get_runner
    base=Path(c['base']);out=root/'cache';out.mkdir(exist_ok=True)
    blocked=install_label_guard(root,read(Path(c['calibration_root'])/'execution_lock.json'),out)
    model=load_model(base).eval();frozen=state_hash(model);assert frozen==c['frozen_hash']
    runner=get_runner(model)
    targets=c['train']+c['dev'];counts=0
    for k,t in enumerate(targets):
        if k%2 != index:continue
        tid=t['target_id'];dest=out/f'{tid}.pt';receipt=out/f'{tid}.json'
        if receipt.exists():
            rec=read(receipt);assert rec['lock_sha256']==lk and rec['sha256']==sha(dest);counts+=1;continue
        start=time.monotonic();L=len(t['sequence'])
        # Reuse the inference runner's exact buckets and special-token padding.
        # Training's pad_to_multiple_of=4 is a different numerical input shape.
        bucket=runner._get_length_bucket(L)
        batch=runner._make_batch_features([t['sequence']],bucket)
        feat={key:torch.from_numpy(v).cuda() for key,v in batch.items()}
        with torch.inference_mode(),seed_context(1,model.device),autocast_context(model.device):
            mlm=model.sample_mlm_mask(feat,.15)
            s,z=model.run_lm_embedder(feat,mlm)
        s=s[0,:L].detach().cpu().clone();z=z[0,:L,:L].detach().cpu().clone()
        assert s.shape==(L,768) and z.shape==(L,L,128)
        for x in (s,z):assert torch.isfinite(x).all()
        e=load_esmc(Path(c['feature_root']),tid,t['sequence_sha256'],L)
        data=dict(s=s,z=z,native_view=compressed_view(s),esmc_view=compressed_view(e),
                  permutation=permutation(tid,L),sequence_sha256=t['sequence_sha256'],lock_sha256=lk)
        # Two Train-only engineering targets, fixed before any probe fitting.
        if tid in c['engineering_targets']:
            captured=[]
            def hook(m,args,value):
                if not captured:captured.extend([v.detach().cpu().clone() for v in value])
            handle=model.lm_stack.register_forward_hook(hook)
            try:
                runner.fold(tid,t['sequence'],num_samples=1,seeds=1,num_recycles=4,mlm_prob=.15,sampling_config=get_sampling_config(L))
            finally:handle.remove()
            assert torch.equal(s,captured[0][0,:L]) and torch.equal(z,captured[1][0,:L,:L])
            write(out/f'{tid}_engineering.json',dict(passed=True,standalone_equals_first_full_forward_pass=True))
        atomic_torch(dest,data)
        write(receipt,dict(complete=True,sha256=sha(dest),lock_sha256=lk,length=L,seconds=time.monotonic()-start))
        counts+=1;print('CACHED',index,counts,tid,flush=True)
    assert state_hash(model)==frozen and not blocked
    write(out/f'worker{index}_complete.json',dict(complete=True,targets=counts,frozen_hash=frozen,lock_sha256=lk))


def labels(root):
    c,lk=init(root)
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    from atlasfold.common import residue_constants as rc
    from engramfold.experiments.atlas_adapter_runtime import atom14_labels
    base=Path(c['base']);out=root/'labels';out.mkdir(exist_ok=True)
    boundaries=torch.linspace(2.3125,21.6875,63)
    for t in c['train']+c['dev']:
        tid=t['target_id'];src=base/'data'/f'{tid}.cif.gz'
        assert sha(src)==c['reference_hashes'][str(src)]
        with gzip.open(src,'rt') as f:xyz,mask=atom14_labels(MMCIF2Dict(f),t,rc)
        beta=torch.tensor([rc.restype_atom14_order[rc.restype_1to3[aa]]['CA' if aa=='G' else 'CB'] for aa in t['sequence']])
        data=make_labels(torch.from_numpy(xyz),torch.from_numpy(mask),beta,len(t['sequence']),boundaries)
        data.update(sequence_sha256=t['sequence_sha256'],lock_sha256=lk)
        atomic_torch(out/f'{tid}.pt',data)
    write(out/'complete.json',dict(complete=True,lock_sha256=lk,targets=104,
          hashes={p.name:sha(p)for p in out.glob('*.pt')}))


def load_data(root,c):
    data={};labels_done=read(root/'labels/complete.json')
    assert labels_done['lock_sha256']==sha(root/'lock.json')
    for t in c['train']+c['dev']:
        tid=t['target_id'];path=root/'cache'/f'{tid}.pt';lp=root/'labels'/f'{tid}.pt'
        assert sha(path)==read(path.with_suffix('.json'))['sha256']
        assert sha(lp)==labels_done['hashes'][lp.name]
        x=torch.load(path,map_location='cpu',weights_only=False);y=torch.load(lp,map_location='cpu',weights_only=False)
        assert x['sequence_sha256']==y['sequence_sha256']==t['sequence_sha256']
        assert x['lock_sha256']==y['lock_sha256']==sha(root/'lock.json')
        data[tid]={**x,**y}
    return data


def to_gpu(d):
    return {k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in d.items()}


def step(model,opt,d,arm,seed,k):
    pairs=d['pairs'];g=torch.Generator().manual_seed(seed*100000+k)
    ix=torch.randperm(len(pairs),generator=g)[:4096].to(pairs.device)
    opt.zero_grad(set_to_none=True)
    loss=F.cross_entropy(model(d,pairs[ix],arm),d['labels'][ix]);assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    opt.step();assert all(torch.isfinite(p).all() for p in model.parameters())
    return dict(loss=float(loss.detach()),grad_norm=float(norm))


@torch.no_grad()
def evaluate(model,data,targets,arm):
    model.eval();rows=[]
    for t in targets:
        d=to_gpu(data[t['target_id']]);losses=[]
        for chunk in d['pairs'].split(8192):
            start=sum(len(v) for v in losses)
            losses.append(F.cross_entropy(model(d,chunk,arm),d['labels'][start:start+len(chunk)],reduction='none').double().cpu())
        vals=torch.cat(losses);long=d['long'].cpu();close=d['close'].cpu()
        rows.append(dict(target_id=t['target_id'],ce=float(vals.mean()),pair_count=len(vals),
                         long_ce=float(vals[long].mean()) if long.any() else None,
                         close_ce=float(vals[close].mean()) if close.any() else None))
    model.train();return rows


def smoke(root):
    c,lk=init(root);data=load_data(root,c);out=root/'smoke';out.mkdir(exist_ok=True)
    values=[]
    for arm in ARMS:
        torch.manual_seed(SEEDS[0]);model=Probe().cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        count=sum(p.numel()for p in model.parameters());logs=[];mid=None
        for k in range(8):
            tid=c['engineering_targets'][k%2];d=to_gpu(data[tid]);logs.append(step(model,opt,d,arm,SEEDS[0],k))
            if k==3:mid=copy.deepcopy(dict(model=model.state_dict(),optimizer=opt.state_dict()))
        end={k:v.clone() for k,v in model.state_dict().items()}
        model.load_state_dict(mid['model']);opt.load_state_dict(mid['optimizer'])
        replay=[]
        for k in range(4,8):
            d=to_gpu(data[c['engineering_targets'][k%2]]);replay.append(step(model,opt,d,arm,SEEDS[0],k))
        assert replay==logs[4:] and all(torch.equal(end[k],v)for k,v in model.state_dict().items())
        assert logs[-1]['loss']<np.log(64)
        # Main graph uses every parameter; no requirement for each tensor to be nonzero at step1.
        values.append(dict(arm=arm,parameters=count,continuous_updates=8,resume_updates=4,exact_resume=True,logs=logs))
    assert len({v['parameters']for v in values})==1
    write(out/'complete.json',dict(passed=True,lock_sha256=lk,runs=values,physical_updates=36))


def train(root,index):
    c,lk=init(root);assert read(root/'smoke/complete.json')['passed']
    seed=SEEDS[index//3];arm=ARMS[index%3];name=f'{arm}_s{seed}';out=root/'formal'/name;out.mkdir(parents=True,exist_ok=True)
    import fcntl
    handle=(out/'lease').open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'complete.json').exists():assert read(out/'complete.json')['lock_sha256']==lk;return
    data=load_data(root,c)
    from engramfold.experiments.cross_backbone_protocol import schedule
    order=schedule(seed,96,c['steps']);torch.manual_seed(seed)
    model=Probe().cuda();opt=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=c['weight_decay']);start=0;logs=[]
    if (out/'latest.pt').exists():
        ck=torch.load(out/'latest.pt',weights_only=False);assert ck['lock_sha256']==lk
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);start=ck['step'];logs=ck['logs']
        torch.set_rng_state(ck['rng'].cpu());torch.cuda.set_rng_state(ck['cuda_rng'].cpu())
    else:
        write(out/'dev_step0.json',evaluate(model,data,c['dev'],arm))
    if start in (384,768,1536) and not (out/f'dev_step{start}.json').exists():
        write(out/f'dev_step{start}.json',evaluate(model,data,c['dev'],arm))
    for k in range(start,c['steps']):
        tid=c['train'][order[k]]['target_id'];d=to_gpu(data[tid]);tick=time.monotonic()
        row=dict(step=k+1,target_id=tid,**step(model,opt,d,arm,seed,k));torch.cuda.synchronize()
        row['seconds']=time.monotonic()-tick;logs.append(row);del d
        if (k+1)%128==0 or k+1==c['steps']:
            atomic_torch(out/'latest.pt',dict(model=model.state_dict(),optimizer=opt.state_dict(),step=k+1,logs=logs,lock_sha256=lk,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()))
            write(out/'progress.json',row);print(name,k+1,flush=True)
        if k+1 in (384,768,1536):
            write(out/f'dev_step{k+1}.json',evaluate(model,data,c['dev'],arm))
    write(out/'train_final.json',evaluate(model,data,c['train'],arm))
    write(out/'complete.json',dict(complete=True,lock_sha256=lk,arm=arm,seed=seed,steps=c['steps'],parameters=sum(p.numel()for p in model.parameters()),checkpoint_sha256=sha(out/'latest.pt')))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--stage',choices=['features','labels','smoke','train'],required=True);p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.stage=='features':cache_features(a.root,a.index)
    elif a.stage=='labels':labels(a.root)
    elif a.stage=='smoke':smoke(a.root)
    else:train(a.root,a.index)


if __name__=='__main__':main()
