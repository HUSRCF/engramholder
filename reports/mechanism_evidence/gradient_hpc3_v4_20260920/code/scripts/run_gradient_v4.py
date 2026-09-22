"""Versioned dual-loss diagnostic; no optimizer and no confirmation training."""
import argparse
import copy
import datetime
import hashlib
import json
import os
import socket
import time
import traceback
from pathlib import Path
os.environ.setdefault('LAYERNORM_TYPE', 'torch')
import torch
import torch.utils.checkpoint
import yaml
from ml_collections import ConfigDict
from protenix.model.loss import ProtenixLoss
from protenix.model.protenix import Protenix
from protenix.utils.permutation.permutation import SymmetricPermutation
from protenix.utils.seed import seed_everything
from protenix.utils.torch_utils import to_device
from engramfold.experiments.gradient_runtime import restore
from engramfold.experiments.gradient_runtime_v3 import DualLossDiagnostic
from engramfold.experiments.learned_direction import FrozenStudents
from engramfold.experiments.interface_runtime import native_task_loss
from engramfold.experiments.train_structure_control import frozen_digest
from engramfold.evaluation.tangent_projection import TangentMap, ridge_projection
from engramfold.models.native_geometry import orthogonal_rotation
from engramfold.experiments.sequence_feature_cache import file_sha256


def write(path, value):
    tmp = path.with_suffix(path.suffix + '.part')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def cpu(x):
    if torch.is_tensor(x): return x.detach().cpu()
    if isinstance(x, dict): return {k: cpu(v) for k, v in x.items()}
    if isinstance(x, tuple): return tuple(cpu(v) for v in x)
    if isinstance(x, list): return [cpu(v) for v in x]
    return x


def adjacent_ok(rows, key):
    flags = [r[key] <= .05 and r['resolved_' + key.split('_')[0]] for r in rows]
    return any(a and b for a,b in zip(flags,flags[1:]))


def main():
    p=argparse.ArgumentParser()
    for name in ['fixture','cache','output','lock']: p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--targets',nargs='+',required=True)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--gate',type=Path)
    p.add_argument('--components',action='store_true')
    for key in ['inventory','features','manifest']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    lock=json.loads(a.lock.read_text())
    cache_index=json.loads((a.cache/'index.json').read_text())
    assert cache_index['schema'] in ['engramfold.native_task_cache.v1','engramfold.offline_diagnostic_cache.v1']
    assert file_sha256(a.cache/'common.pt')==cache_index['common_sha256']
    assert file_sha256(a.fixture/'mini.pt')==cache_index['protenix_checkpoint_sha256']
    for target in a.targets:
        assert file_sha256(a.cache/f'{target}.pt')==cache_index['records'][target]['sha256']
    if not a.smoke:
        assert a.gate and json.loads(a.gate.read_text())['passed'], 'runtime gate required'
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    raw=json.loads((a.fixture/'config.json').read_text())
    raw={k:yaml.safe_load(v) if isinstance(v,str) and '\n' in v else v for k,v in raw.items()}
    cfg=ConfigDict(raw)
    cfg.loss.weight.alpha_bond=1.; cfg.loss.weight.alpha_confidence=0.
    cfg.atom_permutation.train.diffusion_sample=True
    cfg.mc_dropout_apply_rate=0.; cfg.mc_dropout_rate=0.
    model=Protenix(cfg)
    state=torch.load(a.fixture/'mini.pt',map_location='cpu',weights_only=False)['model']
    model.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True)
    del state
    model=model.cuda().requires_grad_(False).train()
    original=model.msa_module; digest=frozen_digest(model)
    loss_fn=ProtenixLoss(cfg)
    permutation=SymmetricPermutation(cfg,error_dir=str(a.output/'permutation'))
    c=torch.load(a.cache/'common.pt',map_location='cuda',weights_only=False)
    w=c['weight'].double().reshape(128,32,32)*(c['depth']/(c['depth']+c['eps']))
    environment={'host':socket.gethostname(),'torch':torch.__version__,'cuda':torch.version.cuda,
                 'hip':torch.version.hip,'gpu':torch.cuda.get_device_name(),'loss_weights':loss_fn.loss_weight}
    environment['source_sha256']={str(f):file_sha256(f) for f in [Path(__file__),Path(__file__).parent/'src/engramfold/experiments/gradient_runtime_v3.py',Path(__file__).parent/'src/engramfold/experiments/learned_direction.py'] if f.exists()}
    environment['lock_sha256']=file_sha256(a.lock)
    write(a.output/'environment.json',environment)
    students=FrozenStudents(a.inventory,a.features,a.manifest,file_sha256(a.fixture/'mini.pt'))
    student_digests={k:frozen_digest(v[0]) for k,v in students.models.items()}
    records=[]
    for target in a.targets:
        if datetime.datetime.now(datetime.UTC)>=datetime.datetime.fromisoformat(lock['dispatch_deadline_utc']):break
        for seed in lock['noise_seeds']:
            if datetime.datetime.now(datetime.UTC) >= datetime.datetime.fromisoformat(lock['compute_deadline_utc']):
                break
            destination=a.output/f'{target}_{seed}.json'
            if destination.exists():
                records.append(json.loads(destination.read_text())); continue
            t=time.monotonic(); record={'target_id':target,'noise_seed':seed,'host':socket.gethostname(),'status':'running'}
            try:
                seed_everything(seed=seed,deterministic=True)
                data=to_device(torch.load(a.cache/f'{target}.pt',map_location='cpu',weights_only=False,mmap=True),'cuda')
                updates=students.prepare(target,data['query'])
                diag=DualLossDiagnostic(model,data,loss_fn,permutation)
                uq=data['query']['query_update']; u=uq.detach().clone().requires_grad_(True)
                loss, dyn, metrics, base=diag.evaluate(u,capture=True,components=a.components)
                gradients={'total':torch.autograd.grad(loss,u,retain_graph=a.components)[0].detach()}
                if a.components:
                    gradients['distogram']=torch.autograd.grad(diag.component_losses['distogram'],u,retain_graph=True)[0].detach()
                    gradients['denoising']=torch.autograd.grad(diag.component_losses['denoising'],u)[0].detach()
                    err=(gradients['total']-gradients['distogram']-gradients['denoising']).norm()/gradients['total'].norm().clamp_min(1e-30)
                    record['component_additivity_relative_error']=float(err)
                    assert err < 1e-4, 'component gradient additivity'
                diag.component_losses={}
                del loss,dyn,metrics
                g=gradients['total']; gn=float(g.double().norm()); un=float(uq.double().norm())
                if not gn > 0: raise RuntimeError('zero gradient: alignment not applicable')
                record.update(baseline=base,gradient_norm=gn,query_norm=un,pre_injection_hashes=diag.cache['pre_injection_hashes'])
                with torch.no_grad(): _,_,_, replay=diag.evaluate(uq)
                assert replay==base, 'cached replay differs from full reference'
                record['cached_replay_exact']=True
                if a.smoke:
                    replay_u=uq.detach().clone().requires_grad_(True)
                    rl,rd,rm,_=diag.evaluate(replay_u)
                    rg=torch.autograd.grad(rl,replay_u)[0]
                    re=float((rg.double()-g.double()).norm()/g.double().norm())
                    record['cached_gradient_relative_error']=re
                    assert re <= 1e-5, 'cached gradient mismatch'
                    del rl,rd,rm,rg,replay_u
                    restore(diag.start_rng)
                    ref,_=native_task_loss(model,original,copy.deepcopy(data['features']),copy.deepcopy(data['labels']),uq,loss_fn,permutation)
                    model.msa_module=original
                    record['native_baseline_loss']=float(ref.detach()); del ref
                    assert record['native_baseline_loss']==base['dynamic_loss'], 'unmodified native mismatch'
                    d=-g/g.norm(); expected=float((g.double()*d.double()).sum())
                    rows=[]
                    for beta in lock['fd_steps']:
                        h=un*beta
                        with torch.no_grad():
                            _,_,_,plus=diag.evaluate(uq+h*d)
                            _,_,_,minus=diag.evaluate(uq-h*d)
                        row={'beta':beta,'h':h,'ad':expected,'branch_changed':plus['dynamic_permutation_changed'] or minus['dynamic_permutation_changed']}
                        for key in ['fixed','dynamic']:
                            delta=plus[key+'_loss']-minus[key+'_loss']; fd=delta/(2*h)
                            row[key+'_fd']=fd
                            row[key+'_error']=abs(fd-expected)/max(abs(expected),1e-30)
                            row['resolved_'+key]=abs(delta)>4*torch.finfo(torch.float32).eps*max(abs(base[key+'_loss']),1.)
                        rows.append(row)
                    record['finite_difference']=rows
                    record['fixed_fd_passed']=adjacent_ok(rows,'fixed_error')
                    record['dynamic_fd_passed']=adjacent_ok(rows,'dynamic_error')
                    record['passed']=record['fixed_fd_passed']
                bundle={'target_id':target,'noise_seed':seed,'query_a':cpu(data['query']['query_a']),
                        'query_b':cpu(data['query']['query_b']),'weight':cpu(w),'gradients':cpu(gradients),
                        'conditions':cpu(diag.conditions),'baseline':base,'query_norm':un}
                torch.save(bundle,a.output/f'{target}_{seed}_gradient.pt')
                if not a.smoke:
                    native=TangentMap(data['query']['query_a'].double(),data['query']['query_b'].double(),w)
                    scale=float(native.mean_gram_diagonal()); projections=[]; component_rows=[]
                    for rot in lock['rotation_seeds']:
                        rw=torch.einsum('dc,cpq->dpq',orthogonal_rotation(128,rot).to(w),w)
                        op=TangentMap(native.query_a,native.query_b,rw)
                        for component in ['denoising','distogram']:
                            cr=ridge_projection(op,gradients[component].double(),relative_ridge=lock['primary_relative_ridge'],gram_scale=scale)
                            cr.pop('projected');cr.pop('increments');cr.update(component=component,rotation=rot);component_rows.append(cr)
                        solved=ridge_projection(op,g.double(),relative_ridge=lock['primary_relative_ridge'],gram_scale=scale)
                        v=solved.pop('projected'); solved.pop('increments')
                        solved.update(rotation=rot,relative_ridge=lock['primary_relative_ridge'],interventions=[])
                        if solved['converged'] and solved['alignment_applicable'] and v.norm()>0:
                            aa,bb=solved['shrinkage_alignment'],solved['projected_energy_fraction']
                            solved['ridge_bounds_valid']=(-1e-8 <= bb <= aa+1e-8 and aa <= 1+1e-8)
                            if not solved['ridge_bounds_valid']: raise RuntimeError('ridge bounds failed')
                            d=(-v/v.norm()).float(); d=d/d.norm()
                            dot=float((g.double()*d.double()).sum())
                            for beta in ([1e-4,1e-3,1e-2] if target in lock['audit_targets'] else lock['relative_norms']):
                                eta=un*beta
                                with torch.no_grad():
                                    _,_,_,plus=diag.evaluate(uq+eta*d)
                                    _,_,_,minus=diag.evaluate(uq-eta*d)
                                item={'relative_norm':beta,'eta':eta,'actual_norm':float((eta*d).double().norm()),'predicted_normalized_decrease':-dot/gn,
                                      'plus':plus,'minus':minus}
                                for key in ['fixed','dynamic']:
                                    item[key+'_normalized_decrease']=(base[key+'_loss']-plus[key+'_loss'])/(eta*gn)
                                    fd=(plus[key+'_loss']-minus[key+'_loss'])/(2*eta)
                                    item[key+'_fd_relative_error']=abs(fd-dot)/max(abs(dot),1e-30)
                                solved['interventions'].append(item)
                        projections.append(solved)
                    record['projections']=projections
                    record['components']=component_rows
                    record['passed']=all(x['converged'] and len(x['interventions'])==(3 if target in lock['audit_targets'] else len(lock['relative_norms'])) for x in projections) and all(x['converged'] for x in component_rows)
                oracle={x['rotation']:next(e for e in x['interventions'] if e['relative_norm']==.001)['predicted_normalized_decrease'] for x in record.get('projections',[]) if x['interventions']}
                record['students']=students.evaluate(diag,uq,g,base,updates,oracle)
                record['passed']=record['passed'] and len(record['students'])==12 and all(x['applicable'] for x in record['students'])
                record['status']='complete' if record['passed'] else 'numerical_failure'
                del data,diag,g,gradients,u
            except Exception as exc:
                model.msa_module=original
                record.update(status='error',passed=False,error=str(exc),traceback=traceback.format_exc())
            record['seconds']=time.monotonic()-t
            write(destination,record); records.append(record)
            print(json.dumps({k:record.get(k) for k in ['target_id','noise_seed','status','seconds','error','fixed_fd_passed','dynamic_fd_passed']}),flush=True)
            if record['status']=='error': break
    unchanged=frozen_digest(model)==digest
    students_unchanged=all(frozen_digest(v[0])==student_digests[k] for k,v in students.models.items())
    unchanged=unchanged and students_unchanged
    expected=len(a.targets)*len(lock['noise_seeds'])
    report={'schema':'engramfold.gradient_result.v4','smoke':a.smoke,'expected_cases':expected,'completed_cases':len(records),
            'passed':len(records)==expected and all(x.get('passed',False) for x in records) and unchanged,
            'frozen_parameters_unchanged':unchanged,'students_unchanged':students_unchanged,'cases':records,'environment':environment}
    write(a.output/'report.json',report)
    print('COMPLETE',report['passed'],flush=True)

if __name__=='__main__': main()
