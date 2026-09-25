"""Read-only CPU acceptance audit against raw E3 and parent artifacts."""
import argparse,ast,datetime,hashlib,json,math
from pathlib import Path
import numpy as np
import torch

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
read=lambda p:json.loads(Path(p).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()
def dump(path,x):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(path)

c=read(r/'execution_lock.json');lk=sha(r/'execution_lock.json');parent=Path(c['parent_root'])
assert lk=='25f162d613455d380113d29a58256b9b5be72d46f669675073ce035c739c7200'
assert sha(r/'protocol.md')==c['protocol_sha256']
for name,h in c['files'].items():assert sha(r/'staging'/name)==h,name
assert sha(parent/'execution_lock.json')==c['parent_execution_lock_sha256']
old=read(parent/'execution_lock.json')
for key,folder in [('source_hashes','source'),('staging_hashes','staging'),('data_hashes','data')]:
    for name,h in old[key].items():assert sha(parent/folder/name)==h,(folder,name)
assert sha(parent/'weights/params_model_3_ptm.npz')==old['weight_sha256']
assert sha(parent/'rotations.npz')==old['rotations_sha256']
for name,h in c['parent_receipts'].items():assert sha(parent/name)==h,name
sel=read(parent/'prediction_lock.json')
assert c['selected_rotations']==[sel['selected_low'],sel['selected_high']]
done=read(r/'analysis/complete.json')
for name,key in [('analysis.json','analysis_sha256'),('metric_records.json','records_sha256'),('combined_metric_records.json','combined_sha256')]:
    assert sha(r/'analysis'/name)==done[key]
assert done['new_predictions']==864 and done['new_failures']==0
def fn_dump(path,name):
    return ast.dump(next(x for x in ast.parse(path.read_text()).body if isinstance(x,ast.FunctionDef)and x.name==name),include_attributes=False)
assert fn_dump(r/'staging/scoring.py','score_jobs')==fn_dump(parent/'operations/scoring_repair_20260924/analyze.py','score_jobs')

train=read(parent/'data/train96.json')['targets']
targets=[t for t in read(parent/'data/evaluation144.json')['targets']if t['panel']=='confirm96']
assert len(train)==len(targets)==96
assert not({t['target_id']for t in train}&{t['target_id']for t in targets})
ref_done=read(r/'references/complete.json');refs={}
for rec in ref_done['records']:
    key=(rec['phase'],rec['target_id']);assert key not in refs;refs[key]=rec
    t=next(t for t in (train if key[0]=='train' else targets)if t['target_id']==key[1])
    assert rec['execution_lock_sha256']==lk and rec['two_seed_exact'] and rec['rng_preserved']
    assert rec['gradient_path']==(key[0]=='train')
    assert rec['sequence_sha256']==hashlib.sha256(t['sequence'].encode()).hexdigest()
    path=r/'references'/key[0]/(key[1]+'.pt');assert sha(path)==rec['sha256']
    obj=torch.load(path,map_location='cpu',weights_only=False)
    assert obj['execution_lock_sha256']==lk and (obj['phase'],obj['target_id'])==key
    assert len(obj['anchors'])==4
    for q in obj['anchors']:
        assert q['a'].shape==q['b'].shape==(len(t['sequence']),32)
        assert q['mask'].shape==(len(t['sequence']),)
        assert all(torch.isfinite(q[k]).all() and not q[k].requires_grad for k in ['a','b','mask'])
assert set(refs)=={(phase,t['target_id'])for phase,ts in [('train',train),('eval',targets)]for t in ts}
assert len(refs)==192 and ref_done['reference_forward_trajectories']==384

from engramfold.models.prospective_orientation import ProspectiveOPMAdapter
from engramfold.evaluation.structure import read_atom_site_positions
from engramfold.evaluation.post_validation import parse_tm_score

raw_new=read(r/'analysis/metric_records.json');combined=read(r/'analysis/combined_metric_records.json')
scores={(x['system'],x['target_id']):x for x in combined};assert len(scores)==len(combined)==1824
assert len(raw_new)==864 and all(scores[x['system'],x['target_id']]==x for x in raw_new)
oldrows=read(parent/'e1/analysis/metric_records.json');oldvalues={(x['system'],x['target_id']):x for x in oldrows}
oldnames={'query'}|{f'native_s{s}'for s in old['formal_seeds']}|{f'{rid}_s{s}'for rid in c['selected_rotations']for s in old['formal_seeds']}
assert all(scores[k]==v for k,v in oldvalues.items()if k[0]in oldnames)
assert sum(k[0]in oldnames for k in scores)==960

pred_sources=[];run_audits=[];latest_prediction_mtime=0
for run in c['runs']:
    folder=r/'formal'/run['name'];cd=read(folder/'complete.json');tr=read(folder/'training_complete.json')
    assert cd['complete'] and cd['execution_lock_sha256']==lk and cd['run']==run
    assert tr['complete'] and tr['steps']==1536 and tr['execution_lock_sha256']==lk
    assert sha(folder/'checkpoint_1536.pt')==tr['checkpoint_sha256']
    log=[json.loads(line)for line in(folder/'training.jsonl').read_text().splitlines()]
    assert [x['step']for x in log]==list(range(1,1537))
    order=np.random.default_rng(run['seed']);expected=np.concatenate([order.permutation(96)for _ in range(16)])
    sequence=[train[j]['target_id']for j in expected]
    assert read(folder/'schedule.json')==sequence==[x['target_id']for x in log]
    assert all(math.isfinite(x['loss']) and math.isfinite(x['gradient_norm']) for x in log)
    assert all(x['forward_hook_calls']==4 and x['backward_hook_replays']==1 for x in log)
    for x in log:
        assert x['reference_sha256']==refs['train',x['target_id']]['sha256']
        assert [j['anchor_index']for j in x['injections']]==[0,1,2,3]
        assert [j['grad_enabled']for j in x['injections']]==[False,False,False,True]
        assert all(not j['backward_replay'] and math.isfinite(j['residual_norm'])and math.isfinite(j['baseline_norm'])for j in x['injections'])
    assert all(log[k-1]['reference_replay_exact'] for k in [384,768,1536])
    ck0=torch.load(folder/'checkpoint_0.pt',map_location='cpu',weights_only=False)
    ckf=torch.load(folder/'checkpoint_1536.pt',map_location='cpu',weights_only=False)
    rot=torch.eye(128)if run['rotation_id']=='I'else torch.from_numpy(np.load(parent/'rotations.npz')[run['rotation_id']+'_fp32'].copy())
    torch.manual_seed(run['seed']);w=ProspectiveOPMAdapter(rot,False)
    assert w.channels is None and not any(k.startswith('channels.')for k in ckf['writer'])
    assert w.state_dict().keys()==ck0['writer'].keys()
    assert all(torch.equal(v,ck0['writer'][k])for k,v in w.state_dict().items())
    assert ck0['step']==ck0['data_position']==0 and not ck0['optimizer']['state']
    assert ckf['step']==ckf['data_position']==1536 and ckf['lr']==[1e-4]
    assert ckf['frozen_backbone_sha256']==old['frozen_backbone_sha256']
    assert ckf['reference_receipt_sha256']==sha(r/'references/complete.json')
    assert all(torch.isfinite(v).all()for v in ckf['writer'].values())
    for group in ckf['optimizer']['param_groups']:
        assert group['lr']==1e-4 and group['weight_decay']==.01 and group['betas']==(.9,.999)and group['eps']==1e-8
    assert all(int(state['step'])==1536 for state in ckf['optimizer']['state'].values())
    assert cd['n_predictions']==96 and cd['failures']==0 and cd['no_evaluation_labels_parsed']
    assert [x['target_id']for x in cd['records']]==[t['target_id']for t in targets]
    for t,rec in zip(targets,cd['records'],strict=True):
        assert rec['status']=='ok' and rec['execution_lock_sha256']==lk
        assert rec['reference_sha256']==refs['eval',t['target_id']]['sha256']
        assert [j['anchor_index']for j in rec['injections']]==[0,1,2,3]
        assert all(not j['grad_enabled'] and not j['backward_replay']for j in rec['injections'])
        path=folder/'predictions'/(t['target_id']+'.npz')
        assert sha(path)==rec['prediction_sha256']==scores[run['name'],t['target_id']]['prediction_sha256']
        pred_sources.append((run['name'],t,path,r/'analysis/coordinates'/run['name']/t['target_id']/'tm.txt'))
        latest_prediction_mtime=max(latest_prediction_mtime,path.stat().st_mtime)
    run_audits.append(dict(run=run['name'],steps=1536,initialization_exact=True,without_C=True,optimizer_steps_exact=True,backbone_checked_by_runtime=True,reference_replays=3,predictions=96,zero_gradient_updates=sum(bool(x['zero_gradient_parameters'])for x in log),training_seconds=sum(x['seconds']for x in log),resume_files=[p.name for p in folder.glob('resume_*.jsonl')],residual_mean_by_round=np.mean([[j['residual_norm']for j in x['injections']]for x in cd['records']],axis=0).tolist()))

for name in sorted(oldnames):
    for t in targets:
        tid=t['target_id']
        if name in old['baselines']:
            item=next(x for x in old['baselines'][name]['predictions']if x['target_id']==tid);path=Path(item['path'])
            assert sha(path)==item['sha256']
        else:path=parent/'e1/formal'/name/'predictions'/(tid+'.npz')
        assert sha(path)==scores[name,tid]['prediction_sha256']
        pred_sources.append((name,t,path,parent/'e1/analysis/coordinates'/name/tid/'tm.txt'))

# Independently score each eligible unordered pair, without fixed_mask_metrics.
reference={};primary_errors=[];tm_errors=[];coverage=[]
for name,t,path,tmfile in pred_sources:
    tid=t['target_id'];data=np.load(path);xyz=data['coordinates'];mask=data['mask']
    assert str(data['sequence'])==t['sequence'] and xyz.shape==(len(t['sequence']),37,3)and np.isfinite(xyz).all()
    assert mask.shape==(len(t['sequence']),37)and mask[:,1].sum()==len(t['sequence'])
    if tid not in reference:
        _,atoms=read_atom_site_positions(parent/'data'/(tid+'.cif.gz'),label_asym_id=t['source_label_asym_id'])
        ca={i:v for(i,n),v in atoms.items()if n=='CA'};ids=sorted(ca)
        assert ids==t['reference_ca_indices']
        rx=np.stack([ca[i]for i in ids]);ii,jj=np.triu_indices(len(ids),k=1)
        rd=np.linalg.norm(rx[ii]-rx[jj],axis=1);sel=(rd>0)&(rd<15)
        reference[tid]=(ids,ii[sel],jj[sel],rd[sel])
    ids,ii,jj,rd=reference[tid];px=xyz[np.array(ids)-1,1]
    err=np.abs(np.linalg.norm(px[ii]-px[jj],axis=1)-rd)
    score=sum(np.count_nonzero(err<cut)for cut in [.5,1.,2.,4.])/(4*len(err))
    primary_errors.append(abs(score-scores[name,tid]['ca_lddt']))
    tm_errors.append(abs(parse_tm_score(tmfile.read_text())-scores[name,tid]['tm_score_fixed_full_length']))
    coverage.append(len(t['sequence']))
assert len(pred_sources)==1824 and max(primary_errors)<1e-12 and max(tm_errors)==0
assert r.joinpath('analysis/metric_records.json').stat().st_mtime>=latest_prediction_mtime
audit=dict(passed=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),execution_lock_sha256=lk,read_only=True,new_GPU_runs=0,reference_caches=192,reference_certified_forward_count=384,training_runs=9,training_updates=13824,predictions_new=864,predictions_including_fixed_baselines=1824,all_checkpoint_and_prediction_hashes_verified=True,all_initializations_reconstructed_exactly=True,all_runs_without_C=True,optimizer_and_schedule_verified=True,reference_replays_verified=27,scoring_after_all_predictions=True,raw_primary_scores_checked=1824,raw_primary_maxabs=max(primary_errors),tm_output_records_parsed=1824,tm_record_maxabs=max(tm_errors),score_function_AST_equal_to_accepted_parent=True,full_lengths=[min(coverage),max(coverage)],runs=run_audits)
dump(r/'audit_20260925/runtime_audit.json',audit)
print(json.dumps({k:v for k,v in audit.items()if k!='runs'},indent=2))
