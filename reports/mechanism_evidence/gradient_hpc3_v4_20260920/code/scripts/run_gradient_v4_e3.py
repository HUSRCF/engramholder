"""Time-gated learned-context residual removal; no training or gradient fitting."""
import argparse,copy,datetime,json,os,time,traceback
from pathlib import Path
os.environ.setdefault('LAYERNORM_TYPE','torch')
import torch
import torch.utils.checkpoint
import yaml
from ml_collections import ConfigDict
from protenix.model.protenix import Protenix
from protenix.model.loss import ProtenixLoss
from protenix.utils.permutation.permutation import SymmetricPermutation
from protenix.utils.seed import seed_everything
from protenix.utils.torch_utils import to_device
from engramfold.experiments.gradient_runtime_v3 import DualLossDiagnostic
from engramfold.experiments.gradient_runtime import restore
from engramfold.experiments.interface_runtime import native_task_loss
from engramfold.experiments.train_structure_control import frozen_digest
from engramfold.experiments.sequence_feature_cache import file_sha256
from engramfold.experiments.learned_direction import FrozenStudents
from run_gradient_v4 import write

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--shard',type=int,required=True);a=p.parse_args();r=a.root
lock=json.loads((r/'source/execution_lock.json').read_text());assert json.loads((r/'outputs/smoke/report.json').read_text())['passed']
assert json.loads((r/'source/e3_decision.json').read_text())['launch']
out=r/f'outputs/e3/w{a.shard}';out.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
raw=json.loads((r/'fixture/config.json').read_text());cfg=ConfigDict({k:yaml.safe_load(v) if isinstance(v,str) and '\n' in v else v for k,v in raw.items()})
cfg.loss.weight.alpha_bond=1.;cfg.loss.weight.alpha_confidence=0.;cfg.atom_permutation.train.diffusion_sample=True;cfg.mc_dropout_apply_rate=0.;cfg.mc_dropout_rate=0.
model=Protenix(cfg);state=torch.load(r/'fixture/mini.pt',map_location='cpu',weights_only=False)['model'];model.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True);del state
model=model.cuda().requires_grad_(False).train();original=model.msa_module;digest=frozen_digest(model)
loss_fn=ProtenixLoss(cfg);permutation=SymmetricPermutation(cfg,error_dir=str(out/'permutation'))
students=FrozenStudents(r/'source/checkpoint_inventory.json',r/'features',r/'source/manifest.json',file_sha256(r/'fixture/mini.pt'),experiment='E3')
cache=r.parent/'gradient_multinode_v3/observed_cache';index=json.loads((cache/'index.json').read_text());targets=lock['e3_targets'][a.shard::8];records=[]
with torch.no_grad():
 for target in targets:
  if datetime.datetime.now(datetime.UTC)>=datetime.datetime.fromisoformat(lock['dispatch_deadline_utc']):break
  assert file_sha256(cache/f'{target}.pt')==index['records'][target]['sha256']
  data=to_device(torch.load(cache/f'{target}.pt',map_location='cpu',weights_only=False),'cuda');uq=data['query']['query_update'];updates=students.prepare(target,data['query'])
  for seed in lock['noise_seeds'][:3]:
   for name,item in updates.items():
    dest=out/f'{target}_{seed}_{name}.json'
    if dest.exists():records.append(json.loads(dest.read_text()));continue
    row={'target_id':target,'noise_seed':seed,'model':name,'rotation':item['rotation']};start=time.monotonic()
    try:
     seed_everything(seed=seed,deterministic=True);diag=DualLossDiagnostic(model,data,loss_fn,permutation);learned=uq+item['delta'];diag.query=learned
     _,_,_,baseline=diag.evaluate(learned,capture=True);_,_,_,replay=diag.evaluate(learned);assert baseline==replay
     restore(diag.start_rng);ref,_=native_task_loss(model,original,copy.deepcopy(data['features']),copy.deepcopy(data['labels']),learned,loss_fn,permutation);model.msa_module=original
     assert float(ref)==baseline['dynamic_loss'],'actual all-recycle route mismatch'
     _,_,_,removed=diag.evaluate(uq)
     row.update(passed=True,status='complete',baseline=baseline,removed=removed,native_replay_exact=True,norm_ratio=float(item['delta'].double().norm()/uq.double().norm()),retaining_effect=removed['dynamic_loss']-baseline['dynamic_loss'])
    except Exception as exc:
     model.msa_module=original;row.update(passed=False,status='error',error=str(exc),traceback=traceback.format_exc())
    row['seconds']=time.monotonic()-start;write(dest,row);records.append(row);print(json.dumps({k:row.get(k) for k in ['target_id','model','noise_seed','status','error','seconds']}),flush=True)
    if not row['passed']:break
unchanged=frozen_digest(model)==digest
write(out/'report.json',{'expected':len(targets)*3*12,'completed':len(records),'passed':len(records)==len(targets)*36 and all(x['passed'] for x in records) and unchanged,'frozen_parameters_unchanged':unchanged})
