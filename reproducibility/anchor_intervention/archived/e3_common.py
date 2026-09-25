import sys,json,time,hashlib
from pathlib import Path
import torch
import parent_core as pc
from parent_core import read,write,save,sha,state_hash
from query_anchor import query_reference,tensor_hash

def contract(root,gpu=False):
 c=read(root/'execution_lock.json');lk=sha(root/'execution_lock.json');parent=Path(c['parent_root'])
 assert sha(root/'protocol.md')==c['protocol_sha256']
 for name,h in c['files'].items():assert sha(root/'staging'/name)==h,name
 assert sha(parent/'execution_lock.json')==c['parent_execution_lock_sha256']
 old,_=pc.contract(parent,gpu=gpu)
 for name,h in c['parent_receipts'].items():assert sha(parent/name)==h,name
 assert c['selected_rotations']==[read(parent/'prediction_lock.json')['selected_low'],read(parent/'prediction_lock.json')['selected_high']]
 return c,old,parent,lk

def target_list(parent,phase):
 if phase=='train':return read(parent/'data/train96.json')['targets']
 return [x for x in read(parent/'data/evaluation144.json')['targets'] if x['panel']=='confirm96']

def certify_reference(model,f,phase):
 cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone();outputs=[];tick=time.monotonic()
 for seed in [20260923,20260925]:
  aa,x=query_reference(model,f,gradient_path=phase=='train',seed=seed);outputs.append((aa,x))
  assert torch.equal(cpu,torch.get_rng_state()) and torch.equal(gpu,torch.cuda.get_rng_state())
 a,b=outputs
 for q,t in zip(a[0],b[0],strict=True):
  for k in ['a','b','mask']:assert torch.equal(q[k],t[k]),(phase,k)
  assert q['eps']==t['eps']
 assert torch.equal(a[1],b[1]),'seed-dependent query coordinates'
 torch.cuda.synchronize()
 anchors=[{k:v.cpu() if torch.is_tensor(v) else v for k,v in q.items()}for q in a[0]]
 return anchors,dict(feature_sha256=tensor_hash(f),coordinate_sha256=tensor_hash({'coordinates':a[1]}),two_seed_exact=True,rng_preserved=True,seconds=time.monotonic()-tick,gradient_path=phase=='train')

def load_reference(root,phase,t,f,lk,device='cuda'):
 path=root/'references'/phase/(t['target_id']+'.pt');rec=read(path.with_suffix('.json'))
 assert rec['execution_lock_sha256']==lk and rec['phase']==phase and rec['target_id']==t['target_id']
 assert rec['sequence_sha256']==hashlib.sha256(t['sequence'].encode()).hexdigest()
 assert sha(path)==rec['sha256'] and rec['feature_sha256']==tensor_hash(f)
 obj=torch.load(path,map_location='cpu',weights_only=False);assert obj['execution_lock_sha256']==lk
 return [{k:v.to(device) if torch.is_tensor(v) else v for k,v in q.items()}for q in obj['anchors']],rec
