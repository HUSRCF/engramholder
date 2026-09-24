import hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
import torch

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(f'.{os.getpid()}.tmp');torch.save(x,tmp);tmp.replace(p)
def contract(r,gpu=False,verify_assets=True):
 c=read(r/'execution_lock.json');lk=sha(r/'execution_lock.json')
 assert sha(r/'protocol.md')==c['protocol_sha256']
 for name,h in c['staging_hashes'].items():assert sha(r/'staging'/name)==h,name
 for name,h in c['source_hashes'].items():assert sha(r/'source'/name)==h,name
 if verify_assets:
  for name,h in c['data_hashes'].items():assert sha(r/'data'/name)==h,name
  assert sha(r/'weights/params_model_3_ptm.npz')==c['weight_sha256']
 assert sha(r/'rotations.npz')==c['rotations_sha256']
 if gpu:
  assert torch.__version__==c['torch_version']and'H100'in torch.cuda.get_device_name()
  torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 return c,lk
def rotation(r,rid):return torch.eye(128)if rid=='I'else torch.from_numpy(np.load(r/'rotations.npz')[rid+'_fp32'].copy())
def state_hash(model):
 h=hashlib.sha256()
 for name,v in model.state_dict().items():h.update(name.encode());h.update(v.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
 return h.hexdigest()
def load_model(r,c):
 from openfold.model.model import AlphaFold
 from openfold.utils.import_weights import assign,generate_translation_dict,process_translation_dict
 from engramfold.experiments.openfold_adapter_runtime import runtime_config,enable_nonreentrant_checkpointing
 config=runtime_config(activation_checkpointing=True);enable_nonreentrant_checkpointing();m=AlphaFold(config).eval().requires_grad_(False)
 arr=np.load(r/'weights/params_model_3_ptm.npz');t=process_translation_dict(generate_translation_dict(m,'model_3_ptm',is_multimer=False));assert set(t)==set(arr.files);assign(t,arr);del t,arr
 m.cuda();assert state_hash(m)==c['frozen_backbone_sha256'];return m,config
class Features:
 def __init__(self,r,config):self.r=r;self.config=config;self.cache={}
 def get(self,t,labels=False):
  from engramfold.experiments.openfold_adapter_runtime import sequence_features,structure_labels
  tid=t['target_id']
  if tid not in self.cache:
   torch.manual_seed(20260921);f=sequence_features(t['sequence'],self.config)
   e=torch.load(self.r/'data'/f'{tid}.pt',map_location='cpu',weights_only=False)
   assert e['sequence_sha256']==hashlib.sha256(t['sequence'].encode()).hexdigest()and e['features'].shape==(len(t['sequence']),480)
   self.cache[tid]=(f,e['features'].float())
  f,e=self.cache[tid];lab=structure_labels(self.r/'data'/f'{tid}.cif.gz',t,f['aatype'][...,0])if labels else None
  return {k:v.cuda()for k,v in f.items()},e.cuda(),{k:v.cuda()for k,v in lab.items()}if lab else None
def writer(r,rid,seed,channels=False):
 from engramfold.models.prospective_orientation import ProspectiveOPMAdapter
 torch.manual_seed(seed);return ProspectiveOPMAdapter(rotation(r,rid),channels).cuda().eval()
def teacher(r,c,rid,seed):
 item=next(x for x in c['teachers']if x['seed']==seed);assert sha(item['checkpoint'])==item['checkpoint_sha256']
 w=writer(r,'I',seed);ck=torch.load(item['checkpoint'],map_location='cpu',weights_only=False);w.load_state_dict(ck['writer'])
 if rid!='I':
  # D rotates the already-formed native residual; no factor rescaling.
  rot=rotation(r,rid).cuda();native_residual=w.residual
  w.residual=lambda *args,**kwargs:native_residual(*args,**kwargs)@rot.T
 return w.requires_grad_(False)
def optimizer(w,channels):
 if not channels:return torch.optim.AdamW(w.parameters(),lr=1e-4,weight_decay=.01)
 theta=[p for n,p in w.named_parameters()if not n.startswith('channels.')]
 return torch.optim.AdamW([dict(params=theta,lr=1e-4,weight_decay=.01),dict(params=w.channels.parameters(),lr=1e-3,weight_decay=0.)],betas=(.9,.999),eps=1e-8)
