import datetime,hashlib,importlib.util,json,random,sys
from pathlib import Path
import numpy as np
import torch

def read(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');q.replace(p)
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');torch.save(x,q);q.replace(p)
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def rng():return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),numpy=np.random.get_state(),python=random.getstate())
def restore_rng(s):
 torch.set_rng_state(s['torch'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in s['cuda']]);np.random.set_state(s['numpy']);random.setstate(s['python'])
def state_hash(model):
 h=hashlib.sha256()
 for n,v in sorted(model.state_dict().items()):h.update(n.encode());h.update(v.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
 return h.hexdigest()
def exact(a,b):
 if torch.is_tensor(a):return torch.is_tensor(b) and a.dtype==b.dtype and torch.equal(a.cpu(),b.cpu())
 if isinstance(a,np.ndarray):return np.array_equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
 return a==b

def module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def contract(root):
 c=read(root/'execution_lock.json')
 for rel,digest in c['files'].items():assert sha(root/rel)==digest,rel
 for path,digest in c.get('external_files',{}).items():assert sha(path)==digest,path
 return c,sha(root/'execution_lock.json')
