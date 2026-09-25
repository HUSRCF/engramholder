"""Isolated prospective validation contracts and output identity checks."""
import contextlib,hashlib,json,os,sys
from pathlib import Path
SEEDS=(20260923,20260924,20260925)
ROTS=(20261001,20261002,20261003)

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.%s.tmp'%os.getpid());q.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');q.replace(p)
def models():
 return ['query']+[f'C_{kind}_{"native" if r is None else "r"+str(r)}_s{s}' for kind in ('factor','generic_plus') for s in SEEDS for r in (None,*ROTS)]
def validate_manifest(m):
 assert m['schema']=='engramfold.protenix_fresh192.sequence.v2'
 expected=2 if m.get('engineering') else 192
 assert len(m['targets'])==expected and len({t['target_id'] for t in m['targets']})==expected
 for t in m['targets']:
  assert set(t)=={'target_id','sequence','sequence_sha256','sequence_length','length_bin'}
  assert len(t['sequence'])==t['sequence_length'] and hashlib.sha256(t['sequence'].encode()).hexdigest()==t['sequence_sha256']
 if not m.get('engineering'):
  from collections import Counter
  assert Counter(t['length_bin'] for t in m['targets'])==dict.fromkeys(['128-191','192-255','256-319','320-384'],48)
@contextlib.contextmanager
def deny_evidence(manifest,teacher_root):
 # A sequence-only process never receives reference mappings; reject all raw
 # archive and known teacher/cache paths, plus alignment files.
 active=True;prefix=str(Path(teacher_root).resolve())+os.sep
 def audit(event,args):
  if not active or event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
  p=str(Path(os.fsdecode(args[0])).resolve())
  if p.startswith(prefix) or '/Dataset/raw/' in p or '/reference/' in p or p.endswith(('.cif.gz','.a3m','.sto','reference_manifest.json')):
   raise RuntimeError('forbidden target evidence read: '+p)
 sys.addaudithook(audit)
 try:yield
 finally:active=False

def frozen_hash(model):
 h=hashlib.sha256()
 for k,v in sorted(model.state_dict().items()):
  h.update(k.encode());h.update(v.detach().cpu().contiguous().view(__import__('torch').uint8).numpy().tobytes())
 return h.hexdigest()

def writer_engineering(writer,task,features,query,decoder):
 import torch,copy
 from engramfold.models.plm_extension import make_writer
 with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
  zero=make_writer('protenix',task['geometry']['kind'],task['geometry']['rotation_seed'],task['seed'],1152,features.device)
  z=zero(features,decoder=decoder,**query)
  assert torch.equal(z,query['query_update']), 'zero-output identity failed'
  unrotated=copy.deepcopy(writer);unrotated.rotation.copy_(torch.eye(128,device=features.device));unrotated.rotation_seed=None
  # Avoid baseline cancellation: evaluate a sparse set of pair residuals.
  if task['geometry']['kind']=='factor':
   da,db=writer.increments(features)
   from engramfold.models.native_geometry import bilinear
   l=len(features);ii=torch.arange(min(l,64),device=features.device);pairs=torch.stack((ii,(ii*17+3)%l),dim=-1)
   w=unrotated.effective_weight(decoder)
   value=bilinear(da,query['query_b']+db,w,pairs)+bilinear(query['query_a'],db,w,pairs)
   wr=writer.effective_weight(decoder)
   actual=bilinear(da,query['query_b']+db,wr,pairs)+bilinear(query['query_a'],db,wr,pairs)
   expected=torch.nn.functional.linear(value,writer.rotation)
  else:
   actual=writer(features,decoder=decoder,**query)-query['query_update']
   value=unrotated(features,decoder=decoder,**query)-query['query_update']
   expected=torch.nn.functional.linear(value,writer.rotation)
  err=float((actual-expected).double().norm()/expected.double().norm().clamp_min(1e-12))
  assert err<=1e-3,('rotation must act on complete residual',err)
 return dict(zero_identity=True,rotation_relative_l2=err)
