import datetime, hashlib, json, os
from pathlib import Path
ROOT=Path('/media/PM982/engramfold/runs/protenix_fourcell_budget_20260926')
OLD=Path('/media/PM982/engramfold/runs/protenix_fresh192_20260925')
A=Path('/media/PM982/engramfold/runs/diamondhill_plm_A_20260923')
CACHE=Path('/media/PM982/engramfold/runs/train384_direction_20260920/cache')
RUNTIME=Path('/media/PM982/onestepfold/protenix_stage0_pkg/v1_1/runtime')
PYTHON='/home/pc/anaconda3/envs/fold/bin/python'
SEEDS=(20260923,20260924,20260925)
ROTS=(20261001,20261002,20261003)
def names():
 return [f'C_{k}_{"native" if r is None else "r"+str(r)}_s{s}' for k in ('factor','generic_plus') for s in SEEDS for r in (None,*ROTS)]
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(f'.{os.getpid()}.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');q.replace(p)
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def locked(root=ROOT):
 c=read(root/'execution_lock.json')
 for f,h in c['files'].items():
  if sha(f)!=h:raise ValueError('execution file changed: '+f)
 return c,sha(root/'execution_lock.json')
def manifest_validate(m):
 import hashlib
 from engramfold.experiments.protenix_fresh_fourcell.common import validate_manifest
 if m.get('role')!='budget_development':return validate_manifest(m)
 assert m['schema']=='engramfold.protenix_fresh192.sequence.v2' and not m.get('engineering')
 assert len(m['targets'])==8 and len({x['target_id'] for x in m['targets']})==8
 for x in m['targets']:
  assert set(x)=={'target_id','sequence','sequence_sha256','sequence_length','length_bin'}
  assert len(x['sequence'])==x['sequence_length'] and hashlib.sha256(x['sequence'].encode()).hexdigest()==x['sequence_sha256']
