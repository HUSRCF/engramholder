import json,hashlib
from pathlib import Path
root=Path('/media/PM982/engramfold/runs/atlas_plm_complementarity_20260926')
old=root.parent/'atlas_posttraining_calibration_20260925';a66=root.parent/'diamondhill_plm_A_20260923'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
prior=read(old/'execution_lock.json');a=read(a66/'execution_lock.json');base=Path(a['atlas_old_root'])
train=read(base/'data/train96.json')['targets'];dev=read(base/'data/dev8.json')['targets']
assert len(train)==96 and len(dev)==8
assert not {t['sequence_sha256']for t in train}&{t['sequence_sha256']for t in dev}
targets=train+dev
for t in targets:assert hashlib.sha256(t['sequence'].encode()).hexdigest()==t['sequence_sha256']
code={str(p):sha(p)for p in (root/'source').rglob('*.py')}
for p in (root/'staging').glob('*.py'):code[str(p)]=sha(p)
for p,h in prior['external_files'].items():assert sha(p)==h;code[p]=h
code[str(root/'PROTOCOL.md')]=sha(root/'PROTOCOL.md')
ckpts=[]
for seed in [20260923,20260924,20260925]:
 p=a66/'atlas/formal'/f'C_factor_native_s{seed}'/'checkpoint_1536.pt';assert p.exists()
 ckpts.append(dict(seed=seed,path=str(p),sha256=sha(p),training_backend='H100'if seed==20260925 else 'MI250'))
config=dict(schema='engramfold.atlas_plm_complementarity.v1',train=train,dev=dev,base=str(base),calibration_root=str(old),
 feature_root=a['feature_root'],feature_index_sha256=sha(Path(a['feature_root'])/'index.json'),
 frozen_hash=prior['frozen_hash'],engineering_targets=[min(train,key=lambda t:len(t['sequence']))['target_id'],max(train,key=lambda t:len(t['sequence']))['target_id']],
 steps=1536,lr=.001,weight_decay=.0001,code_files=code,reference_hashes={str(base/'data'/f"{t['target_id']}.cif.gz"):sha(base/'data'/f"{t['target_id']}.cif.gz") for t in targets},
 esmc_checkpoints=ckpts,propagation_root=str(root.parent/'atlas_propagation_localization_20260926'),parameter_count=131520,
 seeds=[20260923,20260924,20260925],arms=['native_view','esmc','esmc_permuted'],checkpoint_rule='fixed step1536',new_folding_training=0)
assert not (root/'lock.json').exists()
(root/'lock.json').write_text(json.dumps(config,indent=2)+'\n');print(sha(root/'lock.json'))
