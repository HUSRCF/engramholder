import hashlib
import json
from pathlib import Path

root = Path('/media/PM982/engramfold/runs/atlas_propagation_localization_20260926')
parent = root.parent/'atlas_posttraining_calibration_20260925'
def read(p): return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()
old=read(parent/'execution_lock.json')
receipt=read(parent/'atlas/COMPLETE.json')
assert receipt['n_predictions']==960 and receipt['execution_lock_sha256']==sha(parent/'execution_lock.json')
assert receipt['analysis_sha256']==sha(parent/'atlas/formal_analysis/analysis.json')
assert receipt['records_sha256']==sha(parent/'atlas/formal_analysis/metric_records.json')
analysis=read(parent/'atlas/formal_analysis/analysis.json')
assert not any(analysis['failure_counts'].values())
files={}
for p,h in old['external_files'].items():
    # Verify the actual historical third-party source/runtime dependencies.
    assert sha(p)==h,p
    files[p]=h
for p,h in old['large_assets'].items():
    assert Path(p).stat().st_size == h['size'],p
    assert sha(p)==h['sha256'],p
    files[p]=h['sha256']
for p in (root/'source').rglob('*.py'):
    files[str(p)]=sha(p)
runs=[dict(name='query',branch='query')]
for asset in read(parent/'asset_audit.json')['atlas']:
    assert sha(asset['path'])==asset['sha256']
    files[asset['path']]=asset['sha256']
    seed=asset['seed']
    runs.append(dict(name=f'parent_s{seed}',branch='parent',seed=seed,checkpoint=asset['path']))
    for branch in ('C_only','head_only'):
        directory=parent/'atlas/formal'/f'atlas_I_s{seed}_{branch}'
        done=read(directory/'training_complete.json')
        ck=directory/'checkpoint_512.pt'
        assert done['complete'] and sha(ck)==done['checkpoint_sha256']
        files[str(ck)]=sha(ck)
        runs.append(dict(name=f'{branch}_s{seed}',branch=branch,seed=seed,checkpoint=str(ck)))
base=Path(old['parent_roots']['atlas'])
targets=read(root/'atlas_lock.json')['targets']
for t in targets:
    cache=base/'data'/f"{t['target_id']}.pt"
    files[str(cache)]=sha(cache)
files[str(root/'PROTOCOL.md')]=sha(root/'PROTOCOL.md')
cfg=dict(schema='engramfold.atlas_propagation_localization.v1',targets=targets,runs=runs,
         parent_root=str(parent),base=str(base),frozen_hash=old['frozen_hash'],files=files,
         full_forward_budget=26,new_training=0,hip_devices=[6,7],
         prior_receipt_sha256=sha(parent/'atlas/COMPLETE.json'),
         source_inspection='LM stack independent of recycled states; fusion afterward',
         no_structure_scoring=True)
dest=root/'lock.json'
assert not dest.exists(),'Do not overwrite an executed lock'
dest.write_text(json.dumps(cfg,indent=2)+'\n')
print(json.dumps(dict(locked=True,sha256=sha(dest),files=len(files),systems=len(runs),targets=len(targets))))
