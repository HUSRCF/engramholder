import argparse,subprocess,sys,os
from pathlib import Path
from common import *
from support import check
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--index',type=int,required=True);a=p.parse_args();r=a.root
lock,c,lk=check(r,True)
gate=read(r/'repetition_engineering/complete.json');assert gate['passed'] and gate['engineering_updates']==384 and gate['repetition_lock_sha256']==sha(r/'repetition_lock.json')
run=read(r/'e2_execution_lock.json')['runs'][a.index]
folder=r/'e2/formal'/run['name'];folder.mkdir(parents=True,exist_ok=True)
provenance=folder/'repetition.json'
if provenance.exists():assert read(provenance)['repetition_lock_sha256']==sha(r/'repetition_lock.json')
else:
    assert not(folder/'latest.pt').exists(),'Do not import a previous trained checkpoint'
    write(provenance,dict(repetition_lock_sha256=sha(r/'repetition_lock.json'),parent_root=lock['parent_root'],initialization='original_seed_zero_output_identity_C_no_warmstart',slurm_job=os.environ.get('SLURM_JOB_ID')))
# Byte-identical accepted training runner; independent root, no parent outputs copied.
subprocess.run([sys.executable,'-u',str(r/'operations/e2_diagnostic_repair_20260924/run.py'),'--root',str(r),'--mode','e2','--index',str(a.index)],check=True)
