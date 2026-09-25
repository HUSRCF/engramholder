import argparse,datetime
from pathlib import Path
import torch
from common import *
from support import check,flat_error
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
lock,c,lk=check(r);rep_sha=sha(r/'repetition_lock.json');out=r/'repetition_engineering/complete.json'
if out.exists():
    assert read(out)['repetition_lock_sha256']==rep_sha and read(out)['passed']
    raise SystemExit('ALREADY_COMPLETE')
records=[];count=0
for i in range(6):
    folder=r/'repetition_engineering'/str(i);rows={}
    for phase,n in [('continuous',32),('prefix',16),('resume',16)]:
        x=read(folder/(phase+'.json'));assert x['passed'] and x['repetition_lock_sha256']==rep_sha and x['steps']==n
        assert sha(folder/(phase+'.pt'))==x['checkpoint_sha256'];rows[phase]=x;count+=n
    assert [x['target_id']for x in rows['continuous']['rows']]==[x['target_id']for x in rows['prefix']['rows']+rows['resume']['rows']]
    c1=torch.load(folder/'continuous.pt',map_location='cpu',weights_only=False);c2=torch.load(folder/'resume.pt',map_location='cpu',weights_only=False)
    records.append(dict(index=i,direction=rows['resume']['direction'],channels=rows['resume']['channels'],resume_audit=rows['resume']['resume_audit'],independent_32_step_parameter_difference=flat_error(c1['writer'].values(),c2['writer'].values()),final_loss_difference=rows['resume']['rows'][-1]['loss']-rows['continuous']['rows'][-1]['loss']))
assert count==384
write(out,dict(passed=True,repetition_lock_sha256=rep_sha,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),engineering_updates=count,records=records,trajectory_equivalence_claim=False))
