import argparse,time,hashlib
from pathlib import Path
from e3_common import *
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--index',type=int,required=True);a=p.parse_args();r=a.root
c,old,parent,lk=contract(r,True);gate=read(r/'engineering/complete.json');assert gate['passed']and gate['execution_lock_sha256']==lk
model,config=pc.load_model(parent,old);prep=pc.Features(parent,config)
items=[(phase,t)for phase in ['train','eval']for t in target_list(parent,phase)];done=[]
for phase,t in items[a.index::16]:
 path=r/'references'/phase/(t['target_id']+'.pt');f,e,_=prep.get(t)
 if path.with_suffix('.json').exists():
  anchors,receipt=load_reference(r,phase,t,f,lk);done.append(receipt);continue
 anchors,cert=certify_reference(model,f,phase)
 save(path,dict(anchors=anchors,execution_lock_sha256=lk,phase=phase,target_id=t['target_id']))
 rec=dict(**cert,phase=phase,target_id=t['target_id'],sequence_sha256=hashlib.sha256(t['sequence'].encode()).hexdigest(),execution_lock_sha256=lk,sha256=sha(path))
 write(path.with_suffix('.json'),rec);done.append(rec);print('REFERENCE',phase,t['target_id'],cert['seconds'],flush=True)
assert state_hash(model)==old['frozen_backbone_sha256']
write(r/'references'/f'shard_{a.index}.json',dict(complete=True,execution_lock_sha256=lk,records=done,no_labels=True))
