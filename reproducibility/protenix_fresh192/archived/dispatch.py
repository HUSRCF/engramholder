"""Bounded process recovery, same eight devices; no score-driven scheduling."""
import argparse,concurrent.futures,fcntl,os,subprocess,time,traceback
from pathlib import Path
from .common import *
R=Path('/media/PM982/engramfold/runs/protenix_fresh192_20260925')
PYP='/home/pc/anaconda3/envs/fold/bin/python'
RUNTIME='/media/PM982/onestepfold/protenix_stage0_pkg/v1_1/runtime'
OLD=R.parent/'diamondhill_plm_A_20260923'

def run_system(name,gpu,engineering):
 stage='engineering' if engineering else 'formal';out=R/stage
 manifest=R/'engineering_manifest.json' if engineering else R/'panel/inference_manifest.json'
 lock=R/'engineering_lock.json' if engineering else R/'prediction_lock.json'
 feature=OLD/'features/esmc' if engineering else R/'features/esmc'
 env=dict(os.environ,PYTHONPATH=str(R/'source/src'),HIP_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',LAYERNORM_TYPE='torch',PROTENIX_ROOT_DIR=RUNTIME,PYTHONUNBUFFERED='1')
 args=[PYP,'-m','engramfold.experiments.protenix_fresh_fourcell.predict','--manifest',str(manifest),'--checkpoint-lock',str(lock),'--output-root',str(out),'--protenix-root',RUNTIME,'--feature-root',str(feature),'--forbidden-root',str(R.parent/'train384_direction_20260920/cache'),'--system',name]
 attempts=R/'attempts'/stage/name;attempts.mkdir(parents=True,exist_ok=True)
 for i in range(1000):
  if (out/name/'complete_checked.json').exists():
   prior=read(out/name/'report.json');assert prior['complete'] and prior['manifest_sha256']==sha(manifest) and prior['checkpoint_lock_sha256']==sha(lock)
   for rec in prior['records']:
    if rec['status']=='ok':assert sha(rec['prediction_path'])==rec['prediction_sha256']
   return
  existing=list(attempts.glob('process_*.json'));attempt=max([int(p.stem.split('_')[1]) for p in existing]+[-1])+1
  log=attempts/f'process_{attempt}.log'
  with log.open('x') as f:
   p=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,env=env,cwd=R/'source',start_new_session=True)
   write(attempts/f'process_{attempt}.json',dict(pid=p.pid,started=time.time(),gpu=gpu,args=args,log=str(log)))
   rc=p.wait()
  write(attempts/f'process_{attempt}.json',dict(pid=p.pid,ended=time.time(),returncode=rc,gpu=gpu,args=args,log=str(log)))
  if rc==0 and (out/name/'complete_checked.json').exists():return
  if rc!=75:raise RuntimeError(f'system failure {name}: rc={rc}; {log}')
  time.sleep(120)
 raise RuntimeError('process safety cap')

def engineering_audit():
 import numpy as np
 from engramfold.evaluation.structure import read_atom_site_positions
 targets=read(R/'engineering_manifest.json')['targets'];history=read(R/'engineering_history.json');rows=[]
 def coords(p):
  _,pos=read_atom_site_positions(Path(p));return {k:np.asarray(v,dtype=np.float64) for k,v in pos.items()}
 for name in models():
  report=read(R/'engineering'/name/'report.json');assert report['complete'] and report['frozen_unchanged'];assert len(report['records'])==2
  for t,now,old in zip(targets,report['records'],history[name]):
   assert now['target_id']==old['target_id']==t['target_id'] and now['status']==old['status']=='ok'
   assert sha(now['prediction_path'])==now['prediction_sha256'] and sha(old['prediction_path'])==old['prediction_sha256']
   a,b=coords(now['prediction_path']),coords(old['prediction_path']);assert a.keys()==b.keys();keys=sorted(a)
   aa=np.stack([a[k] for k in keys]);bb=np.stack([b[k] for k in keys]);err=float(np.linalg.norm(aa-bb)/max(np.linalg.norm(bb),1e-12))
   assert err<=1e-3,(name,t['target_id'],err)
   assert now['full_length']==t['sequence_length'] and len(now['injection_shapes'])==4
   rows.append(dict(system=name,target_id=t['target_id'],relative_coordinate_l2=err,passed=True))
 write(R/'engineering/complete.json',dict(passed=True,models=25,predictions=50,rows=rows,model_lock_sha256=sha(R/'model_lock.json'),maximum_relative_l2=max(x['relative_coordinate_l2'] for x in rows),completed_time=time.time()))

def main():
 p=argparse.ArgumentParser();p.add_argument('--engineering',action='store_true');a=p.parse_args();stage='engineering' if a.engineering else 'formal'
 lease=(R/(stage+'.dispatch.lock')).open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if not a.engineering:
  ex=read(R/'execution_lock.json')
  for f,h in ex['files'].items():assert sha(f)==h,f
 try:
  def worker(gpu):
   for name in models()[gpu::8]:run_system(name,gpu,a.engineering)
  with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
   fs=[pool.submit(worker,gpu) for gpu in range(8)]
   for f in fs:f.result()
  if a.engineering:engineering_audit()
  else:
   records=[]
   for name in models():
    j=read(R/'formal'/name/'report.json');assert j['complete'] and j['frozen_unchanged'] and len(j['records'])==192
    assert j['manifest_sha256']==sha(R/'panel/inference_manifest.json') and j['checkpoint_lock_sha256']==sha(R/'prediction_lock.json')
    assert j['model_lock_sha256']==sha(R/'model_lock.json') and j['system']==name
    assert [x['target_id'] for x in j['records']]==[t['target_id'] for t in read(R/'panel/inference_manifest.json')['targets']]
    for rec in j['records']:
     if rec['status']=='ok':assert sha(rec['prediction_path'])==rec['prediction_sha256']
    assert any(x['status']=='ok' for x in j['records']), 'entirely failed system'
    records.extend(dict(x,system=name) for x in j['records'])
   assert len(records)==4800 and len({(x['system'],x['target_id']) for x in records})==4800
   write(R/'prediction_completion.json',dict(complete=True,records=records,execution_lock_sha256=sha(R/'execution_lock.json'),completed_time=time.time()))
 except BaseException as e:
  write(R/(stage+'.blocked.json'),dict(error=repr(e),traceback=traceback.format_exc(),time=time.time()));raise
if __name__=='__main__':main()
