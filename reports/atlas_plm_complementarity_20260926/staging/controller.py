import concurrent.futures,datetime,json,os,subprocess,time,traceback
from pathlib import Path
root=Path('/media/PM982/engramfold/runs/atlas_plm_complementarity_20260926')
py='/media/PM982/engramfold/folding_e2e_20260921/venvs/atlasfold-rocm/bin/python'
deps='/media/PM982/engramfold/folding_e2e_20260921/deps'
env=dict(os.environ,HF_HUB_OFFLINE='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONHASHSEED='0',PYTHONPATH=f'{root}/source/src:{deps}/atlas_training_compat:{deps}/atlas_d3')
(root/'logs').mkdir(exist_ok=True)
def status(state,**kw):
    p=root/'STATUS.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(dict(state=state,time=datetime.datetime.now().astimezone().isoformat(),**kw),indent=2)+'\n');t.replace(p)
def run(stage,index,gpu,module='engramfold.experiments.atlas_plm_probe'):
    for attempt in range(1,3):
        log=root/'logs'/f'{stage}_{index}_attempt{attempt}.log'
        args=[py,'-u','-m',module,'--root',str(root)]
        if module.endswith('atlas_plm_probe'):args+=['--stage',stage,'--index',str(index)]
        with log.open('w') as stream:
            p=subprocess.Popen(['timeout','3500',*args],stdout=stream,stderr=subprocess.STDOUT,env=dict(env,HIP_VISIBLE_DEVICES=str(gpu)),start_new_session=True)
            (root/'logs'/f'{stage}_{index}_pid.json').write_text(json.dumps(dict(pid=p.pid,gpu=gpu,attempt=attempt))+'\n')
            code=p.wait()
        if code==0:return
        text=log.read_text().lower()
        if attempt==1 and any(x in text for x in ('hip error','hsa_status_error','memory access fault')):time.sleep(10);continue
        raise RuntimeError(f'{stage} index{index} failed {code}: {log}')

try:
    status('FEATURES_AND_ESMC_REPLAY')
    def replay():
        receipt=root/'esmc_propagation/complete.json'
        if receipt.exists():
            completed=json.loads(receipt.read_text());contract=json.loads((root/'lock.json').read_text())
            assert completed['complete'] and completed['lock_sha256']==contract.get('reused_esmc_replay_parent_lock',completed['lock_sha256'])
            assert completed['full_forwards']==10
            return
        run('replay',0,5,'engramfold.experiments.atlas_esmc_propagation')
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        jobs=[pool.submit(run,'features',0,6),pool.submit(run,'features',1,7),pool.submit(run,'labels',0,''),pool.submit(replay)]
        # Replay is a separate observation; report its failure without altering probe protocol.
        for future in jobs[:3]:future.result()
        try:jobs[3].result()
        except Exception as exc:(root/'esmc_replay_error.json').write_text(json.dumps(dict(error=str(exc)))+'\n')
    for target in json.loads((root/'lock.json').read_text())['engineering_targets']:
        assert json.loads((root/'cache'/f'{target}_engineering.json').read_text())['passed']
    status('GPU_SMOKE');run('smoke',0,6)
    status('FORMAL_RUNNING')
    def worker(gpu):
        for index in range(gpu,9,6):run('train',index,gpu)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures=[pool.submit(worker,gpu)for gpu in range(6)]
        for f in futures:f.result()
    status('ANALYSIS')
    with (root/'logs/analysis.log').open('w') as stream:subprocess.run([py,str(root/'staging/analyze.py')],stdout=stream,stderr=subprocess.STDOUT,env=env,check=True)
    status('COMPLETE',fits=9,formal_probe_updates=13824,replay_complete=(root/'esmc_propagation/complete.json').exists())
except Exception:
    status('STOPPED',error=traceback.format_exc());raise
