"""Reconstruct the completed Stage A contrasts from locked target scores, not CIFs."""
import hashlib,json
from pathlib import Path
import numpy as np

def verify(root):
    root=Path(root); ev=root/'evidence'
    load=lambda n:json.loads((ev/(n+'.json')).read_text())
    a=load('openfold_esmc_A_summary'); c=load('openfold_esmc_A_execution_lock'); rows=load('openfold_esmc_A_records')
    actual=hashlib.sha256((ev/'openfold_esmc_A_execution_lock.json').read_bytes()).hexdigest()
    provenance=root/'notes/writing_branch_20260922/bundled_source_provenance.json'
    if provenance.exists():
        binding=json.loads(provenance.read_text())['evidence/openfold_esmc_A_execution_lock.json']
        assert actual==binding['bundled_sha256']
        actual=binding['original_sha256']
    assert actual==a['execution_lock_sha256']
    assert len(c['new_runs'])==33 and c['steps']==1536 and c['train_size']==384
    assert len(rows)==7200 and all(x['status']=='ok' for x in rows)
    assert len({(x['panel'],x['target_id'],x['system']) for x in rows})==7200
    assert a['n_new_predictions']==4752 and a['new_failures']==0
    checks=0
    for panel,n in [('confirm96',96),('length48',48)]:
      for metric,md in a['panels'][panel].items():
        ids=md['cells']['C_last']['factor_rotation']['target_ids'];assert len(ids)==len(set(ids))==n
        by={(x['system'],x['target_id']):x[metric] for x in rows if x['panel']==panel}
        assert len(by)==50*n
        get=lambda name:np.array([by[name,i] for i in ids])
        draw=np.random.default_rng(20260926).integers(n,size=(20000,n))
        def check(x,obj):
          nonlocal checks
          v=x.mean((0,1));boot=v[draw].mean(1)
          assert obj['target_ids']==ids
          assert np.max(np.abs(v-obj['per_target']))<1e-12
          assert abs(v.mean()-obj['mean'])<1e-12
          assert np.max(np.abs(np.quantile(boot,[.025,.975])-obj['ci95']))<1e-12
          assert np.max(np.abs(x.mean((0,2))-obj['per_seed']))<1e-12
          assert np.max(np.abs(x.mean((1,2))-obj['per_rotation']))<1e-12
          assert int((v>0).sum())==obj['positive_targets']
          p=(1+(np.abs(boot-v.mean())>=abs(v.mean())).sum())/20001
          assert abs(p-obj['centered_bootstrap_p'])<1e-12
          checks+=1
        arrays={}
        for feat in ['E_last','C_last']:
          f=np.stack([get(f'{feat}_factor_native_s{s}') for s in c['formal_seeds']])
          g=np.stack([get(f'{feat}_generic_plus_native_s{s}') for s in c['formal_seeds']])
          fr=np.stack([[get(f'{feat}_factor_r{r}_s{s}') for s in c['formal_seeds']] for r in c['rotation_seeds']])
          gr=np.stack([[get(f'{feat}_generic_plus_r{r}_s{s}') for s in c['formal_seeds']] for r in c['rotation_seeds']])
          df=f[None]-fr;dg=g[None]-gr;psi=df-dg
          arrays[feat]=(f,g,df,psi)
          cell=md['cells'][feat]
          for k,v in [('native',f),('rotated_factor',fr),('gplus',g),('rotated_gplus',gr),('query',get('query'))]:assert abs(v.mean()-cell['means'][k])<1e-12
          for k,x in [('factor_rotation',df),('gplus_rotation',dg),('interaction',psi),('native_minus_gplus',(f-g)[None])]:check(x,cell[k])
        ef,eg,edf,epsi=arrays['E_last'];cf,cg,cdf,cpsi=arrays['C_last']
        for k,x in [('direction_change',cdf-edf),('psi_change',cpsi-epsi),('native_gain',(cf-ef)[None]),('gplus_gain',(cg-eg)[None]),('native_gplus_gap_change',((cf-cg)-(ef-eg))[None])]:check(x,md['plm_interactions'][k])
    md=a['panels']['confirm96']['ca_lddt'];ps={'Psi_C':md['cells']['C_last']['interaction']['centered_bootstrap_p'],'K':md['plm_interactions']['direction_change']['centered_bootstrap_p'],'J':md['plm_interactions']['psi_change']['centered_bootstrap_p']}
    adjusted={};running=0
    for i,(k,p) in enumerate(sorted(ps.items(),key=lambda z:z[1])):
      running=max(running,min(1,(3-i)*p));adjusted[k]=running
    assert adjusted==a['key_secondary_holm_p']
    return dict(passed=True,records=len(rows),new_predictions=4752,failures=0,verified_contrasts=checks,holm=adjusted,scope='Score-array reconstruction; no CIF rescoring. Observed-panel follow-up; no Fresh96 results.',inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ev.glob('openfold_esmc_A_*.json') if 'verification' not in p.name})

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1];result=verify(root)
    (root/'evidence/openfold_esmc_A_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
