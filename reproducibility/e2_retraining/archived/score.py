"""Uniform original E2 statistics, plus independent CIF round-trip scoring."""
import argparse,ast,datetime,subprocess,sys
from pathlib import Path
import numpy as np
from common import *
from support import check

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
lock,c,lk=check(r);parent=Path(lock['parent_root']);out=r/'repetition_analysis'
out.mkdir(exist_ok=True);rep_sha=sha(r/'repetition_lock.json')
if(out/'complete.json').exists():
    done=read(out/'complete.json');assert done['repetition_lock_sha256']==rep_sha and done['complete']
    raise SystemExit('ALREADY_COMPLETE')
assert sha(parent/'e1/analysis/metric_records.json')==lock['parent_e1_records_sha256']
assert sha(parent/'e2/analysis/analysis.json')==lock['parent_e2_analysis_sha256']
for run in read(r/'e2_execution_lock.json')['runs']:
    folder=r/'e2/formal'/run['name']
    assert read(folder/'repetition.json')['repetition_lock_sha256']==rep_sha
    assert read(folder/'training_complete.json')['steps']==1536
    rows=[__import__('json').loads(line)for line in(folder/'training.jsonl').read_text().splitlines()]
    assert [x['step']for x in rows]==list(range(1,1537))
subprocess.run([sys.executable,str(r/'operations/scoring_repair_20260924/analyze.py'),'--root',str(r),'--mode','e2'],check=True)

import gemmi
from openfold.np import residue_constants as rc
from engramfold.evaluation.structure import read_atom_site_positions
from engramfold.evaluation.independent import fixed_mask_metrics
from engramfold.evaluation.post_validation import ca_pdb,parse_tm_score,load_reference_lddt

def export_cif(path,seq,xyz,mask):
    doc=gemmi.cif.Document();b=doc.add_new_block('prediction');b.set_pair('_entry.id','prediction')
    entity=b.init_loop('_entity_poly.',['entity_id','type','pdbx_seq_one_letter_code_can'])
    entity.add_row(['1',gemmi.cif.quote('polypeptide(L)'),gemmi.cif.quote(seq)])
    loop=b.init_loop('_atom_site.',['group_PDB','id','type_symbol','label_atom_id','label_alt_id','label_comp_id','label_asym_id','label_entity_id','label_seq_id','pdbx_PDB_ins_code','Cartn_x','Cartn_y','Cartn_z','occupancy','B_iso_or_equiv','auth_seq_id','auth_comp_id','auth_asym_id','auth_atom_id','pdbx_PDB_model_num'])
    k=0
    for i,aa in enumerate(seq):
        res=rc.restype_1to3[aa]
        for j,atom in enumerate(rc.atom_types):
            if not mask[i,j]:continue
            k+=1
            loop.add_row(['ATOM',str(k),atom[0],atom,'.',res,'A','1',str(i+1),'?',*[format(float(v),'.17g')for v in xyz[i,j]],'1.00','0.00',str(i+1),res,'A',atom,'1'])
    tmp=path.with_suffix('.tmp');doc.write_file(str(tmp));tmp.replace(path)

targets=[t for t in read(r/'data/evaluation144.json')['targets']if t['panel']=='confirm96'];jobs=[];cifs=[]
for run in read(r/'e2_execution_lock.json')['runs']:
    folder=r/'e2/formal'/run['name'];done=read(folder/'complete.json')
    assert [x['target_id']for x in done['records']]==[t['target_id']for t in targets]
    dst=out/'cif'/run['name'];dst.mkdir(parents=True,exist_ok=True)
    for t,rec in zip(targets,done['records'],strict=True):
        tid=t['target_id'];src=folder/'predictions'/(tid+'.npz')
        if rec['status']!='ok':jobs.append((run['name'],t,rec,src));continue
        assert sha(src)==rec['prediction_sha256'];data=np.load(src)
        xyz=data['coordinates'];mask=data['mask'];seq=str(data['sequence']);assert seq==t['sequence']
        cf=dst/(tid+'.cif');export_cif(cf,seq,xyz,mask)
        _,atoms=read_atom_site_positions(cf,label_asym_id='A')
        px=np.zeros_like(xyz);pm=np.zeros_like(mask)
        for(i,atom),v in atoms.items():
            j=rc.atom_order[atom];px[i-1,j]=v;pm[i-1,j]=1
        assert np.array_equal(pm,mask)
        assert np.array_equal(px[mask.astype(bool)],xyz[mask.astype(bool)])
        path=dst/(tid+'.npz');np.savez(path,coordinates=px,mask=pm,sequence=seq)
        jobs.append((run['name'],t,dict(status='ok',prediction_sha256=sha(path)),path))
        cifs.append(dict(system=run['name'],target_id=tid,cif_sha256=sha(cf),masked_coordinates_fp32_exact=True))

# Extract the exact accepted score function, without executing its command-line main.
source=r/'operations/scoring_repair_20260924/analyze.py'
node=next(x for x in ast.parse(source.read_text()).body if isinstance(x,ast.FunctionDef)and x.name=='score_jobs')
from concurrent.futures import ThreadPoolExecutor
exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),globals())
independent=score_jobs(jobs,out/'cif_scoring',True)
original=read(r/'e2/analysis/metric_records.json');by={(x['system'],x['target_id']):x for x in original}
metrics=['ca_lddt','residue_ca_lddt','tm_score_fixed_full_length']
maxdiff={m:max(abs(x[m]-by[x['system'],x['target_id']][m])for x in independent)for m in metrics}
assert all(v==0 for v in maxdiff.values()),maxdiff
new=read(r/'e2/analysis/analysis.json');old=read(parent/'e2/analysis/analysis.json')
comparison={m:{k:dict(original=old['metrics'][m][k]['mean'],repeat=new['metrics'][m][k]['mean'],difference=new['metrics'][m][k]['mean']-old['metrics'][m][k]['mean'],repeat_ci95=new['metrics'][m][k]['ci95'])for k in ['B_R','T_C','native_change']}for m in metrics}
write(out/'historical_comparison.json',comparison);write(out/'cif_audit.json',dict(records=cifs,scoring_maxabs=maxdiff))
lines=['# E2 九组原配方重训练结果','','同种子重新训练，不与历史重复当作独立新种子。固定无 C 基线来自原 E1；R1新特征重放不在此完成声明内。','','|主指标比较|历史值|本次值|本次95%目标条件区间|','|---|---:|---:|---|']
for k,x in comparison['ca_lddt'].items():lines.append(f"|{k}|{x['original']:+.8f}|{x['repeat']:+.8f}|{x['repeat_ci95']}|")
lines+=['','所有864正式尝试结束后统一评分；新CIF独立解析后三项分数与直接坐标评分逐项一致。','完整三指标、种子/目标和原21项统计见 ../e2/analysis/analysis.json；不以是否保持阳性选择复现实例。']
(out/'results.md').write_text('\n'.join(lines)+'\n')
write(out/'complete.json',dict(complete=True,repetition_lock_sha256=rep_sha,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),fits=9,training_updates=13824,new_predictions=864,failures=new['new_failures'],analysis_sha256=sha(r/'e2/analysis/analysis.json'),comparison_sha256=sha(out/'historical_comparison.json'),cif_audit_sha256=sha(out/'cif_audit.json')))
