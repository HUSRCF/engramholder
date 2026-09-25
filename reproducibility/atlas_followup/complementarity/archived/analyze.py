import json,hashlib
from pathlib import Path
import numpy as np
r=Path('/media/PM982/engramfold/runs/atlas_plm_complementarity_20260926')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
c=read(r/'lock.json');lk=sha(r/'lock.json');arms=c['arms'];seeds=c['seeds'];ids=[t['target_id']for t in c['dev']]
values={};curves={};train={};files={}
for arm in arms:
    values[arm]=[];curves[arm]={};train[arm]=[]
    for seed in seeds:
        folder=r/'formal'/f'{arm}_s{seed}';d=read(folder/'complete.json')
        assert d['complete'] and d['lock_sha256']==lk and d['steps']==1536 and d['parameters']==131520
        assert sha(folder/'latest.pt')==d['checkpoint_sha256']
        end=read(folder/'dev_step1536.json');assert [x['target_id']for x in end]==ids
        values[arm].append([x['ce']for x in end])
        train[arm].append(float(np.mean([x['ce']for x in read(folder/'train_final.json')])))
        for step in (0,384,768,1536):
            f=folder/f'dev_step{step}.json';curves[arm].setdefault(str(step),[]).append(float(np.mean([x['ce']for x in read(f)])));files[str(f.relative_to(r))]=sha(f)
    values[arm]=np.array(values[arm])
draws=np.random.default_rng(20260926).integers(0,8,(20000,8));comparisons={}
for name,left in [('primary_native_minus_esmc','native_view'),('secondary_permuted_minus_esmc','esmc_permuted')]:
    dif=values[left]-values['esmc'];target=dif.mean(0);boot=target[draws].mean(1)
    comparisons[name]=dict(mean=float(target.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),per_seed=dif.mean(1).tolist(),per_target=target.tolist(),positive_targets=int((target>0).sum()))
out=r/'analysis';out.mkdir(exist_ok=True)
summary=dict(complete=True,lock_sha256=lk,parameters_per_run=131520,fits=9,updates=13824,engineering_updates=36,
             target_ids=ids,means={a:float(values[a].mean())for a in arms},train_means={a:float(np.mean(train[a]))for a in arms},
             values={a:values[a].tolist()for a in arms},curves=curves,comparisons=comparisons,
             bootstrap='20k target resamples after paired seed aggregation; descriptive observed Dev8',files=files)
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['# Atlas/ESMC条件结构probe结果','',
       '固定Train96/Dev8、三臂三个种子、每臂1536步；仅小头拟合，不是新的folding训练或lDDT评测。所有臂131,520个可训练参数。',
       '原生对照使用原生single的非零第二视图；ESMC与其置乱均经固定128维投影。所有输入有原生single及attention-derived pair信息。','',
       '|输入|最终Train96 CE|最终Dev8 CE|','|---|---:|---:|']
for arm in arms:lines.append(f"|{arm}|{summary['train_means'][arm]:.6f}|{summary['means'][arm]:.6f}|")
lines+=['','|比较（正值支持ESMC）|Dev8均值|条件目标95%区间|三个种子均值|','|---|---:|---|---|']
for name,d in comparisons.items():lines.append(f"|{name}|{d['mean']:+.6f}|[{d['ci95'][0]:+.6f}, {d['ci95'][1]:+.6f}]|{d['per_seed']}|")
lines+=['','这是已观察Dev8上的开发证据。区间不包含完整重新训练不确定性；置乱不是严格条件独立检验。固定最终1536步，不按曲线选checkpoint。',
        '阳性只支持指定读出/归一化/压缩/预算下的增量；阴性不能证明ESMC信息冗余或整个函数类中不存在有用补偿。不据结果自动启动新writer训练。','',
        '逐目标、逐种子、固定节点和长程/近距离次要读数保存在formal目录，主终点没有被子集替换。']
(out/'results.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(comparisons,indent=2))
