"""Summarize the complete fixed observations; no structural rescoring."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
c=read(root/'lock.json');lock_sha=sha(root/'lock.json')
records=[];checks=[]
for target in c['targets']:
    folder=root/'results'/target['target_id']
    done=read(folder/'complete.json');assert done['complete'] and done['lock_sha256']==lock_sha
    assert done['systems']==10 and done['forward_count']==13
    for f,h in done['files'].items():
        if (folder/f).exists():assert sha(folder/f)==h,f
    for f in ('query_smoke.json','adapted_smoke.json'):
        assert read(folder/f)['passed']
    for run in c['runs']:
        d=read(folder/f"{run['name']}.json")
        assert d['run']==run and d['lock_sha256']==lock_sha and len(d['sites'])==22
        assert all(len(v)==5 for v in d['sites'].values())
        records.append(d)
    checks.append(dict(target=target['target_id'],complete=done,engineering_pass=True))
adapted=[d for d in records if d['run']['branch']!='query']
sites=['after_injection','lm_block1','lm_block2','lm_block3','lm_block4','projection_ln','projection_out','trunk_input','trunk_end']
summary={}
for branch in ('parent','C_only','head_only'):
    group=[d for d in adapted if d['run']['branch']==branch]
    summary[branch]={}
    for stream in ('z','s'):
        summary[branch][stream]={}
        for site in sites:
            key=site+'.'+stream
            if key not in group[0]['sites']:continue
            values=[v for d in group for v in d['sites'][key]]
            entry={}
            for stat in ('relative_delta_rms','delta_rms','query_rms','channel_constant_delta_energy_fraction'):
                x=[v[stat] for v in values if v[stat] is not None]
                entry[stat]=dict(median=float(np.median(x)),min=min(x),max=max(x)) if x else None
            summary[branch][stream][site]=entry

report=['# Atlas同位置传播定位结果','',
        '固定两条已观察链，3个Native父模型及各自C_only/head_only：20次观测推理＋6次工程重放，零训练。',
        '本报告不读取真实结构标签、不重评CIF；最终结构变化是相对Query，不能解释为质量改善。','',
        '两链Query重复的22位置×5轮状态均逐元素一致；Query与首种子parent各自有/无观察器输出一致。完整长度、冻结参数、源码与checkpoint哈希验收通过。','',
        '另保留首次工程执行的6次Query前向：模型检查已通过，CPU配准SVD因Torch未编译LAPACK失败；恢复仅改为NumPy同一3×3公式。实际总执行32次（26次完整矩阵/验收＋6次原工程前向），原失败日志未覆盖。重复一致是可复现性检查，不是有限精度误差的严格上界。','',
        '## 既有通道校准，已纳入证据','',
        '6组续训、960预测、0失败；完成收据匹配原始analysis/records哈希。C_only−head_only为−0.00000513，95% CI [−0.00069698,+0.00074909]；C_only−Query为−0.001287，CI跨零。未建立校准收益，且Native-only不检验旋转交互。','',
        '## 父模型pair路径：同位置相对Query变化','',
        '下表为两目标×三种子×五轮共30个观测的中位数及范围，仅描述这些固定案例；不是独立样本区间。','',
        '|位置|Δ RMS / 同位置Query RMS：中位数 [范围]|Δ绝对RMS中位数|Query RMS中位数|',
        '|---|---:|---:|---:|']
for site in sites:
    row=summary['parent']['z'][site];x=row['relative_delta_rms']
    report.append(f"|{site}|{x['median']:.6g} [{x['min']:.6g}, {x['max']:.6g}]|{row['delta_rms']['median']:.6g}|{row['query_rms']['median']:.6g}|")
report +=['','不能将不同层的裸范数比解释成传递率。相对量的分母随位置变化，因此同时保留绝对Δ和Query尺度；LayerNorm、投影、融合和主干的作用只能在对应位置比较。','',
           '## 两支路与续训分支','',
           '|分支|LM末端pair相对变化|投影pair相对变化|主干末端pair相对变化|LM末端single相对变化|主干末端single相对变化|',
           '|---|---:|---:|---:|---:|---:|']
for branch in summary:
    vals=[summary[branch][st][site]['relative_delta_rms']['median'] for st,site in [('z','lm_block4'),('z','projection_out'),('z','trunk_end'),('s','lm_block4'),('s','trunk_end')]]
    report.append('|'+branch+'|'+'|'.join(f'{v:.6g}'for v in vals)+'|')
report += ['','## 父模型末轮的逐目标、逐种子读数','',
           '|目标|种子|注入后pair相对差|LM4 pair相对差|投影pair相对差|融合pair相对差|主干末端pair相对差|主干末端single相对差|',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
for d in adapted:
    if d['run']['branch']!='parent':continue
    vals=[d['sites'][key][-1]['relative_delta_rms'] for key in ['after_injection.z','lm_block4.z','projection_out.z','trunk_input.z','trunk_end.z','trunk_end.s']]
    report.append(f"|{d['target_id']}|{d['run']['seed']}|"+'|'.join(f'{v:.6g}' for v in vals)+'|')
report +=['','## 全部18个适配实例的最终结构变化','',
           '|目标|分支|训练种子|Cα距离差RMS (Å)|对Query配准RMSD (Å)|',
           '|---|---|---:|---:|---:|']
for d in adapted:
    s=d['structure_change'];r=d['run']
    report.append(f"|{d['target_id']}|{r['branch']}|{r['seed']}|{s['ca_pair_distance_delta_rms']:.6g}|{s['aligned_ca_rmsd']:.6g}|")
report +=['','## 范围与后续','',
           '这是一项零训练、两固定案例的传播定位，不能从中证明加入反馈一定有效，也不能把小状态差当作主干完全忽略更新。所有五轮和三个种子的读数保留在results，未挑最有利种子/轮次。',
           '状态条件化writer、晚层接口仍未启动；本次负校准结果保留。源代码路径显示LM stack不接收回收状态，适配历史在后续融合处进入；这与“反馈是否能改善适配”是两个问题。','',
           'Query完整状态原文件留在DiamondHill的本实验results目录（每目标query_states.pt）；本机保存哈希、全臂统计、坐标与CIF。']
out=root/'analysis';out.mkdir(exist_ok=True)
(out/'results.md').write_text('\n'.join(report)+'\n')
(out/'summary.json').write_text(json.dumps(dict(summary=summary,lock_sha256=lock_sha,records=len(records),full_forwards=26),indent=2)+'\n')
(out/'audit.json').write_text(json.dumps(dict(passed=True,targets=checks,local_query_tensors_copied=False,statistics_only=True),indent=2)+'\n')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axs=plt.subplots(1,2,figsize=(12,4),sharey=True)
for ax,target in zip(axs,c['targets']):
    for branch,color in [('parent','#2166ac'),('C_only','#b2182b'),('head_only','#4d9221')]:
        group=[d for d in adapted if d['target_id']==target['target_id'] and d['run']['branch']==branch]
        y=np.array([[v['relative_delta_rms']for site in sites for v in d['sites'][site+'.z']] for d in group]).reshape(3,len(sites),5)
        median=np.median(y,axis=(0,2));low=y.min(axis=(0,2));high=y.max(axis=(0,2))
        ax.plot(range(len(sites)),median,'o-',label=branch,color=color,ms=3)
        ax.fill_between(range(len(sites)),low,high,color=color,alpha=.12)
    ax.set_title(target['target_id']);ax.set_xticks(range(len(sites)),['Injection','LM1','LM2','LM3','LM4','Proj LN','Proj out','Fusion','Trunk end'],rotation=45,ha='right')
    ax.set_yscale('log');ax.grid(alpha=.25);ax.legend(fontsize=8)
axs[0].set_ylabel('Same-site delta RMS / Query RMS')
fig.suptitle('Atlas pair-state changes: medians and ranges over 3 seeds x 5 passes')
fig.tight_layout();fig.savefig(out/'propagation.png',dpi=180);fig.savefig(out/'propagation.pdf');plt.close(fig)
print(out/'results.md')
