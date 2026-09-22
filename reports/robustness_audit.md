# 稳健性、目标分布与暴露边界核对

数值快照生成于 **2026-09-21T14:40:05.823752+00:00**；只是本地已保存产物，不是远端训练实时状态。只计算描述性摘要，不重跑结构预测、不增加bootstrap比较、不按结果删样本。

## 1. 聚合顺序与可复算范围

每个目标先计算 $d_i=\frac13\sum_s[S_{N,i,s}-\frac13\sum_r S_{R,i,s,r}]$，再对目标等权平均。长链与短链不会按pair数混在一起加权；已报告的target bootstrap条件于固定模型。

本次从逐模型、逐目标原始score记录重新构造12个比较，按目标ID对齐，不依赖文件碰巧同序。原目标差数组、均值与保存汇总最大误差均小于1e-12；另验训练集合嵌套、目标ID交集及D中位数。评分本身未重算CIF。

复现命令：

```bash
python scripts/audit_evidence_coverage.py
```

脚本当前从同机实验仓库同步缺少的证据快照；路径在源码内显式给出。快照自身、来源绝对路径、SHA256和源码/协议SHA均保存于[审计JSON](coverage_robustness_audit.json)。无需GPU。后续重跑若同步了新的文件，时间和哈希会相应改变，不会冒充原锁定记录。

## 2. 三个训练种子与三个旋转的边际结果

下列值均为Native−Rotated；G+那一行是Native−G+，不能与旋转行互换。每组三个边际并非三个独立确认实验。

| 对比 | 三个训练种子均值 | 三个旋转均值 | 9个种子×旋转单元为正 |
|---|---|---|---|
| Mini Train24 tangent / Confirm96-B | +0.06970 / +0.04807 / +0.05354 | +0.04659 / +0.06418 / +0.06054 | 9/9 |
| Mini Train24 full / Confirm96-B | +0.05836 / +0.03995 / +0.04118 | +0.03877 / +0.05526 / +0.04545 | 9/9 |
| Mini Train96 full / Confirm96-B | +0.03581 / +0.02776 / +0.01159 | +0.02123 / +0.02863 / +0.02529 | 9/9 |
| Mini Train384 full / Confirm96-B | +0.04001 / +0.03375 / +0.04477 | +0.02415 / +0.05845 / +0.03592 | 9/9 |
| Mini Train384 full / Length48 | +0.03998 / +0.03579 / +0.05180 | +0.02394 / +0.05869 / +0.04493 | 9/9 |
| Tiny Train24 tangent / Confirm96-B | +0.02232 / +0.02492 / +0.01894 | +0.02590 / +0.02217 / +0.01811 | 9/9 |
| Mini Train24 tangent mean-preserving / Confirm96-B | +0.10719 / +0.10041 / +0.09415 | +0.09507 / +0.11760 / +0.08909 | 9/9 |
| Mini Train24 full vs G+ / Confirm96-B (separate recipe) | +0.03902 / +0.05059 / +0.06729 | 不适用 | 不适用 |
| OpenFold Train96 full / confirm96 | +0.02073 / +0.02901 / +0.00969 | +0.01799 / +0.02220 / +0.01924 | 9/9 |
| OpenFold Train96 full / length48 | +0.00877 / +0.01072 / +0.01166 | +0.00854 / +0.00824 / +0.01437 | 9/9 |
| OpenFold Train384 full / confirm96 | +0.01074 / -0.01428 / +0.01263 | +0.00067 / +0.00702 / +0.00140 | 6/9 |
| OpenFold Train384 full / length48 | +0.01014 / +0.00621 / +0.01742 | +0.01060 / +0.00610 / +0.01707 | 8/9 |

**边际一致不等于所有训练实例一致：**OpenFold Train384 Length48三个种子、三个旋转边际均为正，但9个交叉单元只有8个为正；Confirm96同配置为6/9，且一个种子边际反向。因此不能把“三个边际都正”写成“每个拟合模型都显示优势”。完整3×3数值保存在JSON中，不只报告计数。

Protenix这里列出的七个方向比较均为9/9正，说明当前差异不是仅靠一个旋转或一个种子出现。但只有三种子和三个固定旋转，无法由此保证对新的训练随机性或任意正交旋转均稳定。

## 3. 逐目标差异分布

分位数对全部目标计算，未删除失败、不利或不可分类目标。这里的P10/P90仅为分布摘要，不是新成功门槛或筛选依据。

| 对比 | 均值 | P10 | P25 | 中位数 | P75 | P90 | 正向/总数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mini Train24 tangent / Confirm96-B | +0.05710 | -0.00778 | +0.00408 | +0.02952 | +0.10335 | +0.15475 | 78/96 |
| Mini Train24 full / Confirm96-B | +0.04650 | -0.02194 | -0.00021 | +0.02049 | +0.08858 | +0.15697 | 72/96 |
| Mini Train96 full / Confirm96-B | +0.02505 | -0.03030 | -0.00113 | +0.02020 | +0.04043 | +0.08194 | 71/96 |
| Mini Train384 full / Confirm96-B | +0.03951 | -0.01572 | +0.00029 | +0.02288 | +0.05665 | +0.12555 | 72/96 |
| Mini Train384 full / Length48 | +0.04252 | -0.00582 | +0.01263 | +0.02288 | +0.05604 | +0.11934 | 41/48 |
| Tiny Train24 tangent / Confirm96-B | +0.02206 | -0.01740 | -0.00730 | +0.01236 | +0.04165 | +0.06561 | 61/96 |
| Mini Train24 tangent mean-preserving / Confirm96-B | +0.10059 | -0.00814 | +0.01069 | +0.07363 | +0.16181 | +0.26500 | 81/96 |
| Mini Train24 full vs G+ / Confirm96-B (separate recipe) | +0.05230 | -0.02900 | -0.00184 | +0.02793 | +0.10439 | +0.17699 | 71/96 |
| OpenFold Train96 full / confirm96 | +0.01981 | -0.02779 | -0.00918 | +0.01420 | +0.03625 | +0.08290 | 59/96 |
| OpenFold Train96 full / length48 | +0.01038 | -0.01032 | -0.00429 | +0.00750 | +0.01955 | +0.03920 | 34/48 |
| OpenFold Train384 full / confirm96 | +0.00303 | -0.03290 | -0.01553 | +0.00221 | +0.01881 | +0.05273 | 50/96 |
| OpenFold Train384 full / length48 | +0.01126 | -0.00844 | -0.00169 | +0.00787 | +0.02202 | +0.04331 | 35/48 |

这揭示了平均数之外的范围：Protenix Train384/Confirm96与Length48中位数都约+0.02288，低于各自均值；较大正向尾部确实会抬高平均值，但72/96与41/48正向表明不只是几个极端目标。它不意味着所有蛋白都受益。OpenFold Train384/Confirm96只有50/96为正，中位数约+0.00221，和接近零的均值/跨零区间一致，不能包装成稳定普遍规律。

**这些摘要能排除“完全靠一个极端点/一个边际实例”的某些简单解释，不能证明结论不受任何聚合或选择影响。**目标并未保证家族独立，已经观察目标上的后续研究选择与只用三训练种子的有限覆盖，都不在固定模型target bootstrap中。我们未尝试多个子集，未新增删离群值后结果，也未用这些摘要替代锁定主指标。

## 4. 主锁定与后续扩展的时间关系

以下区分协议/文件记录的先后与可直接读出的时间字段。文件名中的20260923等训练种子只是整数标识，**不能当实际训练日期**。

| 阶段 | 当时锁定/使用面板 | 后续身份与可核查记录 |
|---|---|---|
| 原Factor−旧Generic | Confirm96-A首次锁定比较 | 本报告保留原主终点；旧Generic不补作G+ |
| 2026-09-19方向研究 | 新Confirm96-B；唯一主比较Train24 tangent/384；Full/384为同研究次要比较 | [执行锁](../evidence/protenix_direction_lock.json)及[完成审计](../evidence/protenix_direction_completion.json)记录模型/源代码/配对随机性；执行锁本身没有created_utc字段，不能据文件名伪造精确时间戳 |
| Tiny/均值保持/Train96完整旋转 | 复用已观察Confirm96-B | [扩展汇总](../evidence/protenix_extensions.json)明确development_diagnostic_not_confirmation；不是第二次盲确认 |
| Train384及数据交互 | 同一已观察Confirm96-B | [训练锁](../evidence/protenix_train384_training_lock.json)、[评测锁](../evidence/protenix_train384_eval_lock.json)；交互是后续补充分析 |
| G+固定checkpoint补评 | 2026-09-20T09:00:07Z锁定，复用Confirm96-B | [锁](../evidence/protenix_gplus_lock.json)明确independent_confirmation=false；原Dev选LR已发生，六checkpoint不重新选择 |
| Length48 v2 | selection 2026-09-20T13:47:39Z；evaluation 13:52:24Z | [selection](../evidence/length48_selection_lock.json)、[evaluation](../evidence/length48_evaluation_lock.json)；对固定Protenix模型是新锁面板 |
| Length48补工程检查 | 正式启动后、打开评分前 | [补充门槛](../evidence/length48_supplement_gate.json)明确此先后，28工程预测与672正式预测分开；不能改写成全都启动前通过 |
| OpenFold Train96/384 | 复用Confirm96-B与Length48 | 对AF2均是已观察目标的跨底座后续验证，不因底座变化恢复盲确认身份 |

这次核对的是现存锁、哈希、协议与产物的一致性；没有新增独立远端日志取证。缺少机器可读绝对时间的旧阶段只给记录所支持的先后/协议身份，不以当前文件mtime代替历史锁定时间。

## 5. 训练/评测隔离与累计暴露

直接核对[Train96 split](../evidence/protenix_split96.json)、[Train384 split](../evidence/protenix_split384.json)、[Length48清单](../evidence/length48_manifest.json)：Train24⊂Train96⊂Train384；Train384与Dev8、Confirm96-B及旧两组96无目标ID交集，Length48与这些已记录集合也无ID交集。**目标ID不重合只是一层检查，不等于序列/家族隔离。**

原方向协议另规定序列筛查：旧query/new-chain比较至少50个配对残基、identity≥0.3且覆盖较短序列≥0.7即排除；旧MSA语料采用候选长度覆盖率门槛。Length48 v2使用单独封存的BLAST/HSP筛查，不应把前一种算法误称为v2实现。v2检索了584条记录query及15,605条参考序列，并以128条开发实例校准；规则/合并策略由selection锁引用的原协议确定，本轮没有重新搜索。

四个真实CATH同家族漏检被此前审计归为未检出满足条件的HSP，不是已证明严格同家族排除。v2只承诺封存序列规则下的长度外推；不能据合成控制通过推出真实远缘关系全覆盖。

| 暴露类型 | 记录覆盖 | 仍未解决 |
|---|---|---|
| 当前适配训练 | 384条；Train96内嵌、新增288条 | 原生和旋转已见同一训练集合，不能从历史清单删链消除暴露 |
| 开发/配方选择 | Dev8加已记录校准与观察面板 | 反复观察与模型选择偏差不能由target bootstrap消除 |
| CATH旧审计 | 原Train96+Dev8共104条 | 12条历史未解决；新增288条尚未CATH审计 |
| 已观察确认链 | Confirm96-A/B共192条；之后Length48也已观察 | 不再用于后续项目的首次盲确认宣称 |
| 历史teacher/研发完整暴露 | [累计记录](../evidence/cumulative_exposure.json)明确不是全部历史暴露认证 | 某些早期teacher、调参信息来源未完整重建 |
| 基础模型预训练 | 冻结主干/PLM继承已有知识 | 未建立与训练结构、家族或长度的预训练隔离 |

现有cumulative_exposure文件的584条记录union=384训练+8开发+192旧确认，是Length48选择前的快照；加入后来已观察48条，**这里已列出的query union为632**，但这仍不是所有历史研发暴露的完整证明。未知注释保持未知，不能默认为新家族。

## 6. 本轮能得出的审阅判断

- Protenix主方向均值有目标覆盖和已测seed/rotation一致性支持，尤其Length48不是仅复用旧短链；但不能称普遍定律。
- OpenFold存在清晰条件边界，短链Train384接近零及G+长链胜出必须与正结果放在同一视野。
- Protenix G+覆盖确有缺口：只在独立Train24选配方的Confirm96-B有现成完整结果；Train96/384主方向配置和Length48都没有同配置G+。旧Generic与旧G+不能跨配方拼接补齐。
- 完整主比较表有空格应保持空格，这份报告不因此自动新开训练或新增benchmark。
