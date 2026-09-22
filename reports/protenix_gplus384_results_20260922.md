# Protenix Train384 / 1536：匹配 G+ 完成结果

2026-09-22 13:01:36 HKT完成；16:18 HKT收集并重新核验。三个训练种子20260923/24/25均完成1536步，新增4608次更新、432次正式预测（Confirm96-B 288、Length48 144），零失败、无HIP重试。四次工程预测另计。

**在这套匹配的Protenix配方下，Native同时优于旋转Factor和显式因子访问的G+；两个面板的Native−G+区间均为正。** 这补齐了原先Train384和Length48的G+缺格，不改变OpenFold中G+均值更高、AtlasFold未建立原生优势的边界。

## 完整同序列输入比较

主指标为Cα pair-lDDT；Native/G+各平均三个种子，Rotated平均三个种子×三个旋转。Query不含适配器额外ESM2特征，因此相对Query的全部改善不能归因于方向。

| 面板 | Query-only | Native Full | Rotated Full | G+ | Native−G+ [95% CI] | 正向目标 |
|---|---:|---:|---:|---:|---|---:|
| Confirm96-B（主要） | 0.37931 | 0.62572 | 0.58621 | 0.54942 | +0.07630 [+0.05677, +0.09795] | 78/96 |
| Length48（次要） | 0.32122 | 0.52919 | 0.48667 | 0.44781 | +0.08138 [+0.05968, +0.10571] | 43/48 |

G+相对Query的描述性主指标均值改善分别为+0.17011、+0.12659；G+也有适配收益，不能把Native优势解释成G+完全失效。Rotated均值亦高于G+，但这里不新增事后确认性检验。

## 种子、目标分布与补充指标

| 面板 | Native−G+：seed23 | seed24 | seed25 | 目标差值中位数 | 目标差值P10 / P90 |
|---|---:|---:|---:|---:|---|
| confirm96 | +0.06922 | +0.06098 | +0.09871 | +0.04631 | -0.01456 / +0.21538 |
| length48 | +0.07908 | +0.06925 | +0.09581 | +0.05966 | +0.00598 / +0.16811 |

每个配对种子均正，目标平均差值分别78/96、43/48为正；中位数也为正，优势并非只靠极少数目标。Confirm96仍有18条、Length48有5条目标反向；均值高于中位数，较大改善目标也有贡献。未删除离群值或选择有利子集。

| 面板 | 指标 | Native−G+ | 95%条件区间 | 正向目标 |
|---|---|---:|---|---:|
| confirm96 | 逐残基Cα-lDDT | +0.07184 | [+0.05321, +0.09251] | 77/96 |
| confirm96 | 固定对应TM-score | +0.08256 | [+0.05889, +0.10803] | 74/96 |
| length48 | 逐残基Cα-lDDT | +0.07830 | [+0.05731, +0.10164] | 43/48 |
| length48 | 固定对应TM-score | +0.09515 | [+0.06814, +0.12610] | 44/48 |

三指标在三个种子上的均值差均正；指标来自同一批结构，不能算三份独立复制。先在目标内平均种子，再以目标为单位bootstrap20,000次(seed20260926)，不是把3×N当独立蛋白。区间条件于当前拟合模型。六个对比的逐目标数组、均值和CI从原始评分独立复算，误差<1e-12。

## 配对、输入与执行审计

- 复用Train384原缓存，原Mini-default、冻结ESM2-35M、query-only特征和pair stack；AdamW LR5e-5、1536更新。全部1536步样本顺序及实际记录的noise_level与历史Native逐项一致。原训练链152–383残基，Length48完整392–750残基。
- G+为既有InterfaceHead(generic_plus)，显式接收query_a_i、query_b_j及相同ESM2表征，末层零初始化。378144可训练参数，Native373824；不是严格参数量相等。旧Generic不充作G+，旧Train24独立选LR结果不混入本表。
- 三种子初始encoder hash、主干/PLM/cache/common、runner_config一致；第一步loss和噪声与Native精确相同。第二步encoder梯度正常，最终全部frozen_parameters_unchanged=true。完整顺序/噪声复核通过。
- 推理c4/s5、seed101、单sample0，模板/MSA关闭，完整输入输出核验；原参考mask、固定对应评分、失败计零规则保持。所有432预测完成后才评分；CIF、checkpoint、面板、封存源码及评分输入hash全部复核通过。
- 三组训练循环累计约4.09 GPU小时（单组约1.36小时，不含初始化等）；从11:26:53正式释放至13:01:36完成评分约1小时35分。工程前缀和预检查另外计时。
- DiamondHill原ROCm环境运行。Confirm96与旧Native在同后端；Length48历史Native/Rotated来自H100，新G+来自MI250。最长750残基Native回放相对坐标L2=0.00032018，低于预定1e-3工程阈值；短链精确复现。这是有限工程一致性证据，**不是全部48目标或Native−G+差值的跨后端误差上界**。

## 可以增加的结论与仍保留的边界

1. Protenix Train384/1536中，在相同新增表征、query factors访问和公共训练配方下，Native完整Factor优于这一个已定义的G+头；其优势覆盖本次两个已观察面板。原先“Length48缺G+，不能比较通用头”的证据缺口已补齐。
2. 这是公共固定学习率/预算下的方法比较，不是G+充分调优后的性能上限，也不证明优于所有通用适配器。自由函数集合更大不保证这一优化配方更好；当前结果不能唯一归因于表达约束或原生方向。
3. Native−G+与Native−Rotated回答不同问题。**本次没有新增Protenix旋转G+，不能在Protenix计算头类型×旋转的直接交互Ψ**；该交互证据仍来自OpenFold Train96。
4. 两面板已被观察，本次后续研究不重新称为盲确认。Length48的原Native−Rotated仍保留当时预锁定身份；新增G+比较属于后来提出的对照。累计暴露、CATH家族隔离和基础模型预训练隔离的限制不变。
5. 跨底座结论继续有条件：Protenix本配方支持Native优于G+；OpenFold Train384中G+均值更高；AtlasFold未建立方向优势。不能恢复“Native普遍优于Generic”的表述，也不新增训练来修正这些边界。

## 可复核产物

- [汇总JSON](../evidence/protenix_gplus384_summary.json)、[432条新三指标记录](../evidence/protenix_gplus384_records.json)。
- [执行锁](../evidence/protenix_gplus384_execution_lock.json)、[收集完整性审计](../evidence/protenix_gplus384_collection_audit.json)、[独立统计复算](../evidence/protenix_gplus384_verification.json)。
- [完整本地归档](/home/husrcf/Code/onestepfold/engramfold/reports/protenix_gplus384_20260922)；[预锁定协议](/home/husrcf/Code/onestepfold/engramfold/docs/protenix_gplus384_v1.md)。

本次更新MD证据与独立结果快照；未修改论文数值生成锁、生成表格或正文。论文集成应通过既有生成脚本显式接纳这些新来源，不能静默覆盖旧证据锁。
