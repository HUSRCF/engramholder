# DiamondHill A66：封存协议的统一评分实现

2026-09-24。只评分已经完成的固定矩阵，不新增训练、预测、目标或终点。
原协议为 `source/protocol.md`（SHA 0c3a3290a9930bc82c2c78dfcc245ec5213562144c36dac3c3ac0bd8f7edc64e）。

- 必须先有完整 A66 completion、Protenix33/Atlas33完整评测与9504终局记录；成功CIF逐个SHA核验，失败保留计零。
- 原科学源码/参考结构/两面板/原子checkpoint及适用的执行锁和恢复修订全部记录。
- 每底座复用旧ESM2 F3/FR9/G3及query，加入E-GR9/C-F3/FR9/G3/GR9，形成48个适配器＋query。
- 三指标为固定参考mask Cα pair-lDDT、逐残基Cα-lDDT、固定对应且按完整序列长度归一化TM-score；重用原AlphaFold lDDT源码和TMscore二进制。
- 每底座主要读数为Confirm96-B的C-final Factor−Rotated。G−GR、Ψ、Native−G+、PLM直接对比及Length48/补充指标均保留其后续分析身份。
- 先在目标内平均三训练种子与三旋转，目标bootstrap 20,000次，seed20260926。Length48按原三长度层各16条分层重抽。完整逐目标、种子、旋转、3×3单元保留。
- 原DH协议没有规定OpenFold A的Holm次要family，不事后把其照搬成新预设规则；本报告区间均明确未作多重校正，不将两底座之一阳性宣传为通用确认。
- Atlas ESMC seed20260925完整8臂为H100，其余在MI250；核对归属、HPC执行锁、回执和仅路径重定位。报告按seed后端列明；不把后端与seed重合的分组当独立后端复现。此seed的ESMC−ESM2也混有后端变化。
- Protenix历史Length48 Factor与query的H100来源、旧G+与当前新臂MI250来源同样保留，不能称所有PLM比较均同后端。
- Atlas饱和/恢复事件归档，不删除发生事件的臂或目标，不更改权重或checkpoint。统一检查策略与HPC API修订属于既有执行来源记录。

本文件是原封存协议的评分实现说明，写于预测完成后、统一新分数计算前，不冒充重新预注册。所有旧分数保持字节来源可追踪；未改写论文。
