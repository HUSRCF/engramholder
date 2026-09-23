# OpenFoldInteractionFresh96 v1 — 固定模型的新目标验证

2026-09-23；用户明确授权立即准备并提交。零训练，24个旧Train96/Full/1536 adapter＋Query，共25×96＝2400次正式预测。既有A/B不取消、不改科学配方。

## 锁定科学设置

模型身份以model_lock.json为准，24个checkpoint由原训练锁、1536步、冻结权重与文件SHA核验。F3/FR9/G3/GR9；种子20260923/24/25，旋转20261001/02/03。固定ESM2-35M layer12/480维、AF2 model_3_ptm、第一主OPM接口、48 Evoformer blocks、四次trunk、seed20260921、FP32、TF32关闭、query-only/无模板/无homolog、无early stop、每系统一份预测。保留完整现场anchor与原生下游。禁止新训练与按结果选择checkpoint。

## 新目标

四层128–191/192–255/256–319/320–384各24，一PDB一目标。沿原Confirm96-B资格：pre2021-09-30、monomer_clean、单模型X-ray≤2.5Å、标准20AA、实体全序列映射、参考CA覆盖≥90%。不读父项目保留测试清单。按SHA256('openfold-interaction-fresh96-v1|'+target_id)排序。

排除全部可审计历史训练、开发、评分与工程/筛查开发目标的PDB/ID/序列；未实际使用的候选目录与被筛掉的候选记录不自动算已评分。记录所有扫描根及未解析文件。基础模型预训练与完整teacher历史未知，不宣称家族隔离。保持原teacher序列排除库。

原全局比对：match2/mismatch−1/open−8/extend−1，取first optimal alignment；paired=identities+mismatches（gap不入identity分母），paired≥50、identities/paired≥0.30、paired/coverage_denominator≥0.70；query/面板比较用shorter，其他teacher MSA行用candidate长度。新query重复项优先shorter规则。不改用BLAST。每层最多2048候选，选择计算10800秒上限，不足96停止且不放松。

先锁模型和本协议，资格筛查后锁96目标与参考mask。全长输入不裁剪、目标不替换。推理manifest只含ID、sequence、sequence hash、length、length_bin，不含参考坐标或mask。

## 工程门槛

旧Confirm96-B前两条3gxb_A、6a89_A（清单顺序，不看效果）；全部24适配器加Query回放，共50工程预测，另计，不是正式2400。坐标相对L2≤1e-3且RMS≤0.02Å，沿旧数值门槛的量级，全部50统一执行；4次trunk、全长、finite、mask、冻结权重核验。拆分评分用旧目标复算与历史评分一致。门槛在新目标评分之前，不据新分数调整。

旧runner现场评分必须移至独立CPU程序。正式所有25×96尝试终止、锁/覆盖核验后才能评分。可恢复产物必须同时匹配科学锁、checkpoint、目标序列与文件hash；原子写入。每目标最多2次120秒同配置重试，保留失败日志。不裁短、换目标或单臂改采样。系统性导入/权重/锁失败终止任务，不把整组缺失强行记零。终局个体失败三指标记零、保留分母并单列覆盖。

## 唯一主终点

每目标先按种子与旋转配对：Psi_i=mean_s[(F_s−mean_r FR_sr)−(G_s−mean_r GR_sr)]，再96链等权平均。主指标固定reference pair-lDDT（15Å；阈值0.5/1/2/4Å；固定参考对；缺预测计零）。20000次整目标配对bootstrap、seed20260921，普通目标重抽与旧主分析相同；不把种子/旋转当独立蛋白。

报告四格与Query绝对均值、两行方向差、Psi、Native−G及相对Query、3×3与边际、逐目标分布、失败覆盖。residue CA-lDDT、固定对应TM-score及长度层是未校正次要指标。新面板单独报告，不与旧面板合并为唯一主区间；不因不显著追加目标、训练或换模型。

## 部署

HPC3 acd_u、最多4预测worker；保留现有A/B。CPU选择可在DiamondHill，GPU推理只在HPC3。先模型/源代码核验和旧目标回放、ESM特征校验，再放正式预测；score使用afterok/完整终局覆盖双门槛。全部准备及提交有哈希/日志。正式数据选择不依赖ESMC新结果。
