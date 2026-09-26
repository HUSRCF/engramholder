# OpenFold 注入时序重训练：完整72组执行协议

2026-09-26，用户最新明确恢复三个旋转、三个种子的72组方案，取代此前36组总规模。旧36组工程准备和失败记录保留，新的正式锁独立建立。此前没有正式训练。

## 矩阵与第一波

Train96 / ESM2-35M layer12 480维 / Full / 1536更新，三时序First、All、Last。每时序：Factor Native3、Factor Rotated9、G+ Native3、G+ Rotated9，共24；总72，110592更新。种子20260923/24/25，原稠密旋转20261001/02/03。Native不按旋转重复训练。

第一波36：三时序×(Factor Native、Factor R1、G+ Native、G+ R1)×三种子，全部HPC3/acd_u/H100，最多16worker。先处理First/All长任务，随后Last；这是调度次序，不是改变科学比较。worker可顺序执行多个独立拟合，以避开QOS总提交数量限制。正式36个fit不是36张卡，也不是36个种子。

第二波36：三时序×两个head×R2/R3×三种子。当前只准备、不自动启动，待第一波完成再按用户要求考虑HPC3 24/HPC2 12。HPC2仅i64m1tga800ue。跨后端正式分配仍需H100/A800共同起点工程回放；不能把GPU型号差异默认为零。完整72组主分析等第二波齐全；第一波完成只报告执行覆盖，不据结果调整后半批。

## 共同训练规则

冻结完整AF2 model_3_ptm，eval、无dropout、FP32、禁TF32、chunk64、四次完整trunk，无模板/同源MSA/提前停止。仅首Evoformer OPM现场残差门控，旋转仅应用一次；其余48-block主干结构不改。所有时序重新初始化、最终pass同一原生FAPE+distogram+torsion损失。跨pass原生MSA/pair图保留，无中间损失/人工梯度旁路；坐标距离分箱保持原生不可微。checkpoint不可变pass_id在反传重算中恢复。Last早轮无头参与可自然无图。All不能复用旧末轮求导模型。

AdamW原betas/eps，weight_decay=.01，全局梯度裁剪1，固定LR、无scheduler。所有时序同一head/seed的初始化相同，训练样本顺序使用原schedule(seed,96,steps)，每步随机seed为run.seed+zero_based_step。同样本不跳过，零梯度不改None、不跳optimizer.step；每个预期参数梯度连接、梯度/参数/loss有限性、冻结主干必须通过。单张量零梯度只作诊断。

原子checkpoint保存完整writer、optimizer、CPU/CUDA/NumPy/Python RNG、run、数据顺序hash、step、lock、固定LR。每96步保存latest，保留0/384/768/1536节点；从实际断点恢复，未入断点的日志尾部另存，不冒充逐位长轨迹恢复。失败日志不覆盖。NaN/Inf、冻结参数变化、合同不一致立即停止，不自动换配方。工程中已修复CPU标签与GPU aatype混用；结构labels只在训练/评分中读。

## 有限公共校准（不计入72个正式fit）

沿此前明确拟定规则封存：LR 5e-5与1e-4，三个策略×四臂（原列表首R）×开发seed20260922，每条192更新；共24个短校准、4608更新和192个Dev预测。以全部12策略/臂的Dev8绝对pair-lDDT均值选择一个共同LR，精确并列取小LR。不按Psi/Confirm96/第一阶段结果选择。其余R复用此公共配方，不再校准。

每条校准在第一步实际断点上恢复并重放第二步，下一前向loss相对误差≤1e-5；记录权重差异，不将其当长期轨迹等价。First在实际H100 final-loss反传中验首轮pair梯度。此检查使用正式runner，补齐A800工程通过不等于H100执行器已验证的边界。所有24条校准/Dev评分通过后自动封存selection，才释放第一波36组。任何失败阻断释放，不用确认集调参。

## 推理与统计

第一波36×Confirm96-B96=3456完整预测，Query96另做/核验。其余同样，72适配模型总6912。所有正式模型使用各自训练时序，四pass，推理seed20260921，全长CIF及原始坐标保留。推理阶段通过读文件guard禁止读参考结构/MSA；标签不决定门控、幅度或失败选择。先完成全矩阵终态，再统一独立评分。失败目标保留分母、主分数零、完整工程原因；不裁剪/换目标/加步数。

完整矩阵主要量D_write=Psi_all−Psi_first，Psi=(Factor Native−三R均值)−(G+ Native−三R均值)，先每目标聚合配对三种子/三旋转，后目标均值。First/Last、All/Last预设次要；所有四格绝对分数、相对Query、种子/R边际及目标分布保留。主Cα pair-lDDT、次要逐残基Cα-lDDT/TM-score，固定完整对应，20000次目标配对bootstrap，seed20260926。同一抽样用于直接差，区间条件于拟合实例。Confirm96-B已观察，不称新盲确认；不把多个头/旋转当独立蛋白。

调度/代码与全部数据身份写入execution_lock.json。第一阶段结果的正负不控制本阶段放行；截至本锁未据第一阶段统计值改变配置。无新模型/补偿器/anchor干预，不改论文正文。
