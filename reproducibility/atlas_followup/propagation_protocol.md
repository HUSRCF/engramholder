# Atlas 同位置传播定位：2026-09-26

用户批准先定位传播；状态条件化 writer、改变注入位置仍是候选，本轮不训练。

## 已完成的先验记录

Native 通道校准六个512步分支、960次预测、0失败；远端完成收据及汇总/逐目标记录SHA256匹配。
只学C−普通续训 −0.00000513，95%CI [−0.00069698,+0.00074909]；只学C−Query亦未建立收益。
此处仅核对既有汇总，不重评CIF。Native-only，不是旋转交互证据。

## 固定范围

- 沿用旧传播诊断的两链 **7hln_A、9qdy_A**，不是新增确认面板，不根据本轮输出挑选。
- ESM2/Train96/1536 Native三种子20260923/24/25；每个对应已完成C_only与head_only512步分支。
- 每链Query+3parent+3C_only+3head_only，共20次观测推理。
- 每链另做一次有观察器Query重复、一次无观察器Query、一次无观察器首种子parent，共6次工程重放。
- 总计 **26次完整推理、零训练**。仍用完整序列，原AtlasLM、Torch后端、BF16 autocast、单样本、seed设置1、请求4 recycle（实际5轮）、MLM .15、20 diffusion steps。
- DiamondHill空闲MI250逻辑GPU6/7，各负责一条链；不占HPC3正式训练资源。限2 GPU小时；HIP失败最多原配置重试一次，保留日志，不改变模型或样本。

## 源码观测位置

每轮同时观察pair与single；首个注入前后只有对应pair读数。

1. `lm_stack.blocks[0]`输入s/z（检查相同随机条件与原生LM输入）；
2. 首个`pairwise_prod_diff`完整输出、`tri_mul_out`输入（真实加法后状态）；
3. LM四个block各自输出s/z；block4即LM stack末端；
4. `proj_{s,z}_lm[0]` LayerNorm输出及完整projection输出；
5. `recycle_{s,z}`输出（仅辅助确认反馈）；
6. `main_stack`输入（投影已与初始化/回收状态融合）与输出；
7. 最终全序列Cα成对距离变化RMS、刚性配准RMSD（相对Query，不是质量）。

源码显示LM stack每轮从随机掩码的AtlasLM重新计算，不读取适配历史；历史状态在后续融合处加入。
本轮不改变该路径。Hook只读取并立即clone，避免后续in-place加法污染快照；不调用额外随机函数。

## 读数与验收

所有差值是 **同目标、同轮、同位置、同随机条件的 Adapted−Query**。
保存全长实残基（剔除输入padding）状态的绝对差RMS、相对Query RMS、Query/Adapted RMS、cosine、改变比例，以及差分的通道常量能量比例。
统计在CPU FP64累加原精度值；不宣称全路径FP64，不把不同层裸范数相除作为衰减率。
Query全状态快照保留；全部臂保留每轮完整统计、最终坐标、CIF及checkpoint/source hash。

两链均要求Query重复状态精确一致；Query与首种子parent各自有/无观察器完整输出精确一致；冻结模型hash不变；完整长度；5轮22个位置记录；禁止推理读取参考结构。
失败保留，不靠放宽精度/换种子通过。

两个目标仅作定位案例，不能据此建立普遍机制。若变化在projection/融合前后减小，只定位本路径信号，不证明唯一原因；若下游变化明显但既有质量未改善，报告此边界。
无结构质量排名、无新显著性主张、无自动启动状态条件化或晚层新训练。
