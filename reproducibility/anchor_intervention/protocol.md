# E3：残差 query-only anchor 干预执行协议

2026-09-24。作者已授权系列矩阵，现明确推进E3的独立工程验证及通过后的原定9组。
E1的X预测失败与E2的有限范围阳性均保留；本实验不恢复预测规则，不自动启动E4。
不改论文正文，不改E1/E2任何历史锁或complete文件。

## 唯一干预与配方

继承E1 OpenFold AF2 model_3_ptm、Train96、ESM2-35M layer12/480、Full、1536更新。
Native及原prediction_lock的低X r20270107、高X r20270104，各种子20260923/24/25。
不带E2补偿器C，不热启动。theta/AdamW/lr1e-4/wd.01/betas(.9,.999)/eps1e-8/
全局clip1、逐步seed+step、历史数据顺序、现场OPM baseline和48个Evoformer均沿旧路径。
仅用各轮对应的query-only参考(aQ_t,bQ_t)替代残差构造中的现场(a_t,b_t)；原始
baseline+delta的加法与现场recycle保持。四个anchor不混用，第4轮checkpoint反向重算复用第4anchor。
零增量应精确回放原query函数；不更改mask、归一化、decoder、loss或推理四轮。

## 参考缓存的验收

代码核对旧模型在eval运行、dropout关闭、特征由固定query产生；不靠此推断确定性。
训练与评测参考分开（最终一轮grad enabled与全no_grad可能选择不同inplace路径）。
每一Train96与Confirm96目标分别完整运行两次参考，seed=20260923与20260925；
要求4轮a/b/mask和最终坐标逐位相同，两次参考自身都不消耗CPU/CUDA RNG；fork_rng
恢复调用者随机状态。每条缓存绑定特征张量hash、全长序列hash、输入清单、源代码、
骨干hash、数值路径、Torch版本、参考文件hash，跨种子和臂共享仅通过认证的缓存。
这使正常参考成本为192条×2=384次完整forward，工程额外单计；不是随意一次缓存。
若任何参考不满足，不自动改成近似缓存、不放宽容差，停止正式队列并记录原因。
缓存建立不读取目标结构标签。训练标签只用于原监督；评测标签只在全部864尝试结束后评分。
训练侧断点384/768/1536的固定已出现目标做缓存重放抽查，避免模型状态改变导致静默缓存错误。

## 工程与资源

固定第一条Train96和最长Train96为工程链，不按结构结果选择。
检查zero native/R回放、原生loss真实反传、4forward+1backward checkpoint重放、
参考mask/eps一致、非零teacher下实际现场baseline不被替换、参考/现场anchor差异、
参数及优化器原子保存恢复、全部冻结参数hash、dropout状态、CPU/CUDA RNG。
训练零梯度允许但记录；NaN/Inf、断图、冻结状态变化立即停止。
工程先在HPC3/acd_u H100两任务运行；全部通过再参考缓存16分片、正式9任务并发9(上限16)。
不得混用DiamondHill后端。预计训练沿E1约1.7–2小时/模型，加缓存与评分另计；ETA以实测为准。
保存首次失败和恢复尝试；恢复相同原子断点/optimizer/RNG/order/lr，不更换种子或跳step。
缺失训练实例不评分成功子集。预测有限值/全长失败留分母，原评测失败计零；不截短换目标。
资源/服务瞬时失败仅允许同设置一次记录性重试，算法/断言失败须单独诊断。

## 主终点和解释

Confirm96-B已观察面板；9×96=864新预测、13824训练更新；现场基线复用E1已验收预测。
先在目标内平均三种子与两个旋转，主读数
T_A=mean_r[(N_Q-R_Q)-(N_live-R_live)]。
同时必报B_R=S_R(live)-S_R(Q)及N_Q-N_live、四个绝对值、相对query和逐旋转结果。
T_A与B_R都为正且相应区间支持才支持“现场anchor帮助旋转臂减轻代价”；仅Native提高或
两个Q系统都失去适配收益不作为完整机制成功。方向相反按反证报告。
主指标Cα pair-lDDT，补充逐残基Cα-lDDT和固定全长TM-score；不替换主指标。
目标配对bootstrap20000、seed20260924、95%百分位，三个训练种子边际及全逐目标数组保留。
区间条件于现有模型和两个原选旋转，不把seed、pair或864预测当独立蛋白。
参考评分沿E1已修正的same-input-dtype一致性检查，原1e-6门槛不变；不重写主分数。
训练及评测统一保存每轮完整残差范数。只干预残差anchor历史依赖，不是关闭全部反馈，
也不证明冻结骨干重新学习、函数类不可能补偿或E1预测成功。
