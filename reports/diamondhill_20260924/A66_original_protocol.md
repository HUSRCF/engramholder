# DiamondHill A：Protenix → AtlasFold 顺序执行

用户授权2026-09-23：两底座A组今晚运行，先顺序推进，增加watchdog。
每底座33新fits：E-final G+R9；C-final Factor3/FR9/G+3/GR9。
Protenix Train384/1536 lr5e-5 wd1e-4；Atlas Train96/1536 lr1e-4 wd.01；
clip1，AdamW默认betas/.eps，旧种子20260923/24/25与R20261001/2/3；旧样本顺序。
保留原生loss、原生PLM、完整序列与推理配方。Atlas保留AtlasLM3B；新增ESMC不是替换原生LM。
输入480→1152仅替换embedding，独立种子+1000000初始化，其他共享参数与旧初始化精确相同。
G+旋转只作用于自由末层产生的完整残差；不旋转query基线；Full二次项保留。

工程：CPU三R吸收/宽度初始化测试，全536特征hash/序列/形状；每底座两固定训练链
（首条和最长）×E/C×F/FR/G/GR两步真实loss、零初始化回放、冻结、第二步encoder梯度。
再做750aa完整输入推理与旧checkpoint短链回放；先训练工程后推理工程，所有gate通过再正式。
正式每32步保存writer/optimizer；任务内flock；进程存在不能仅因heartbeat超时重启。
HIP最多2次重试、间隔120秒，失败日志保留。非HIP自动停止对应科学阶段，不能无限重试。
watchdog负责恢复已退出controller；不能杀仍活着的任务；子任务锁防重入。
Protenix整组训练/预测完成后运行Atlas组；底座内最多8逻辑GPU，先单fit短期确认进展再放开。
HPC3 OpenFold A/B不动，DiamondHill4物理MI250/8逻辑GCD。

评测固定Confirm96-B+Length48；33×144=4752/底座，9504总预测。
全部正式尝试结束才统一评分，失败计零、无裁剪换目标、不得以分数改矩阵/超参。
复用旧15适配器与query需hash/配方匹配；G+不能用旧Generic替代。
主要C-final F−FR；其他G−GR、Psi=(F−FR)−(G−GR)、PLM交互和N−G。
seed/R/噪声先在目标内汇总；条件于固定fits的目标bootstrap20000，seed20260926；
Length48按既有三长度层分层重抽；多指标/次要比较明确后续分析，不算新盲确认。
两个底座训练集不同，不作纯底座因果解释。ESMC输入层多86016参数，不称等参数纯容量效应。
B不在此授权矩阵内自动启动；不重开局部微分/有限步共享迁移。
