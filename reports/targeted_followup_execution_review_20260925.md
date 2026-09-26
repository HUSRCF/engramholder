# 同起点通道校准：提案复核与执行前补充

> 执行更新（2026-09-25 21:24 HKT）：用户已授权 smoke 后自动正式运行；A/B 真实 GPU smoke 已验收，两条开发校准均已完成，共同倍率均为1.0。OpenFold正式27组已自动提交HPC3/acd_u（652077，最多16并发），Protenix正式6组已在DiamondHill运行；无再次等待批准。下文“尚未提交”是提案审查时的历史状态；实时任务与工程修订见[执行记录](../../onestepfold/engramfold/reports/posttraining_channel_calibration_20260925/README.md)。

2026-09-25。对应[原提案](targeted_followup_options_20260925.md)。本次已做源码检查、远端CPU断点读取和候选矩阵整理；**没有提交GPU工程、校准、正式训练或评测任务**。本文件是待封存的实施建议，不替代原实验锁。没有修改稿件或既有实验结果。

## 1. 建议保留 A+B，问题与原 E2 分开

推荐这两个问题：A检验已训练头固定后，共享通道校准是否优先帮助旋转臂；B检验在原生Protenix中，后续预算用于通道校准是否优于继续更新原头。

原E2与重训练都是从零联合学习头和C，不能解释成“在已训练头上续训”。本方案是在第1536步分叉的新干预；其结果也不能追溯性地把原E2差异唯一归因于某个机制。

| 线 | 起点 | 新分支 | 正式更新 | 评测安排 |
|---|---|---|---:|---|
| A / OpenFold | Train96 / ESM2；I、r20270107、r20270104 × 3旧种子，无C，第1536步 | C-only、head-only、joint各9 | 27×512=13,824 | 既有Confirm96-B；2,592次新增预测 |
| B / Protenix | Train384 / ESMC；Native × 3旧种子，无C，第1536步 | C-only、head-only各3 | 6×512=3,072 | 已观察Fresh192；1,152次新增预测 |
| 可选迁移 | A的C-only；同方向、不同种子固定循环 | 不训练 | 0 | 9×96=864次新增预测 |

默认A+B共33个短任务、16,896更新、3,744次新增正式预测；旧00基线须先回放核对再复用。可选迁移单列，不据A结果决定是否追做。B固定为既有Fresh192的后续比较，不建立新面板，不再称其首次盲确认。B是否执行不依赖A是否为正。

## 2. 断点审计已通过，仍有一个恢复边界

[资产审计](../../onestepfold/engramfold/reports/posttraining_channel_calibration_20260925/asset_audit.json)保存全部12个父起点路径、SHA及实际字段；CPU读取未占用GPU。

- OpenFold三个Native来自原Train96正式模型；六个Rotated来自原E1的R7/R4正式模型，均为无C，不使用原E2或重训练的带C模型作为父模型。九份checkpoint均与原收据SHA相符，step=1536，保存15个参数的AdamW状态，状态步数均1536。
- Protenix三个`task_1536.pt`只用于模型身份：它们**没有optimizer**。三个同目录`resume.pt`都有完整AdamW状态，step=1536，头参数与已评分的task文件逐项完全相同，应从resume读取续训状态。
- OpenFold旧Native checkpoint和Protenix resume未保存完整RNG；OpenFold六个E1旋转断点有Torch/CUDA RNG。不能声称所有分支逐位延续未中断的历史训练轨迹。

拟定恢复规则：10与11恢复同一父头及其AdamW一、二阶矩、参数step；01冻结同一父头，仅为新C创建零动量。11的新C也为零动量、step=0，不继承任何E2补偿器。优化器必须按原参数名称/顺序和shape验收，不把C插进原参数列表后盲目load_state_dict。

对所有分支统一确定新的配对执行序列：按原schedule生成2048步，验证前1536步与父运行的target顺序一致，再使用第1537–2048步。不得重新调用512步schedule并从第一条重新开始。OpenFold保留原零起算的`seed+step_index`，Protenix保留原一起算的`seed+step`。构造器/额外诊断不消耗正式训练随机数。此为“同模型、同优化器状态、同后续顺序的配对分叉”，不宣称FP32轨迹逐位复原。新分支从step0起完整保存Python/NumPy/Torch/CUDA RNG、全局数据位置和学习率，并测试本轮断点恢复。

## 3. C的数学对象与插入位置必须保持一致

继续使用现有`SharedOrthogonalChannels`：128通道中的均值方向固定，127维正交补空间上学习反对称指数映射，8,001个坐标参数。设u为归一化全1向量、B为其正交补基，

\[
C=I+B[\exp(A-A^\top)-I]B^\top.
\]

C从严格恒等初始化；理想算术中C正交且C1=1。沿用该较窄变换族，不临时改成自由128×128矩阵或带缩放的映射。

OpenFold已有`ProspectiveOPMAdapter`先由父类形成含R的完整残差，再执行channels，故列向量记号为

\[
U=U_0+C R\Delta U_\theta.
\]

不是R C，也不旋转U0。每个checkpoint用一个全蛋白共享、四轮共享的C；保留原native mask、分母、四轮live anchors及最后一轮反传。

Protenix `NativeGeometryHead.forward`返回的是**baseline+residual**。不能直接对其返回值乘C，也不应先计算大baseline+residual再减baseline取微小残差。新接入应在已有解析Full残差（一次项＋二次项、D=508.5及原decoder均不变）完成后、加回query_update前施加C。00/10保留原计算路径，01/11的C=I必须通过完整回放。

静态Protenix且θ冻结时，同一输入的残差内容固定；C改变其通道方向，理想算术中保持每个pair的范数和pair向量间内积，不保持baseline+residual的范数。OpenFold的现场anchors可能随适配状态变化；冻结θ并不固定全部残差内容，不能从A声称隔离了纯方向或所有反馈。

冻结θ用`requires_grad_(False)`及优化器排除；**不得把整个OpenFold残差路径包进no_grad或额外detach live anchors**。冻结参数和切断输入求导是两回事；沿用原recycle截断边界即可。

## 4. 优化器和校准预算：明确提出，尚未选择

| 参数组 | OpenFold历史值 | Protenix历史值 | 续训规则 |
|---|---:|---:|---|
| θ学习率 | 1e-4 | 5e-5 | 10与11一致，继承历史moment |
| θ weight decay | 0.01 | 1e-4 | 保持底座各自原值 |
| 新C学习率基值 | 1e-3（原E2） | 1e-3（待工程验证） | 01与11一致，moment从零 |
| C weight decay | 0 | 0 | 不衰减C坐标 |

AdamW betas=(0.9,0.999)、eps=1e-8；更新原精度，全局L2裁剪1。01仅裁剪活动C，10仅活动θ，11联合裁剪活动θ+C；记录各组裁剪前范数和共同缩放系数。joint和head-only的实际θ步长可能因全局裁剪而不同，结论属于完整训练配方，不能归成单独的函数类效应。

建议最多两个公共倍率q∈{0.3,1}，同时作用于该底座θ/C基准学习率。使用首个父种子20260923的全部方向与分支，每条64步，完整Dev8评分：A为3方向×3分支×2倍率=18条开发运行；B为1方向×2分支×2倍率=4条，共1,408更新、176次Dev预测。开发副本弃用，正式一律从父1536步重新分叉；不能warm start校准结果。

每底座按所有预设分支、方向、Dev8的**等权绝对结构分数**选同一个q，不按T_frozen、C优势或旋转差值选；均分差≤1e-6时取较小q。不能只为C-only补额外学习率。完整校准表保留。这里首个父模型参与了开发，正式结果应条件于该开发选择，不能把三个种子当成完全未经选择的新训练重复。

工程可预留每个新条件16次实际更新（8连续，对照4+恢复4）：A首种子九条件144更新；B首种子两条件32更新，共176。加正式和上述校准为18,480更新，约12.03个1536步任务的**更新量**；并非仍只有11个完整训练的总成本。额外基线回放、推理和排队分别计数。

若只是Dev没有增益，不应自动解释成实现错误；若非有限、梯度断连、冻结参数变化、C不正交或恢复不一致，则属于工程失败。候选范围内不追加搜索。正式终点固定512新步，无按评测结果追加步数或替换实例。

## 5. 终点与解释：需要直接比较，不能比较显著性标签

A唯一主要量：

\[
T_{\rm frozen}=\operatorname{mean}_R(S_{01,R}-S_{00,R})-(S_{01,N}-S_{00,N}).
\]

同时完整报告旋转臂自身收益B_R^01及Native自身收益。只有T_frozen和旋转臂自身收益都得到正向支持，才谈有用的方向选择性缓解。

A固定次要量：01−10、11−10、11−01，分别在Native、两个R及R平均中报告。既然已有四格，也记录

\[
J_d=S_{11,d}-S_{10,d}-S_{01,d}+S_{00,d}
\]

作为共同更新的非加性描述，不增加训练。“01未显著、11显著”不能证明联合更新必要；即使11直接优于两个单独分支，也只能说在本预算/配方下更好，不能证明不存在其他单独优化解。

B唯一主要量：Native的S01−S10；另报告S01−S00。若前者正而后者不改善，只能说相对更少伤害，不能推荐正向校准策略。

A使用原Confirm96-B目标bootstrap；B继承Fresh192四个长度层内抽样；均20,000次，建议固定seed20260925。先在同一目标内聚合种子/旋转，再以整条目标为单位，保留全部目标及失败政策、种子和方向边际。Cα pair-lDDT为主要指标，逐残基lDDT/TM-score次要。A和B回答不同问题，分开报告其95%条件区间；不把“任一为正”组合成一个经过多重校正的共同成功声明。

两条评测面板均已被观察，本轮是预先规定终点的后续研究，不是新盲确认。全部正式尝试结束后统一评分，不用中途分数选倍率、checkpoint、停止或增加任务。

## 6. 可选跨种子转C

同方向固定20260923→20260924→20260925→20260923；将供体01分支学到的C放到接收者**原1536步无C头**，接收者θ不变，不转optimizer、不再训练。不选择供体或逆矩阵；I方向同样执行。对照接收者自己的C和I。若启用，在看主结果前列入清单，额外864预测，结果仅作次要迁移边界。

## 7. 工程验收与部署顺序

1. 本轮已完成父权重/optimizer CPU审计。下一步在隔离输出根实现两种注入路径和显式参数组，复用原冻结环境、特征和损失。
2. H100验证OpenFold九个工程条件，MI250验证Protenix两个工程条件；检查C=I的旧函数回放、θ冻结哈希、C梯度路径、正交/均值误差、完整序列及恢复。保留偶发零梯度，不跳step、不将零改None。诊断运算不改变训练RNG。
3. 每底座完成固定校准，写父SHA、数据顺序、学习率选择、最终终点及重试政策的执行锁，才释放正式阵列；A/B可独立通过各自工程门槛后并行，不按对方结果放行。
4. A优先HPC3/acd_u/H100，最多16个单卡worker；B优先DiamondHill/MI250，6个单卡任务。可用卡数提交时核对，不因本报告假设当前独占资源。
5. 按三类分支分别实测step秒数，再估ETA；原E2每个1536步约1h49–1h58，仅作OpenFold预算参照。512步训练的粗线性值约36–39分/任务，尚未包含新起点/冻结分支差异、工程、校准、完整评测、排队和网络恢复；不视为承诺。
6. 只提交已锁矩阵，全部结果含阴性完整保存。本轮不重跑E2以挑选更好结果，不启动E4、不增加底座或优化器矩阵，不改论文正文。

## 8. 文献与源码依据

GitNexus技能已读取，但当前会话未提供对应查询服务/索引资源；本次实际依据直接源码与远端CPU断点，不声称完成了图索引验证。

- C参数化和施加顺序：[prospective_orientation.py](../../onestepfold/engramfold/src/engramfold/models/prospective_orientation.py)。
- live anchor及原残差图：[live_opm.py](../../onestepfold/engramfold/src/engramfold/models/live_opm.py)。
- Protenix完整解析残差：[native_geometry.py](../../onestepfold/engramfold/src/engramfold/models/native_geometry.py)。
- Adam论文研究的是**参数空间**旋转的敏感性；它支持区分函数类、优化和坐标选择，但不能直接证明这里的输出残差旋转由Adam造成。[Understanding Adam Requires Better Rotation Dependent Assumptions](https://arxiv.org/abs/2410.19964)。
- LoRA-Pro讨论低秩适配的优化差异，提供背景，不能替代本研究的实际对照。[LoRA-Pro: Are Low-Rank Adapters Properly Optimized?](https://arxiv.org/abs/2407.18242)。

完整候选33任务与状态见[执行准备目录](../../onestepfold/engramfold/reports/posttraining_channel_calibration_20260925/README.md)。目前状态为设计及资产就绪，尚非GPU验收通过或正式运行。
