# ICLR写作分支：结构建议与开篇候选

2026-09-22。用户在分叉中明确要求协助draft paper，并询问Introduction→Results→Discussion→Methods是否合适。本文件据此开始写作讨论；不是已批准的最终稿。仅新增此文件，不修改主线训练、历史报告、README工作状态或根LaTeX模板，不提交/推送。

已读取：AGENTS.md、notes/workflow.md、reports/evidence_review.md、math_contract.md、results_update_20260922.md、reviewer_priorities_20260922.md、protenix_gplus384_status_20260922.md。数值来自这些已有核验报告，本分支没有重新评分CIF或查询远端训练。旧数学笔记中的Atlas“待完成”已被9月22日结果更新取代。匹配Protenix Train384 G+正在训练，无最终分数，不预填。

## 1. 结构决策

建议保留生物论文的问题驱动结果叙事，但将理解实验必需的方法定义前置：

Introduction → Native update construction and orientation controls → Experimental design → Results → Discussion and limitations → Related work → Conclusion。

详细Implementation and reproducibility methods放附录。Related work也可并入Introduction的定位并保留一个短节；不让文献综述阻断问题到方法的推进。

ICLR 2027官方Author Guidelines规定初稿主文≤9页、参考文献与附录另计，且审稿人无义务阅读附录。所查指南未规定IMRaD或具体章节顺序。因此这不是格式禁止，而是论证可读性选择。
来源：https://iclr.cc/Conferences/2027/AuthorGuidelines （2026-09-22查阅）。

不推荐将全部Methods放到Discussion后：读者在看到任何Native−Rotated结果时，就必须知道旋转作用于训练中的通道残差、每组从零残差开始独立配对训练、以及G+的函数集合为何可以吸收旋转。缺少这些定义，最强对照会被误读成对已训练模型的事后破坏。

## 2. 中心问题与主张

研究问题：冻结结构预测器的内部更新接口，在保持因子构造和局部谱性质的情况下，其输出方向是否影响适配；这种影响是否依赖头的参数化与底座设置？

候选中心句：

> Preserving a pretrained interface's update orientation can benefit constrained adaptation of frozen protein structure predictors, but the effect depends on the backbone and adaptation setting and does not imply that native factor heads universally outperform generic adapters.

“can benefit”不是所有模型的保证。“依赖底座”是本研究各配方下观察到的条件性，不能从非随机架构比较唯一归因于架构。

论文类型应定位为一个具体适配构造加受控实证研究。不要按高性能folding替代系统写，也不要把尚未成立的梯度机制当核心理论。

## 3. 九页内的草拟安排

以下是排版预算，非硬分节要求；总计约8.8页，含图表及摘要。

| 部分 | 预算 | 必须回答的问题 |
|---|---:|---|
| Abstract + Introduction | 1.2页 | 为什么冻结适配需要研究原生更新方向？现有比较漏掉了什么？ |
| Native update construction and orientation controls | 1.5页 | 数学对象、冻结/可训练边界、零初始化、Full/Tangent、旋转位置、G+是什么？ |
| Experimental design | 0.65页 | 输入、训练/评测面板、配对与统计单位、各底座差异、首次锁定与后续分析身份 |
| Results | 4.0页 | 方向干预、可吸收旋转对照、范围与竞争边界、长度外推 |
| Discussion and limitations | 0.65页 | 哪些成立，哪些没有；为什么不等于普遍最优或完整机制解释 |
| Related work | 0.65页 | 原生算子适配与低秩/表征适配、蛋白单序列预测的联系与区别 |
| Conclusion | 0.15页 | 单段回答研究问题，不增加新主张 |

## 4. 前置方法应当短而可审查

先给一次完整的信息流：query序列→冻结PLM→可训练预测器→逐残基因子增量→冻结原生算子→pair残差→冻结folding下游。

对接口k和recycle t定义：

\[
(\delta a,\delta b)=f_\theta(e(x)),\qquad
\Delta U_{k,t}=G_{k,t}(a_{k,t}+\delta a,b_{k,t}+\delta b)-G_{k,t}(a_{k,t},b_{k,t}),
\]
\[
U^{(R)}_{k,t}=U_{k,t,0}+\mathcal R_k\Delta U_{k,t}.
\]

说明R仅在每个pair的输出通道上左乘，所有pair用同一个固定正交矩阵；不是旋转坐标或训练标签，也不是训练后临时扰动。三个固定R均从头配对训练。

Protenix正文给Full展开及Tangent删去二阶项的区别；D=508.5来自历史Train24训练侧MSA深度中位数，沿用为残差固定尺度，原始depth=1 baseline独立保留。数列和源码证据放附录，不把来源藏起来。OpenFold现场anchor依赖recycle，Atlas使用difference/product，不沿用Protenix的D。

零初始化是末层W/b为零，非整个encoder置零。冻结权重仍允许对注入输入反传。

同谱控制只在相同现场状态/参数处保持相关局部映射的范数和谱；不保证不同训练后输出范数或完整recurrent系统Jacobian相同。

G+显式接收相应query factors，与旧Generic分开。给自由输出仿射吸收R的三行证明；明确函数集合相同不保证AdamW轨迹等价。该性质是对照理由，不包装成深刻新定理。

## 5. Results按问题而非项目日期展开

### 4.1 Orientation changes adaptation within a fixed factor construction

先给Protenix锁定原生−旋转主要比较（Mini tangent Train24/384，Confirm96-B +0.05710 [0.04249,0.07274]），紧邻完整构造+0.04650及绝对分数。说明tangent是辨别性实验，不是最佳方法。早期Factor−Generic为构造动机和历史结果，不能替代G+对照。

### 4.2 Rotation sensitivity differs between constrained and absorbable heads

OpenFold Train96四格Factor/Rotated Factor/G+/Rotated G+与直接交互Ψ。Confirm96 +0.01882 [0.00662,0.03209]为本项主比较；Length48 +0.01400 [0.00604,0.02203]为次要。G+区间跨零不证明等效；交互不能唯一拆分表示约束和AdamW优化。它直接回应“是否所有头旋转后都会同样受损”。

### 4.3 Benefits depend on the backbone, training setting, and comparator

统一表给Protenix、OpenFold、Atlas各已完成配方的Query/Native/Rotated/G+，所有缺项显式保留。OpenFold Train384 Confirm96方向区间跨零，Length48为正；G+在OpenFold长链更高。Atlas两个面板未建立Native−Rotated或新增适配收益，不放入成功复现计数，也不只放附录。

Train96→384交互只用同为Full/1536的配对项。Protenix交互+0.01446，OpenFold Confirm96−0.01678；给直接区间，不用分别显著性比较。可留紧凑小图/表，切勿用tangent24替代Full96。

### 4.4 Fixed adapters extend beyond their adaptation-training lengths

Length48实际392–750，适配训练152–383；Protenix Native−Rotated +0.04252 [0.02899,0.05765]。绝对质量Native .52919、Rotated .48667、Query .32122、官方Mini-ESM .92139。新增匹配G+未完成，留待填且不承诺胜出。

说明Length48仅对最初Protenix长度研究为预先锁定新面板，后续跨底座/G+研究已观察。不能统一标为所有方法的盲确认。序列筛查不是家族隔离，也不是基础模型预训练隔离。

若版面不足，4.3和4.4合并成Scope and limits，不删Atlas或G+边界。

## 6. 图表优先级

1. Figure 1：原生残差/旋转残差/G+三条路径。标清frozen、trainable、零初始化、旋转位置；不能让图示暗示不同底座anchor均固定缓存。
2. Table 1：同序列来源下的Query/Native/Rotated/G+绝对分数，分底座/训练配方/面板，带额外PLM信息说明。原生系统与不同PLM参考单列，不能伪装同信息排行榜。
3. Figure 2：主要Native−Rotated配对差值及区间，含OpenFold接近零和Atlas未建立优势条件，标明原锁定与后续身份。完整种子/旋转交叉格和目标分布附录；正文保留关键不一致。
4. Figure 3或紧凑Table 2：OpenFold可吸收旋转四格与Ψ；不拿“一个显著另一个不显著”代替交互。
5. 长度结果可并入前述表/图；无需为每项历史分析新开图。

## 7. Discussion与附录分工

正文Discussion必须保留：方向优势≠Native普遍胜G+；Atlas本配置未建立效应；局部oracle几何未预测训练后目标收益；家族和预训练隔离未证明；系统质量差距明显；目标bootstrap条件于拟合模型。

局部梯度只用一两句概括，不在正文新增大节抢占结构证据。附录给完整局部响应、零/负关联、有限步共享迁移未建立正结果、数值失败与恢复。Engram早期支线按可复现历史归档，不作为现有主张的平行贡献。

训练超参数、版本/权重、归一化、mask/失败计分、逐目标与种子统计、时序及暴露清单、工程日志均放详细方法附录。正文仍要给足以判断比较有效性的最小配置表和终点定义。

## 8. Introduction开篇候选（英文，待共同修改）

Protein structure predictors combine learned sequence representations with internal computations that construct and refine pairwise states. Freezing these predictors and adapting a small interface offers a setting in which the structure of an update can be studied separately from changes to the downstream model. A natural construction predicts residue-level factor increments and combines them through a pretrained pair-generating operator. This construction raises a question that parameter count alone does not answer: does retaining the operator's output orientation matter for adaptation to the frozen downstream computation?

We study this question using query-anchored residual updates. The adapters start from the same unmodified baseline, and fixed orthogonal transformations rotate only their residual pair channels. Each rotated adapter is trained from initialization under paired conditions, rather than being perturbed after training. At a matched interface state, these transformations preserve the spectrum of the relevant update mapping while changing its orientation relative to the frozen downstream model. We additionally examine a generic head with explicit access to the query factors and an unrestricted output layer, for which an output rotation can be absorbed by a parameter reparameterization. This control separates the representable function class from the training behavior of that class; it does not assume that AdamW is rotation equivariant.

Our experiments reveal both benefits and limits of preserving native orientation. Protenix adapters retain an advantage over trained rotation controls across the tested settings, including a prospectively locked panel beyond the adaptation-training length range. OpenFold experiments extend the effect to an AlphaFold2 implementation in several conditions and show a positive interaction between head parameterization and rotation sensitivity. However, the direction advantage is not established in every OpenFold setting, generic heads can outperform native factor heads, and the tested AtlasFold interface does not establish a native-direction benefit. Together, these findings identify update orientation as a consequential but conditional aspect of frozen adaptation, rather than a guarantee of superior adapter performance.

此段为可编辑草案而非文献综述完成稿；相关工作文献需要下一步逐项核对。它不假定待完成Protenix G+胜负，不把Atlas负边界解释为已证实的天花板原因。
