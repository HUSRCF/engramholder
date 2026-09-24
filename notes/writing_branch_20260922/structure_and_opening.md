# ICLR写作分支：结构建议与开篇候选

> 后续状态：A66及Protenix Train96完整四格现已进入正式稿与v7来源锁，见[当前稿件状态](../../reports/manuscript_fill_20260922.md)。下文保留原审阅／设计时点的缺项说明，不是当前完成清单。

2026-09-22，v2。用户已确认贡献优先级：参数化×方向交互 → query-anchored方法构造 → 迁移范围与未成立条件。中心主张采用用户原句；以下组织和英文文字仍是待审阅草案。v1已归档为structure_and_opening_v1_before_claim_alignment.md。本分支只修改写作文件，不修改主线训练、历史报告、README工作状态或根LaTeX模板，不提交/推送。

已读取：AGENTS.md、notes/workflow.md、reports/evidence_review.md、math_contract.md、results_update_20260922.md、reviewer_priorities_20260922.md、protenix_gplus384_status_20260922.md。数值来自这些已有核验报告，本分支没有重新评分CIF或查询远端训练。旧数学笔记中的Atlas“待完成”已被9月22日结果更新取代。匹配Protenix Train384 G+正在训练，无最终分数，不预填。

## 1. 结构决策

建议保留生物论文的问题驱动结果叙事，但将理解实验必需的方法定义前置：

Introduction → Native update construction and orientation controls → Experimental design → Results → Discussion and limitations → Related work → Conclusion。

详细Implementation and reproducibility methods放附录。Related work也可并入Introduction的定位并保留一个短节；不让文献综述阻断问题到方法的推进。

ICLR 2027官方Author Guidelines规定初稿主文≤9页、参考文献与附录另计，且审稿人无义务阅读附录。所查指南未规定IMRaD或具体章节顺序。因此这不是格式禁止，而是论证可读性选择。
来源：https://iclr.cc/Conferences/2027/AuthorGuidelines （2026-09-22查阅）。

不推荐将全部Methods放到Discussion后：读者在看到任何Native−Rotated结果时，就必须知道旋转作用于训练中的通道残差、每组从零残差开始独立配对训练、以及G+的函数集合为何可以吸收旋转。缺少这些定义，最强对照会被误读成对已训练模型的事后破坏。

## 2. 已确认的中心主张与贡献顺序

用户确认的中心句，后续各节围绕它展开：

> Native update orientation can matter for constrained adaptation, but its effect depends on the parameterization and frozen predictor; preserving native directions neither guarantees adaptation gains nor makes native factor heads uniformly superior to generic alternatives.

三个贡献层次：

1. **核心发现：参数化与旋转敏感性发生交互。** OpenFold Train96四格在相同固定旋转和AdamW配方下，Factor的方向差异比具有自由输出层的G+更大。用直接交互Ψ检验，不用一组显著/另一组不显著替代。该发现限制于这项已执行设计，尚无跨底座的完整Ψ复制；不能把它写成普适规律或唯一的表达能力解释。
2. **方法构造：query-anchored原生因子残差与方向控制。** 共享预测器输出因子增量，通过冻结原生算子构造零初始化残差；固定正交变换在匹配现场状态下保留相应局部谱性质。它是具体适配方法和检验工具，不以普遍胜过G+为贡献前提。
3. **适用范围：迁移与边界并列。** Protenix/Tiny、长度与训练规模、部分OpenFold配置提供支持；OpenFold的G+竞争优势、部分方向比较未成立，以及Atlas非OPM配置未建立优势限制一般性。

这里“depends on the frozen predictor”概括的是不同底座及各自配方下的条件性；跨底座同时改变接口、损失、原生表征等，尚不能把差异唯一归因于架构。

**叙事中心不改写研究时序。** OpenFold四格是后续辨别性研究，其Confirm96-B交互为该研究预定主要比较，不是整个项目最初的主要假设；两个面板此时均已观察。最初Protenix锁定主比较及Length48首次锁定身份按历史保留。图表按科学问题排序，附录保留完整协议时间线。

论文类型：具体方法支持的受控实证发现。标题/摘要优先突出orientation与parameterization，不能承诺高性能folding替代、完整机制解释或普遍方法最优。

## 3. 九页内的草拟安排

以下是按用户后续审阅修订的工作预算，含图表及摘要；预留约0.5页。实际页数以编译为准。

| 部分 | 预算 | 必须回答的问题 |
|---|---:|---|
| Abstract + Introduction | 1.2页 | 为什么方向干预的结果需要结合参数化解释？这项研究发现了什么交互？ |
| Native update construction and orientation controls | 1.5页 | 数学对象、冻结/可训练边界、零初始化、Full/Tangent、旋转位置、G+是什么？ |
| Experimental design | 1.0页 | 输入、训练/评测面板、配对与统计单位、各底座差异、首次锁定与后续分析身份 |
| Results | 3.4页 | 优先四格交互，再呈现方向证据、迁移范围与竞争边界 |
| Discussion and limitations | 0.6页 | 哪些成立，哪些没有；为什么不等于普遍最优或完整机制解释 |
| Related work | 0.6页 | 原生算子适配与低秩/表征适配、蛋白单序列预测的联系与区别 |
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

## 5. Results按贡献优先级展开

### 4.1 Rotation sensitivity interacts with adapter parameterization

Results首节给OpenFold Train96四格，先绝对质量、再方向差值、最后直接交互：

| 面板 | Factor | Rotated Factor | G+ | Rotated G+ | Ψ [95% CI] |
|---|---:|---:|---:|---:|---|
| Confirm96-B | .50190 | .48209 | .49534 | .49435 | +.01882 [.00662, .03209] |
| Length48 | .38872 | .37834 | .39849 | .40211 | +.01400 [.00604, .02203] |

\[
\Psi=(S_F-S_{F,R})-(S_{G+}-S_{G+,R}).
\]

Confirm96为这项后续研究的唯一主要比较，Length48为次要。每目标先按种子/旋转聚合；CI条件于固定拟合模型，不能把9个模型组合当9倍蛋白。两面板三个种子、三个旋转的主指标交互边际都正，但9个交叉单元为8/9和7/9正，不能写“所有运行均正”。

函数集合可吸收旋转是理论性质；实验没有证明G+经过AdamW获得旋转不变训练。交互涉及整个头参数化（含自由度与优化几何），不是严格控制其他一切后的“仅表达约束”效应，也未建立对训练动力学的唯一因果归因。

### 4.2 Native orientation benefits factor adaptation in the tested Protenix settings

给Protenix锁定主比较（Mini tangent Train24/384，Confirm96-B +.05710 [.04249,.07274]），紧邻Full +.04650及绝对分数。介绍Tiny、均值保持旋转与同构造Full Train96/384作为不同控制/范围。Tangent是辨别性工具，不是最佳方法。Factor−旧Generic保留为历史构造证据，不能顶替G+。

这一节展示方向效应的覆盖，不声称Protenix已复制头×旋转交互；没有Protenix旋转G+四格就没有该交互证据。

### 4.3 Fixed adapters transfer beyond their adaptation-training lengths

Length48实际392–750，适配训练152–383；Protenix Native−Rotated +.04252 [.02899,.05765]。绝对质量Native .52919、Rotated .48667、Query .32122、官方Mini-ESM .92139。匹配G+尚无最终结果，保留明确待填。

Length48仅对最初Protenix长度研究为新锁定面板，后续跨底座/G+研究均已观察。长度外推相对适配训练而非基础模型预训练，序列筛查不保证家族隔离。不能用不同面板均值相减声称纯长度因果效应。

### 4.4 Native orientation does not guarantee gains or superiority over generic heads

统一主表同时给Protenix、OpenFold、Atlas各配方Query/Native/Rotated/G+。OpenFold Train384 Confirm96方向区间跨零，而Length48有支持；OpenFold两种训练规模的长链G+竞争优势有直接配对结果。Atlas两个面板未建立Native−Rotated或适配收益，正文呈现，不将其列入成功复制。

同构造Full/1536的数据规模交互：Protenix Confirm96 +.01446 [.00350,.02586]，OpenFold Confirm96 −.01678 [−.03214,−.00261]。这描述相应配方下不同变化，不解释为AF2学会纠正旋转或Protenix获得更多进化知识。不能用Tangent24替代Full96。

四格核心发现与边界主表应相互交叉引用，不让读者到Discussion才发现G+可以更好。若版面不足，合并4.2和4.3，不删Atlas与G+。

## 6. 图表优先级

1. **Figure 1：方法与辨别性控制。** 用2×2矩阵表示Factor/G+ × 原方向/固定旋转，旁边给残差信息流。标清冻结/可训练、零初始化、旋转位置；现场anchor与Protenix缓存差异用脚注。
2. **Figure 2：核心四格结果与直接Ψ。** 先给各系统绝对得分与两种方向差，再给目标级配对交互CI；不能依靠柱高或各自显著性推出交互。
3. **Table 1：范围与竞争边界。** 分底座、训练配方和面板同时列Query/Native/Rotated/G+及缺项，包含OpenFold近零和Atlas未建立优势。Query未接入额外ESM2的表征差异明确标注。
4. **Figure 3：方向比较与长度迁移。** 展示主要方向差及区间，标明锁定/后续身份。完整逐种子、逐旋转、目标分布放附录，正文保留关键不一致。
5. 官方系统质量参照用小表或主表单独区域；不同原生PLM/输入/预算不是同信息排行榜。不给每轮历史研究新增一个图。

## 7. Discussion与附录分工

正文Discussion必须保留：方向优势≠Native普遍胜G+；Atlas本配置未建立效应；局部oracle几何未预测训练后目标收益；家族和预训练隔离未证明；系统质量差距明显；目标bootstrap条件于拟合模型。

局部梯度只用一两句概括，不在正文新增大节抢占结构证据。附录给完整局部响应、零/负关联、有限步共享迁移未建立正结果、数值失败与恢复。Engram早期支线按可复现历史归档，不作为现有主张的平行贡献。

训练超参数、版本/权重、归一化、mask/失败计分、逐目标与种子统计、时序及暴露清单、工程日志均放详细方法附录。正文仍要给足以判断比较有效性的最小配置表和终点定义。

## 8. Introduction草案v2（英文，待共同修改）

Adapting a frozen protein structure predictor requires learning updates that its pretrained computation can use. An adapter built around a native pair-generating operator inherits both a restricted update construction and an output orientation relative to the downstream model. These properties are distinct: an orthogonal change of output coordinates can preserve local spectral properties while changing the updates presented to a fixed downstream computation. Yet a performance change under rotation does not, by itself, establish whether that sensitivity is specific to the adapter parameterization or shared by more flexible heads.

We investigate this distinction with query-anchored residual adapters. A shared sequence-conditioned predictor produces residue-level factor increments, which are combined through a frozen native operator to form pair updates. Zero initialization recovers the unmodified query-only baseline. We compare native updates with fixed orthogonal rotations of the residual channels, training every condition from initialization under paired settings. We also construct a generic head, G+, with explicit access to the corresponding query factors and an unrestricted affine output. This head can absorb an output rotation by reparameterizing its last layer without changing its representable function class. This property supplies a discriminating control, although it does not make coordinate-wise AdamW updates rotation equivariant.

The central empirical result is an interaction between adapter parameterization and rotation sensitivity. In the tested OpenFold Train96 setting, the native-minus-rotated difference is larger for the factor head than for G+. The direct interaction in Cα pair-lDDT is 0.01882 on Confirm96-B (95% target-bootstrap interval [0.00662, 0.03209]) and 0.01400 on Length48 ([0.00604, 0.02203]). This follow-up study uses previously observed target panels; its primary comparison was fixed before the new rotated-G+ runs. The result shows that the two parameterizations respond differently to the same rotation intervention under the specified training recipe. It does not uniquely attribute the difference to representational restrictions rather than optimization dynamics.

The broader experiments establish a scope for this finding without making it universal. Protenix experiments support native-direction advantages across the tested checkpoint and training settings, including a prospectively locked length-extrapolation study using fixed adapters. OpenFold supports the direction effect in several conditions, but not on Confirm96-B after Train384 adaptation, and generic heads achieve higher scores in some matched comparisons. The tested non-OPM AtlasFold configuration does not establish a native-direction benefit or an adaptation gain. Native update orientation can therefore matter for constrained adaptation, while neither guaranteeing useful adaptation nor making native factor heads uniformly preferable to generic alternatives.

Our contributions are threefold:

1. **A controlled empirical interaction.** We compare rotation sensitivity between a native factor head and a generic head that can absorb the rotation, and directly estimate their interaction under a matched AdamW recipe.
2. **A concrete adaptation construction and intervention.** We formulate query-anchored residuals around frozen native operators, with zero initialization and fixed orthogonal output controls whose local spectral invariances are explicit.
3. **Evidence of transfer and limits.** We evaluate direction effects across tested predictors, training-set sizes, and adaptation-length ranges, while reporting conditions without established benefits and cases in which generic adaptation is more effective.

写作备注：相关工作引用尚待补全；不是投稿可直接使用的无引用终稿。正式Introduction可将两面板完整数值移至Results以减轻首节负担。中心句在摘要/引言末尾/结论保持语义一致，但不重复机械粘贴。Protenix匹配G+未完成，不预写胜负。

## 9. 必须避免的机制升级

| 目前可以写 | 目前不能写 |
|---|---|
| OpenFold中头参数化与旋转敏感性存在实证交互 | 已唯一证明表达约束导致全部方向收益 |
| G+可用末层重参数化吸收固定旋转 | G+训练已经学会旋转不变性；AdamW轨迹等价 |
| Atlas当前接口/配方未建立原生优势 | 只有弱基线需要适配；强基线必然无效 |
| 多个已测试底座/设置表现不同 | 已识别出是哪一个架构组件造成差异 |
| 原生方向优势可延伸到适配训练长度之外 | 任意长度、严格家族或基础模型预训练外推已成立 |
| 当前配方下Native可能不如G+ | 原生方向因此没有作用，或G+必然普遍最优 |
