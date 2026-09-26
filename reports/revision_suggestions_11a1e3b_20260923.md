# 论文修改建议：基于 11a1e3b 的继续盲审

审阅对象：X570 `/home/husrcf/Code/engramholder`，提交 `11a1e3b`，以及该版本对应的编译 PDF。日期：2026-09-23。本文给出稿件修改意见，未修改论文、实验协议、checkpoint 或评分数据，未读取当前稿件之外的新面板分数。行号对应上述快照；后续编辑后会移动。

本轮判断：前轮涉及数据定义、G+ 对照边界、固定预算解释和绝对性能的主要表述问题已有修复。尤其应保留 Train24/Dev8 为继承集合的说明，不能将新增候选的发布日期条件套到它们身上。当前最有价值的修改是让读者一次理解问题、统计量和结论适用范围，减少来回翻页与重复限制语。

以下前五项建议优先处理，后三项用于加强定位和压缩正文。均不依赖新增训练。

1. **摘要先提出具体问题，紧接着呈现正结果与范围限制。**

   位置：[摘要](../sections/00_abstract.tex#L2)。当前开篇较泛，随后连续介绍构造和控制；读者接触科学问题较晚。Train96 的核心结果与 Train384 的完整交互结果之间又插入了 Protenix 和范数实验，使主结论范围需要回读才能明确。

   建议顺序：问题 → 四格设计 → Train96 正交互 → Train384 两种特征下尚未确立正交互 → Protenix 的补充结果 → 有条件的结论。直接把目前 Train384 结果移到 Train96 结果之后。保留主要效应量与区间，压缩实验内部标签和流程描述。

   可参考的开头：

   > We ask whether an adapter's sensitivity to output-channel rotation depends on its parameterization. In frozen protein structure predictors, we compare query-anchored residuals composed through pretrained native operators with a factor-aware pair head whose free output layer can absorb the rotation. A paired four-cell design trains unrotated and rotated versions of both heads from the same zero-residual function.

   不应将 Train384 写成“证明交互不存在”；当前区间支持的是尚未确立正交互。也不能根据一行显著、另一行不显著就断言两行之间显著不同。

2. **把完整交互范围表移到核心结果旁边，让全文围绕同一个问题展开。**

   位置：[核心结果开头](../sections/04_results.tex#L3)、[交互范围表](../sections/04_results.tex#L148)。实际 PDF 中核心交互在第 5 页，三套完整交互的 Table 3 到第 7 页才出现；中间的范围图主要展示的是 Factor 自己的旋转差 Δ_F，读者容易混淆两个统计量。

   建议将 Table 3 移至 4.1，紧接 Train96 核心交互。统一呈现 Train96/ESM2、Train384/ESM2、Train384/ESMC 的 Ψ，再由后续章节解释训练规模、特征替换和其他底座的结果。保留当前表注“这些区间不是行间差异的直接检验”。

   可采用的结果组织：完整四格交互 → 训练规模与特征下的范围 → Protenix 的性能与长度结果 → 范数对照 → 其余边界或负结果。按科学问题合并相关段落即可，不必机械照搬这一顺序。

   当前第 6 页 ESMC 表浮到 Protenix 和长度章节上方，而对应 ESMC 文字位于后面。调整浮动位置时一并修正，使表格与首次解释相邻。PDF 未发现明显裁切或重叠，本项是阅读顺序问题。

3. **明确基线 U 的物理含义与注入位置，修正“baseline unchanged”的潜在歧义。**

   位置：[残差与注入定义](../sections/02_construction.tex#L7)、[旋转控制](../sections/02_construction.tex#L53)、[Figure 1](../sections/02_construction.tex#L91)。

   当前 U_{k,t,0} 的“0”容易被理解为整个训练过程中恒定的张量；“baseline update”也可能被误读成整个 pair representation。建议改为 U^{base}_{k,t}，紧接公式说明它是所选接口的更新量，加入残差后仍按主干原有方式接入 pair state。冻结参数并不意味着 live 接口的基线输出始终不变。

   可直接改写为：

   > U^{base}_{k,t} denotes the baseline update at the selected interface, rather than the accumulated pair representation. The adapted interface supplies U^{base}_{k,t} + R_k ΔU_{k,t}, retaining the backbone's surrounding pair-state additions. Protenix uses its cached query update; at live interfaces, the baseline is evaluated at the current adapted state.

   “the baseline ... [is] unchanged”宜改成“the rotation acts only on the adapter residual”。这说明干预规则；不同训练轨迹下 live baseline 的数值仍可能因当前状态不同而变化。

   Figure 1 建议增加两条明确的输入支路：native anchor → residual composition，以及 baseline update → addition。分别标注 Protenix 的 cached query 和 OpenFold/AtlasFold 的 live anchor，让读者从图中看出实际数据流。

4. **统一参数符号和效应量定义，避免将不同概念写成同一个字母。**

   位置：[冻结解码器](../sections/02_construction.tex#L30)、[G+ 吸收关系](../sections/02_construction.tex#L72)、[统计定义](../sections/03_design.tex#L80)、[训练规模比较](../sections/04_results.tex#L228)。

   建议冻结 decoder 用 W_nat，G+ 的可训练最后一层用 V、c。当前 W 同时表示冻结与可训练矩阵，b 同时表示 native factor 与 affine bias，读者容易误读 G+ 的吸收证明。相应写成 V_R = RᵀV、c_R = Rᵀc。用一句话或小符号表交代 factor 维度 r、pair-channel 维度 C 和 R∈R^{C×C}；若两侧 factor 维度不相等则分别写 r_a、r_b。

   在现有 d_{H,i} 后显式补充 Δ_H = (1/N) Σ_i d_{H,i}，于是 Ψ = Δ_F − Δ_{G+}。正文已大量使用 Δ_F，却没有在统计定义处完成这一映射。

   规模比较统一写 Δ_{F,384} − Δ_{F,96}。目前的“(N−R)384−(N−R)96”既与 N=目标数、R=旋转矩阵冲突，也容易让人忘记它只比较 Factor 的旋转差，不能替代 Ψ 的变化。

5. **训练设计用清楚的四格计数取代历史扩展过程的叙述。**

   位置：[Training and pairing](../sections/03_design.tex#L36)。

   “Train24/384 updates”建议改成“24 training proteins and 384 updates”，或统一用 (n_train, T) 表达。训练集合大小与更新步数是解释本文结果的关键，不能依赖斜线简写。

   完整 OpenFold 四格矩阵可一次写清：Factor Native 3 个训练种子，Factor Rotated 3 种旋转 × 3 个种子，G+ Native 3 个种子，G+ Rotated 3 × 3，共 24 次训练。每种矩阵分别适用这一结构；其他非完整研究明确注明缺哪些格子。

   当前“Train96 extension adds nine rotated G+ fits”仍偏向实验历史，在已经有三套完整矩阵时容易让人误以为只有 Train96 有旋转 G+。历史 chronology 留附录；正文给最终设计。

   保留以 target 为重采样单位的表述。训练数、旋转数与补充指标数都不能并成更多独立蛋白。先在每个 target 内按种子和旋转形成 cell mean，再做配对差；不要把 Native 的 3 个模型与 Rotated 的 9 个模型摊平混算。

6. **补足最接近的相关工作，明确本文新增的观察是什么。**

   位置：[Related work](../sections/06_related_work.tex#L3)。目前有 LoRA、ReFT、PiSSA、BOFT 以及 AdamW，但与“旋转后训练行为为何变化”最直接相关的两条研究线没有充分定位：

   - [Understanding Adam Requires Better Rotation Dependent Assumptions](https://arxiv.org/abs/2410.19964)：研究 Adam 对参数空间旋转的敏感性。
   - [LoRA Done RITE: Robust Invariant Transformation Equilibration for LoRA Optimization](https://arxiv.org/abs/2410.20625)：研究 LoRA 因子缩放、旋转下优化更新的变换不变性，并提出相应优化方法。

   建议新增一段，明确这些工作与本文研究对象的联系和差别：本文在冻结折叠主干接口旋转 adapter 的输出残差，用两种头及 G+ 的闭包性质组织对照，观察特定训练配置下的 head-by-rotation interaction。现有 Adam 基依赖性意味着不应将“旋转后训练不同”本身当作全新的发现；本文需要突出具体控制设计与蛋白结构任务中的实证发现。

   这项建议是文献定位，不自动要求增加 LoRA-RITE 训练，也不表示上述文献已解释本文效应的唯一机制。

7. **Introduction 与 Discussion 应各承担不同任务，减少重复的限制语。**

   位置：[Introduction 后半部分](../sections/01_introduction.tex#L43)、[Discussion 开头](../sections/05_discussion.tex#L3)。当前对“不是性能上界、不是唯一机制、不是普遍成立”的说明已经相当完整，但在摘要、方法、结果、讨论反复出现，挤占了正面贡献的解释。

   建议 Introduction 的贡献归纳为：query-anchored residual construction；区分单头旋转差和跨头交互的四格实证设计；刻画其跨配置适用范围并检验全局残差范数解释。不要将同一个正结果拆成过多独立贡献。

   Discussion 第一段先回答“读者应从结果学到什么”：评估此类 adapter 时，绝对适应收益、单头旋转敏感性、跨头交互、Native–G+ 性能差是不同问题，应该分别报告。随后总结本研究在哪些配置下支持哪一项。

   详细函数类限制集中留在方法；具体负结果留在对应结果段；观察过的面板、种子不确定性和未确立收敛等推断边界集中在讨论末段。压缩重复，不删除关键边界。

8. **附录数据表与 D 常数再做两处小澄清。**

   位置：[数据组成表](../appendices/implementation_and_evidence.tex#L91)、[D 的定义](../sections/02_construction.tex#L44)。

   数据表 N 列如有“新增 288”这种行，可改为“Component size”，并在表注写 Train96=24+72、Train384=96+288，防止把新增链数当成整个集合大小。保持每一部分各自的筛选条件、发布日期协议和首次用途；已修正的 Train24/Dev8 说明无需再改回。

   D=508.5 容易使读者误以为残差被放大数百倍。可紧接偏置抵消公式说明，在该 residual expression 中实际系数为 D/(D+ε)≈0.999998；D 是沿用的训练侧常数，原始 depth-one query update 仍是 baseline。保留来源说明，不需要另增一个 D 消融来解决纯表达歧义。

完成这些修改后，主结论仍应保持目前证据能够支持的范围：特定 OpenFold Train96/ESM2 配置下出现参数化相关的旋转敏感性差异，其他配置的完整交互与绝对性能结果刻画其边界。文字重组能显著提高可读性和说服力，但不能将尚未在当前稿件呈现的独立确认当成已有证据。

执行记录：本报告只落盘于当前工作目录及 X570 论文仓库 `reports/`。未写入系统记忆，未更改任何 AGENTS.md；上传前发现的其他未跟踪实验文件保持原状。
