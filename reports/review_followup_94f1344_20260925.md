# 修订稿复审：94f1344

审阅日期：2026-09-25。对象为 X570 `/home/husrcf/Code/engramholder` 的 `94f1344ce7d9b148b705f5a3a5949076b380d6e4`，以及 `build/feedback_v39/paper.pdf`。本轮读取时论文源文件无未提交修改；其他未跟踪报告未改动。

总体判断：前轮主要逻辑意见已经落实，摘要、结果、讨论与结论现在一致。未发现新的主论证矛盾，不建议继续重排主线或反复增加限制语。剩下两处应优先处理的局部表达，以及两处较小的范围澄清。

核对范围：本地快照中全部 65 份编译输入及 PDF 哈希与编译记录一致；`generated/numbers.tex` 与上一轮快照逐字节相同。本轮查看了 PDF 第 1、6、7 页，涉及摘要、交互图和补偿表；这不是全稿视觉或最终投稿验收。未重新计算实验统计量或评分，未检查投稿页数，未启动实验。

已解决的前轮问题：两个新目标研究同级且提前呈现；Fresh192 的正面证据与适用范围同时明确；补偿表并列绝对分数、B_R、B_N、T_C 及区间；“未重新建立”没有被写成零效应；重建诊断限定为预设指标和八种旋转；补偿旋转子群、接口基线、Δ_H 定义及 Factor 闭包保证的措辞已补齐。新增执行条件表也明确区分已记录条件与尚未定位的差异。

1. **应改：把“证实没有变化”改为“未确立变化”。**

   [结果第 253 行](../sections/04_results.tex#L253)目前写：

   > OpenFold Length48 establishes no change.

   这里依据是变化区间包含零，不是等效性检验；该措辞可以被读成已经确立零变化。建议直接替换为：

   > The OpenFold Length48 interval does not establish a change.

   类似地，[第 246 行](../sections/04_results.tex#L246)可统一为：

   > Neither an adaptation benefit nor an interaction is established for AtlasFold with either feature source.

   这与全文已修正的统计解释保持一致，不改变任何数据或结论范围。

2. **建议优先：在主表对应行直接标出混合计算后端。**

   位置：[主表表注](../sections/04_results.tex#L210)、[生成表格](../generated/complete_fourcell_means.tex#L3)。正文和[附录第 635 行](../appendices/implementation_and_evidence.tex#L635)已经披露来源，因此问题在于表格独立阅读时的信息位置，而非没有披露。

   建议用两种上标区分：

   | 行 | 已披露的来源 | 对读者理解的影响 |
   |---|---|---|
   | Protenix / Train384 / ESM2 / Length48 | Factor 和 Query 的历史预测在 H100；G+ 两格在 MI250 | head 比较同时包含后端差别 |
   | AtlasFold / ESMC 的两个面板 | 一个种子的各模型在 H100，另外两个种子在 MI250 | 后端与种子混杂；特征替换比较也改变后端 |

   表注只需写清这两个标记并指向现有附录。不要把第二种情况写成 Native/Rotated 使用不同后端，也不应把每一行都统称为跨硬件的独立重复。上标应在表格生成逻辑中加入，不手改锁定数字。

3. **小澄清：把“三旋转、相同优化器”的总述限定到核心四格研究。**

   [Experimental design 第 39 行](../sections/03_design.tex#L39)的 “Within each recipe”范围略宽；后面的预测研究有八枚旋转，补偿研究有两枚旋转及额外 C 参数组。细节各处正确，建议总述直接限定为：

   > Within each core four-cell recipe, heads share data order, three training seeds, random-condition rules and optimizer settings. Each of three fixed rotations uses all three seeds. The prediction and compensation studies use their separately specified rotation sets and parameter groups.

   这样读者不会把主矩阵的计数规则套到所有实验。

4. **小澄清：明确“没有联合多重校正”的作用范围。**

   [讨论第 45 行](../sections/05_discussion.tex#L45)的 “Follow-ups lack joint correction”可以指没有跨研究统一校正，但也可能被读成所有后续研究均没有校正。OpenFold 特征研究已有预设 Holm 校正，建议用一句话消除歧义：

   > The OpenFold feature study applies its prespecified Holm correction; no correction is applied jointly across all follow-up studies.

   无需新增统计检验。

可顺手处理的文字细节：摘要先提 Factor 的差距，随后才描述该构造，generic head 又未在第一次出现时注明 G+。建议在首次描述各构造时明确命名 Factor、G+，随后再使用简称。摘要尾部对 Fresh192 与补偿重复的总结略重复，但不构成逻辑错误，也无需为此重写整段。

本轮建议以以上局部修改收口。对知识增量和机制解释的评价仍取决于既有证据；文字修订的收益是让这些证据更容易被准确理解，不需要为了回应本轮意见自动启动补实验。

本报告只保存于当前工作目录并同步至 X570 论文仓库 reports；未写入系统记忆，未修改论文、实验或 AGENTS.md，未提交其他线程的工作。
