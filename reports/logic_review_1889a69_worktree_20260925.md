# 逻辑与表达审阅：1889a69 上的 2026-09-25 工作稿

审阅对象：X570 `/home/husrcf/Code/engramholder`，HEAD 为 `1889a6972730a8f23e432b41f75448e0bf286d5f`，包含尚未提交的 Fresh192 与补偿重训整合。本文行号对应本地 `../` 快照；文件哈希保存在该目录的 `REVIEW_SNAPSHOT.json`。

审阅时 `build/iclr2027_conference.pdf` 仍为旧正文，摘要没有 Fresh192 或补偿重复结果。因此本轮以最新 TeX、生成表格及整合说明为准，不用旧 PDF 页码评价新稿排版。本轮审阅论证与表达，没有重新训练、评分或复算统计检验，也未把建议直接写入论文。

整体判断：现在的证据结构已有重要变化。Fresh192 为一个预先固定的 Protenix/ESMC 配方提供了新目标交互支持；补偿重复执行保留旋转臂的质量收益，却未再次确立相对 Native 的额外收益。论文应围绕“现象在明确范围内得到验证；一种干预改善质量，但其选择性缓解尚未得到重复支持”组织。两项结果回答不同问题，不应相互抵消，也不宜把两者都包装成机制验证。

1. **优先修改：让中心问题和贡献与现在的证据结构一致。**

   位置：[摘要开头](../sections/00_abstract.tex#L2)、[引言问题](../sections/01_introduction.tex#L14)、[贡献段](../sections/01_introduction.tex#L49)。

   “After task training”没有在问题层面提醒读者这是固定预算；“shared output freedom”又在介绍具体干预之前使用抽象概念。建议明确两个问题：固定预算下，输出通道旋转对不同参数化的性能影响是否不同；增加共享、可训练的正交输出映射，能否缩小 Factor 的 Native–Rotated 差距。

   可参考：

   > Under a fixed task-training budget, does output-channel rotation affect adaptation differently across parameterizations? We also test whether a shared trainable orthogonal output map reduces the Native–Rotated score gap within Factor.

   贡献段目前仍主要列“构造、四格比较、干预”，应明确写出 Fresh192 带来的实证增量。补偿实验可作为对解释的检验，不必要求它成为一个已经可靠解决方向代价的方法，才算贡献。

   > The study combines trained four-cell comparisons with a prospective test of fixed Protenix/ESMC models on new targets. A separate within-Factor intervention examines whether additional output freedom preferentially benefits rotated models.

   摘要结尾的 “identify a conditional design effect”较抽象。直接说“固定配方的新目标上确认了参数化相关的旋转敏感性；有用的输出映射没有在重复执行中重新建立选择性缓解”，比再概括一层 design effect 更清楚。关于评价原则的句子可保留一句。

2. **优先修改：并列展示补偿的三个量，解释变化来自哪里。**

   位置：[补偿结果](../sections/04_results.tex#L131)、[现成比较表](../generated/e2_retraining_comparison.tex#L1)。

   现在主文按原执行、重复执行的时间顺序分段，主表只给绝对均值；读者容易只记住一次区间排除零、一次包含零。更有解释力的信息是：

   | 量：pair-lDDT 分数差 | 原执行 | 同种子重复执行 |
   |---|---:|---:|
   | 旋转臂收益 B_R^C | +0.02973 | +0.03025 |
   | Native 收益 B_N | +0.01290 | +0.02960 |
   | 差距缩小 T_C=B_R^C−B_N | +0.01684 | +0.00065 |

   建议把附录现有三行对比及各自区间移入主文，与绝对均值表合并或相邻。用未舍入分数计算差值，继续保持当前做法。

   > The rotated-arm gain is similar in the two executions, whereas the Native gain is larger in the repeat, leaving little estimated differential mitigation. A greater gain than Native was established only in the original execution.

   这里是点估计的描述，不是新的执行间总体效应差异检验。原 T_C 区间为 [+0.00497,+0.02923]，重复为 [−0.01223,+0.01338]；不能从两者的显著性状态不同直接推导总体效应已改变，也不能证明重复执行中效应为零。

   因此，统一[摘要第 19 行](../sections/00_abstract.tex#L19)的 “relative mitigation does not” 和[讨论开头](../sections/05_discussion.tex#L7)，采用正文较准确的 “was not re-established”。

   还有一处新旧叙述衔接：[锚点附录第 1363 行](../appendices/implementation_and_evidence.tex#L1363)仍写 “useful mean compensation effect unchanged”。其本意是锚点实验不改写另一个实验，但在新增重复结果后容易成为无条件背书。建议改为：

   > These anchor results neither validate the reconstruction ranking nor resolve the compensation findings, whose relative benefit over Native was not re-established in the repeat.

3. **优先修改：两个新目标检验平级呈现，同时正面说明 Fresh192 确认的价值。**

   位置：[Fresh96/Fresh192 结果](../sections/04_results.tex#L68)、[讨论的选择限制](../sections/05_discussion.tex#L42)。

   Fresh96 当前是无标题段落，Fresh192 单独有 “New-target confirmation in Protenix” 标题。建议合并到 “New-target tests of fixed recipes” 下，或使用对称的小标题，避免成功结果获得额外视觉强调。

   | 固定配方 | 新面板 | 主要交互 Ψ，95% 区间 |
   |---|---|---|
   | OpenFold / Train96 / ESM2 | Fresh96 | +0.01014 [−0.00318,+0.02347] |
   | Protenix / Train384 / ESMC | Fresh192 | +0.02515 [+0.01832,+0.03213] |

   两项研究的底座、训练规模、特征和面板都不同。可以分别报告支持与未确立，不能据此声称两个底座的交互有显著差异。无需新增检验，只需避免读者将并列结果误读为直接比较。

   另一方面，讨论中的 “its new targets do not remove this selection”容易把合理的范围限制写得像是在否定新验证。历史数据选定配方后，再固定模型与方案到新面板检验，恰恰增加了针对该固定配方的前瞻证据。应直接说明对象和范围：

   > For a recipe selected on earlier data, Fresh192 provides prospective evidence of greater Factor rotation sensitivity on new targets. The inference concerns the fixed Protenix/ESMC models and the screened short-chain population; it does not establish robustness to retraining or transfer across recipes.

   保留 Fresh96 未确立交互、目标区间条件于拟合模型及不保证家族/预训练隔离的说明即可；不必在每次提 Fresh192 时重新叠加同一组限制。

4. **优先修改：“重建不能预测”必须限定到实际检验；cost 要写明是性能损失。**

   位置：[摘要第 20 行](../sections/00_abstract.tex#L20)、[引言第 47 行](../sections/01_introduction.tex#L47)、[结论第 7 行](../sections/07_conclusion.tex#L7)。

   “retraining costs”通常容易让读者想到训练时间或算力开销。这里检验的却是八种旋转在任务重训后的 Native-minus-Rotated 分数差。建议统一用 “post-training rotation penalty” 或 “Native–Rotated score gap after retraining”。

   结论的 “Reconstruction does not predict costs”范围也太大。现有结果是一个预先指定、有限预算的 teacher-update reconstruction 指标，在八个旋转上未确立预测的正向排序关联，不能升级为所有重建指标普遍无预测能力。

   摘要可改：

   > A prespecified reconstruction diagnostic did not establish the predicted ranking of post-training rotation penalties.

   讨论可更精确：

   > The prespecified teacher-update reconstruction score showed no established positive association with Native–Rotated score gaps after retraining across the eight tested rotations.

   这是把阴性结论的作用域写准确，不需要淡化或删除阴性结果。

5. **解释补偿实验之前，补一条从四格结果到干预的逻辑桥。**

   位置：[Results 4.2 开头](../sections/04_results.tex#L108)。

   读者需要知道：四格的 Ψ 比较两个完整参数化，不能单独把差异归于输出层自由度；接下来是在 Factor 内加入 C，检验这一干预是否优先帮助旋转臂。这样两部分不会被读成“发现 G+ 闭包 → 已解释 Ψ → 加 C 必然解决”。

   另一个必要条件目前仅在附录完整交代：补偿使用的是独立八旋转研究中的两枚保均值旋转，并非前面四格的三枚历史 dense rotations。C 的吸收保证也只针对指定子群。

   可补两句：

   > To test whether added output freedom preferentially benefits rotated Factor, we jointly train an orthogonal output map with the head. This study uses two mean-preserving rotations from the separate prediction experiment, selected by low/high reconstruction error before their task-training scores were observed.

   再将闭包句限定为：

   > For these rotations in the mean-preserving subgroup, the compensated Factor classes are equal in ideal arithmetic.

   C 与头联合重训改变了输出自由度和优化过程；T_C 衡量这项联合干预的差中之差，不唯一识别表达能力机制。当前讨论已有这一边界，正文只需给读者清楚的承接。

6. **让每个结果段落自带底座、配方和比较对象，减少远距离指代。**

   位置：[Results 第 44 行](../sections/04_results.tex#L44)、[第 55 行](../sections/04_results.tex#L55)、[第 68 行](../sections/04_results.tex#L68)。

   具体替换：

   | 现用表达 | 建议 |
   |---|---|
   | The completed Train96/Full follow-up | The Protenix Train96/Full follow-up |
   | The Train384 feature study | The Protenix Train384 feature study |
   | the same 24 OpenFold adapters | the 24 fixed OpenFold Train96/ESM2 adapters |
   | All nine new fits（补偿段） | Each execution trains nine compensated models |

   前两个段落紧接 OpenFold 内容，主语省略确实容易导致误读。新增 repeat 后，“九次训练”也要限定到每次执行，不能让人误以为原执行加重复总共只有九次。

   同时可把摘要和引言的 “independently train / independent task training”改成“train separately under paired settings”：前者本意是各自训练，但容易与统计独立性混淆，后者同时保留了分开拟合和配对设计。

7. **符号与概念定义还需做少量收口，随后压缩重复限制。**

   [统计定义](../sections/03_design.tex#L79)仍只定义 d_{H,i} 和 Ψ，没有显式定义后来使用的 Δ_H。建议补：

   > We write Δ_H = N⁻¹ Σ_i d_{H,i} for the signed Native-minus-Rotated score contrast; positive values indicate a rotation cost. Thus Ψ = Δ_F − Δ_{G+}.

   这也明确了“代价”是带符号分数差，负值允许旋转有益。

   [方法的 baseline 定义](../sections/02_construction.tex#L7)仍可加一句：U_{k,t,0} 是所选接口的原生更新量，不是整个 pair state，也不是所有分支共享的一次独立 query-only 轨迹。OpenFold 在各自当前状态上计算；Protenix 则复用缓存 query update。新图已经更清楚，正文定义应达到同样精度。

   [Related work 最后一句](../sections/06_related_work.tex#L15)的 “Factor lacks a general compensating reparameterization”比方法的“没有对应一般保证”更强。若未给出非闭包证明，建议与方法统一为 “no analogous general closure guarantee is established for Factor”。

   最后，[讨论第 20–31 行](../sections/05_discussion.tex#L20)连续列举重建、范数交换、局部兼容性、小更新、共享迁移、锚点实验，读者难以保持问题主线。建议按作用归纳：重建指标未预测排序；锚点干预未支持平均帮助；范数交换约束了全局幅度解释。旧诊断细节保留在附录，主讨论着重解释 Fresh192 与补偿重复结果。

建议落实顺序：先调整补偿三量并列表和全稿一致措辞，再统一两个新面板的呈现，随后改摘要/贡献/结论，最后整理术语与段落指代。本轮没有发现需要为了修复这些逻辑表达而新增训练的理由。

本报告只保存于当前工作目录并同步至 X570 论文仓库 `reports/`；未写入系统记忆，未更改 AGENTS.md 或论文文件，也未提交其他写作线程的改动。
