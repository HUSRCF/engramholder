# 七项措辞评审：比较对象、信息顺序与术语

2026-09-26。以`dd9d009`的v48叙事修订为基础（本轮开始时该修订尚在工作区，随后已独立提交）。
用户确认另一任务已完成编辑，可继续润色。本轮保留其章节顺序、预算／时序总览和研究身份说明，继续完成本轮措辞修订。

## 修改原则

这些意见主要修正理解成本和比较对象，不构成新实验或加强结论的理由。
最重要的是区分三类量：

| 数学对象 | 统一写法 | 含义 |
|---|---|---|
| \(\Delta_H\) | signed Native–Rotated score difference | 同一头的有符号分数差；正值有利于Native，不能默认为绝对敏感程度 |
| \(\Psi=\Delta_F-\Delta_{G+}\) | interaction | 两种头的行内差之差，不直接排序结构质量 |
| \(T_C=B_R^C-B_N\) | additional compensation gain for rotated Factor, relative to Native Factor's gain | 两臂加C之后的增益之差，不是补偿后Rotated−Native绝对分数 |

这些是正文主线的术语。历史协议、代码字段和封存统计输入不改名；其他独立析因干预仍保留自己的明确公式。

## 七项落实

1. **摘要与引言先定义对象。** 摘要先提出固定预算问题，再分别定义Factor、G+和四格比较，最后引出共享正交输出映射。引言第一段先定义Factor。
2. **E2明确比较增益。** 主文直接说明旋转臂两次估计增益相近，重复执行中Native估计增益较大。只有原执行建立旋转Factor相对Native Factor的额外补偿增益；保留数值、区间、同种子身份和“显著性状态不同不等于总体执行间差异”的限制。
3. **通用anchor称谓修正。** 方法改用interface anchors；G+的编码器架构相同，但参数分别拟合。Protenix专属的缓存query anchors和公式中的下标q保留；OpenFold现场状态依赖与Atlas融合前LM条件仍明确区分。局部谱控制就近限定同输入、同anchors、同参数。
4. **首次配方展开。** 明写Train24训练384更新，Train96／Train384分别训练1536更新；后续表格仍可使用简写。
5. **补齐比较维度。** Full、Tiny接口、保均值旋转分别构成Native−Rotated比较；ESMC在Length48、ESM2在两个面板的交互分别指明。
6. **时序段直接解释差值。** All的Factor均值相对First更高，Ψ更小；All的Factor行内区间含零，G+行内差为负，两者之差仍为正。保留R1、重新训练和跨轮反传的范围。
7. **结论分开实验时序。** Fresh192首次固定模型检验为前瞻证据，之后在已观察Fresh192上的预算延长为后续研究；OpenFold Fresh96未建立交互的结果保留。K直接称为paired change in interaction。

讨论、相关工作及可编辑附录同步统一“收益之差”和有符号分数差。原Generic研究专属的query factors描述保留；未把既有协议或历史记录重新命名。

## 核验与交付范围

来源锁仍为v14，384个输入、1985个完整数值对象不变；既有正负结果及统计身份保留。
本轮新增的是表述和既有结果总览，不增加训练、预测、CIF评分或统计终点。
完整验证记录见
[本轮检查](../notes/writing_branch_20260922/draft_validation.terminology_revision_20260926.json)。

[当前PDF](../build/feedback_v49_terminology/paper.pdf)和
[匿名补充包](../build/anonymous_terminology_v49.zip)承接工作区v48草稿。
最终篇幅、全稿视觉验收和二进制匿名性检查仍留到收尾；本轮只检查改动区域和可执行的已有复算入口。

核验完成：全部既有分数复算、27项检查及两个CPU算子示例通过；编译无未定义引用或overfull。摘要、方法、时序比较、E2增益表与结论作了局部视觉检查。包内稿源与编译源一致，包含537文件；未检查页数。
