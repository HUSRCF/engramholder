# Experimental design：组织与统计范围修订

2026-09-26，基于 `5809c08`。用户确认按最新版落实四项剩余意见。论文源码仅修改 `sections/03_design.tex`；开头的信息边界、配方表、final-recycle句、交互公式与原符号，以及已修订的R1时序表行均保持不变。

## 四项落实

1. 数据段提前至训练预算之前，先定义嵌套Train24/96/384和Dev8，再介绍评测面板。Fresh96/Fresh192并列说明固定模型、四层各24/48目标、实际长度范围与唯一主要交互。继承Train24/Dev8不套用新候选日期截止；ID排除未被升级为家族或预训练隔离。补偿、重建和Dev8研究身份移至独立统计／研究身份段。
2. 公式前补充H、S、训练种子、旋转和N的定义，限定三种子／三旋转比较；原公式与R1特例不改写。
3. 目标bootstrap总述限定为结构分数对比，说明按研究协议保留分层方式；Fresh192和预算扩展按四层重采样，未把Fresh96误写为分层bootstrap。八旋转重建关联检验单独说明旋转单位和精确排列方法。Holm只校正OpenFold特征研究的三项预设次要检验p值，正文及表格明确置信区间未作多重比较校正。
4. tangent主要比较明确为该构造下Native−Rotated；“target-label selection”改为不使用评测标签选择checkpoint或推理样本。

## 核对与交付

独立只读复核与附录一致。v14的384输入哈希、1985完整数值对象以及generated文件和图均未改变。无新训练、预测、评分或统计终点。

编译无未定义引用或overfull，设计节所在第4–6页及与Results的衔接作了局部视觉检查。匿名材料沿用既有27项检查与两个CPU合成输入示例，均通过；包内69份编译相关源码与实际编译源码字节一致，537项manifest哈希均匹配。检查在X570的BIO环境使用CPU，不是OpenFold完整训练复现。最终页限、全稿视觉及二进制匿名性仍留到统一收尾。

- [当前PDF](../build/feedback_v51_design/paper.pdf)
- [匿名补充包](../build/anonymous_design_v51_final.zip)
- [验证记录](../notes/writing_branch_20260922/draft_validation.design_revision_20260926.json)
