> **2026-09-23 正式审读更新：** 贡献与摘要已压缩，混合干预范围已补充，数值缺失键保护通过测试；正式稿编译成功，不做最终PDF视觉检查。见[修改记录](editorial_review_response_20260923.md)。

> **2026-09-23 最新状态：完整推理来源×范数干预及 v4 边界已进入正文/附录。** 活跃证据锁 v3，匿名材料候选 v6；编译成功，PDF 排版由作者在 Overleaf 检查。见[本轮更新](full_inference_manuscript_update_20260923.md)。以下保留此前填稿记录。

> **最新更新：匹配 Protenix G+ 已完成论文整合。** 两个 Pending 已填入；当前证据锁为 v2、匿名候选包为 v4，正文仍 8 页、全文 13 页。详见[整合记录](protenix_gplus384_manuscript_integration_20260922.md)。以下保留初次填稿时的历史状态与验证结果。

# 2026-09-22：从章节骨架到有证据支撑的正文初稿

本次用户要求不再增加占位或实验，按现有证据正式填稿。工作在写作仓库main本地完成；
尚未commit/push。主线训练与实验锁未修改，未重新评分CIF或改变统计终点。

## 已完成

- 摘要、Introduction、方法、Experimental design、Results、Discussion、Related work、Conclusion均已替换为正式草稿文字。
- OpenFold直接交互与Protenix原始锁定方向/长度研究并列承担论证；明确交互为已观察面板后续研究。
- 主表填入全部已完成均值，Unadapted标签与表征差异、Train96/Train384及1536步表注明确；只有新Protenix G+两格Pending，Train96缺项为---。
- Native−G+配对区间在正文邻近表呈现，Atlas/OpenFold未成立条件保留在主表与范围图。
- 三张矢量图已生成：算子和四格控制、OpenFold交互、范围与长度效应。实际预览后调整了字级及溢出。
- 10条实际参考文献已建立并在正文引用。来源核对表在notes/writing_branch_20260922/reference_sources.json；这不是完整新颖性检索认证。
- 附录已写入算子细节、训练配方、筛查/时序、计分、完整方向表、机制边界与工程恢复记录。

## 单一数值来源

运行：`python scripts/build_paper_assets.py`。

输入快照由notes/writing_branch_20260922/paper_sources.lock.json固定；输入变化会使脚本报错，
不能静默重建锁。输出generated/numbers.tex、主表/配对表/范围表和三张图。
generated/cell_sources.json记录每个数值的源JSON字段、聚合操作、原值与显示值。

本次检查348个系统×指标均值、144个逐目标OpenFold交互；所有差异<1e-12。
129个数值字段共同供正文/图/表使用。没有重做bootstrap、选新指标或改变缺失处理。
一些背景计数与配置常数仍在正文直接说明，依据方法/锁，不把它们冒称为新统计结果。

## 编译与页数

使用DiamondHill的CPU LaTeX工具编译，不占训练GPU。当前主文结束于第8页；
含声明、参考文献和附录共13页。无undefined references/citations，无overfull box；
8条underfull排版警告主要来自窄配置列，不是编译失败。
官方sty/bst和math_commands.tex与已提交原版字节一致，匿名模式保留。

[PDF](../build/iclr2027_conference.pdf)；[验证记录](../notes/writing_branch_20260922/draft_validation.json)。

## 复现与匿名化候选包

[候选ZIP](../build/anonymous_artifact_v3.zip)含论文源、固定评分快照、图表生成脚本、
真实adapter/operator源码快照、固定配置与CPU检查入口；不含.git历史、个人README、
旧草稿或原始集群日志。个人仓库URL、已知用户名与绝对用户路径已清理；匿名化后
重新生成全部图表并核对129个数值保持一致。

真实算子的5项检查通过：Factor/G+零初始化、Full/Tangent旋转零回放、OpenFold G+
自由输出层吸收旋转；后者最大绝对误差2.22e-16。

**这是分析和算子级复现材料，不是完整重训练包。** 未打包预训练权重、全量结构/PLM缓存、
全部checkpoint或经测试的一键训练环境。模型依赖和发布许可仍须按来源处理。
源码快照hash与历史执行锁分开记录，不声称每个当前文件都是所有历史run的原执行文件。
作者仍需人工审核匿名化、AI使用/伦理声明及最终提交内容；已知字符串扫描不是匿名性证明。

## 尚待收口

1. 主线完成Protenix匹配G+后，按已有锁的Confirm96-B主要/Length48次要比较更新对应两格和一小段文字。
2. 作者逐段审阅主张、篇幅、文献定位和声明；当前是可审阅初稿而非可直接提交的终稿。
3. 若要求完整训练复现发布，需要另行整理上游环境/数据入口/权重获取与运行验证，当前不提前承诺。

本轮没有新训练、没有新面板或新比较。远端main仍为此前推送快照，当前填稿尚未推送。
