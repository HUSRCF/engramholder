# 当前稿件状态（2026-09-24）

此处是写作仓库的唯一当前状态入口，记录**已纳入稿件的证据**，不报告远端队列实时进度。

| 项目 | 当前状态 |
|---|---|
| 稿件版本 | `main`；本轮修订基线 `11a1e3b`，包含本状态更新的提交是本轮稿件版本。可用 `git log -1 --format=%H -- reports/manuscript_fill_20260922.md` 查询精确提交。 |
| 活跃来源锁 | [paper_sources.v6.lock.json](../notes/writing_branch_20260922/paper_sources.v6.lock.json)；532 个生成数值字段（旧449项及来源映射未变）。旧锁保留，不覆盖历史记录。 |
| 已纳入的核心与范围结果 | Protenix 原始 Factor/Generic、原生/旋转、Tiny、Train96/384、Length48、匹配 Train384 G+；OpenFold Train96 与 Train384/ESM2、Train384/ESMC 完整四格；已完成的 AtlasFold 原始适配矩阵与系统参照；Fresh96完整2,400预测及主交互未建立的结果。 |
| 已纳入的边界与审计 | 完整推理范数交换及局部/共享迁移阴性结果；嵌套数据与成员定义核验；训练预算报告的“1536 步收敛未建立”限制；头内同谱与跨头匹配边界；ESM2 Train384−Train96 的完整交互变化后续分析。 |
| 尚未纳入的独立工作 | B 组、SGDM、DiamondHill 后续新矩阵。此列表表示未进入本稿的证据范围，**不表示这些队列此刻仍未完成**；本轮未查询其实时状态或导入结果。 |
| 本轮修改与检查 | [Fresh96与训练交互整合](fresh96_training_psi_integration_20260924.md)、[单目标预测复现](openfold_single_reproduction_20260924.md)；[机器检查记录](../notes/writing_branch_20260922/draft_validation.fresh96_repro.json)。 |
| 匿名材料范围 | 本轮候选为本地 `build/anonymous_artifact_v14.zip`，包含分析、算子检查及单目标OpenFold预测入口；具体检查结果见本轮验证。 |
| 复现交付范围 | 一个固定OpenFold模型、一个184残基目标，在新隔离Python环境完成“序列→重算PLM特征→完整CIF→主指标评分”。全矩阵重生成、完整重新训练及跨硬件保证仍未建立；本轮不是全稿视觉验收。 |

本轮导入已锁定Fresh96完整结果，计算明确标注的后续交互变化，并完成单目标工程复现；无新增训练、无修改原确认终点、无推断其他队列结果。

**2026-09-24追加MD审计：** [架构条件依赖分析](architecture_dependence_analysis_20260924.md)区分算子可达性、回收反馈与学习阶段，并纠正旧数学笔记的Atlas归类。该分析未写入论文机制结论、未新增实测字段或计算任务；活跃数值锁仍为v6。

## 历史快照（以下不代表当前状态）

- **2026-09-23，11a1e3b：** [固定预算与状态入口收尾](fixed_budget_status_followup_20260923.md)，v5、449字段；当时Fresh96尚未导入，单目标完整预测尚未执行。

- **2026-09-23，v5：** [7c701ed 盲审落实](blind_review_7c701ed_response_20260923.md)：短链数据定义、控制范围、六个既有补充统计字段；共449字段，验证记录主文9页，局部检查附录数据表。
- **2026-09-23，v4：** [OpenFold ESMC A组整合](openfold_esmc_A_integration_20260923.md)及[呈现修订](esmc_review_response_20260923.md)，共443字段。
- **2026-09-23，正式审读：** [贡献压缩、数值键保护与混合干预边界](editorial_review_response_20260923.md)，当轮未做最终PDF视觉检查。
- **2026-09-23，v3：** [完整推理来源×范数及局部机制v4边界](full_inference_manuscript_update_20260923.md)，匿名候选v6；“机制v4”与“来源锁v3”是不同版本对象。
- **2026-09-22，v2：** [匹配 Protenix G+整合](protenix_gplus384_manuscript_integration_20260922.md)，匿名候选v4，当轮主文8页、全文13页。

## 历史快照：2026-09-22 初次填稿（原始记录）

以下包括当时的 Pending、页数和“尚未推送”等表述，仅描述该次初稿；当前状态以上表为准。

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
