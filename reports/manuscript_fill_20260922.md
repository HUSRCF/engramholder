# 当前稿件状态（2026-09-26）

此处是写作仓库的唯一当前状态入口；记录已纳入证据，不报告远端实时队列。

| 项目 | 当前状态 |
|---|---|
| 当前反馈版 | `main`匿名；本轮基于`5c3952d`补入已完成 Atlas 原生／ESMC probe 边界。作者资料不提交。 |
| 科学更新 | [联合整合报告](fresh192_retraining_integration_20260925.md)：Fresh192唯一主要Ψ得到支持；E2重训练保留旋转收益，但额外缓解T_C未重新建立。原结果与重训练分开，不合并为新种子。 |
| 稿件呈现 | [Atlas probe 收口](atlas_probe_integration_20260926.md)：主文、讨论与主图不变；现有附录增加主要读出比较及ESMC传播短段。信息互补性、下游利用与优化仍是候选，不承诺新的接口训练。 |
| 来源与数值 | [v13锁](../notes/writing_branch_20260922/paper_sources.v13.lock.json)，349输入、1860字段。旧266输入哈希及1851完整数值对象不变；新增83输入、9字段。旧锁保留。 |
| 直接核验 | [本轮验证](../notes/writing_branch_20260922/draft_validation.atlas_probe_20260926.json)：九个probe的72条最终Dev分数、全部六项预定比较和六个预测距离变化复算一致；不包含重新训练、提取缓存或结构质量重评分。 |
| 当前PDF | [反馈版PDF](../build/feedback_v46_atlas_probe/paper.pdf)。最终篇幅与全稿验收留到收尾。 |
| 匿名材料 | [v46补充材料](../build/anonymous_atlas_probe_v46.zip)同步本轮稿源及有界证据；全稿、二进制匿名性及最终篇幅仍待统一验收。 |
| 已完成任务 | P192、R2六条件／384更新、R3九组／13824更新／864预测均完成。相关协议、分数、实现身份与收据已纳入；不再列为待启动。 |
| 已审阅、尚未纳入 | [同起点通道校准A+B](posttraining_calibration_review_20260925/interpretation.md)：33组／3744预测完成，69项原比较及30项已有补充汇总独立复算一致。两项主要优势未建立，OpenFold联合续训保留次要平均收益。仍未纳入正文、v13或匿名包；本轮新增的是独立的 Atlas Native-only 校准。 |
| 注入时序、报告阶段 | [第一阶段](openfold_injection_schedule_phase1_20260926.md)7,008轨迹／14,112评分／0失败，261项复算通过；All−First主要交互差未建立，First自身交互保留。只进入MD，未纳入正文、v13或匿名包。 |
| 注入时序、重训练阶段 | 执行侧已恢复72组，第一波36含全部Native与R1，第二波36补R2/R3；前轮只读快照：HPC3校准652881运行，652882正式队列等待校准，HPC2工程12865355_0/1均RUNNING。本轮未查队列；工程不等于正式放行，最终状态依验收收据。 |
| 未完成交付 | R1公共FASTA／新特征完整重放仍未完成。本次R3复用原环境与缓存、固定原无C基线；R2b不自动追加。 |
| 其他范围 | E3完成且阴性，不追加旋转；E4未启动。B组／SGDM等独立工作未据此判断进度。本轮补偿短程诊断已完成；未启动新的完整训练矩阵。 |

Fresh192支持固定Protenix／ESMC配方的新目标交互，不保证普遍参数化定律或家族／预训练隔离。E2原执行支持平均额外缓解，同种子完整重复未重新建立该更强结论；两者不能互相替代。详细统计身份与执行边界见整合报告。

## 同日较早历史记录（其“尚未导入／v6”描述已被当前状态取代）

v6结果整合已导入锁定Fresh96完整结果、明确标注的后续交互变化，并完成单目标工程复现；后续方法文字修订不改变这套数值来源。

**2026-09-24方法文字修订：** [Atlas anchor分类与统计身份](atlas_anchor_statistical_scope_revision_20260924.md)。三类接口已进入正式方法表和附录；局部rank／完整共享映射、非零注入／预测变化／质量收益的区别已明确。A66统计角色写入整合约束，数字尚未导入。CPU编译主文9页、全文23页，修改页视觉检查通过，来源锁与532字段不变；[验证记录](../notes/writing_branch_20260922/draft_validation.anchor_scope.json)。

**2026-09-24追加MD审计：** [架构条件依赖分析](architecture_dependence_analysis_20260924.md)区分算子可达性、回收反馈与学习阶段，并纠正旧数学笔记的Atlas归类。该分析未写入论文机制结论、未新增实测字段或计算任务；活跃数值锁仍为v6。

**2026-09-24最新MD结果审阅（未纳入正文）：** [DiamondHill收尾与第二底座交互](diamondhill_final_evidence_review_20260924.md)。A66及Protenix Train96 G12完成，Protenix两种规模／两种特征的完整四格不再缺项；Atlas阴性边界保留。三底座几何／传播18/18已齐。当前正式稿仍为v6、532字段，下一次正式整合须同步来源锁、表图与主张，不能把此MD误当作已经写入正文。

**较早同日审阅：** [共享重建与OpenFold部分观测](shared_reconstruction_openfold_review_20260924.md)保留当时的部分完成状态。共享重建为优于零输出的部分补偿，未达到低误差；传播测的是预测变化，不是质量提高。这些边界不被本次完成度更新改写。

**历史设计与执行前修订：** [架构解释实验计划](architecture_followup_plan_20260924.md)及[四项意见落实](architecture_followup_review_response_20260924.md)保留原线程1,160次曲线、18组置换与六组重建的设计身份，以及零输出参照和源码约定。当时六组为未执行候选；现已完成的范围以上方结果审阅为准。既有科学锁未改写。

**2026-09-24串行研究设计与执行衔接：** [前瞻预测与主动干预方案](prospective_orientation_series_20260924.md)。原线程已基于9d87049形成独立执行协议并提交E1；E2是按其阶段门槛执行的独立增容干预，E3/E4保留后续锁。本写作轮只读取协议和提交元数据，不读取E1科学分数来修改预测规则，也不重复提交；正文与v6数值不变。

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
