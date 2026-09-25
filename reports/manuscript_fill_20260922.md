# 当前稿件状态（2026-09-25）

此处是写作仓库的唯一当前状态入口，记录已纳入稿件的证据，不报告远端队列实时进度。

| 项目 | 当前状态 |
|---|---|
| 稿件版本 | `main`保留匿名版；在 `6f22973` 的E2整合正文上，压缩Table 1／4、统一绝对分数四位显示，并将交互图改为逐目标小提琴＋均值区间。科学结果不变，作者资料只存本地。 |
| 本轮图表检查 | [修订说明](figures_tables_revision_20260925.md)；[验证记录](../notes/writing_branch_20260922/draft_validation.figures_tables_20260925.json)。1,522原值／来源不变，168项仅改显示格式；Table 4仍为15行／75个分数。主文9页、全文35页，无未定义引用／overfull；第4、6、8、9页已视觉检查，非全页人工验收。 |
| 活跃来源锁 | [paper_sources.v9.lock.json](../notes/writing_branch_20260922/paper_sources.v9.lock.json)，93份输入、1,522字段；前轮已增4项E2目标分布描述，本轮不增字段或改变原值／来源，仅修改168项显示格式。v8及此前旧锁保留。 |
| 已纳入结果 | Protenix方向／长度／G+及两规模完整四格；OpenFold两规模／两PLM四格；AtlasFold两PLM四格；Fresh96；范数干预；E1阴性预测；E2共享补偿；带符号置换；Dev8三节点曲线。 |
| 统计身份 | E2平均BR、TC在原书面设计与代码中预设，属于已观察面板机制对比。置换C96-B Ψ为该后续研究主要量。A66各底座主要量仍为C96-B ESMC Factor−Rotated，其Ψ等未统一校正；这些均不替代Fresh96。 |
| 已完成但独立保留 | 六组共享重建和三底座18次几何／传播数值仍为独立报告；接口反馈分类已进入方法。[E3验收审阅](e3_anchor_review_20260925.md)：9组／864次新预测完成，本轮从保存分数复算42项对比一致；现场anchor补偿的主要假说未获支持，尚未纳入正文／v9。 |
| 其他独立工作 | B组、SGDM未纳入本稿，本入口不推断其队列进度。E3已收口，不追加配置；E4未启动。本轮未核实时队列。 |
| 候选实施计划 | [Protenix新目标与E2重放](protenix_fresh_e2_replay_plan_20260925.md)：P192固定模型4,800预测；E2完整推理1,824预测＋有界训练入口验收，可选九组完整重训练分别计数。本轮未选目标、未提交新任务。 |
| 前轮评审落实 | [E2正文与复现](e2_presentation_review_20260924.md)；[验证记录](../notes/writing_branch_20260922/draft_validation.e2_presentation_20260924.json)。四项目标分布为事后描述，原E2终点不变。主文9页、全文35页，无未定义引用／overfull；修改页已视觉检查，非全页人工验收。 |
| 前轮检查 | [E2及评审落实](e2_completed_review_20260924/interpretation.md)；[验证JSON](../notes/writing_branch_20260922/draft_validation.completed_controls_v9.json)。数值复算与18项CPU测试通过；主文9页、全文35页，无未定义引用／overfull。修改正文与新增附录已视觉检查，非全页人工验收。 |
| 匿名材料 | 本地 `build/anonymous_artifact_v29.zip`，212文件；1,522数值一致，18项测试、既有算子检查及新增共享C三条件CPU运行通过。包含本轮最终图表和显示格式；已知身份字符串扫描通过，作者资料不在当前材料中。旧包保留历史身份。 |
| 复现范围 | 一个固定OpenFold模型／目标已完成隔离环境FASTA→特征→完整CIF→主指标评分；新增封存共享C／Factor／优化器的合成输入CPU运行；全矩阵重预测／E2完整重训练复现未建立。 |

主文围绕“方向效应是否存在、是否因参数化而异、能否用针对性干预缓解、是否等于更好的适配器”组织。适配器双支图保留输入、冻结与训练边界、旋转位置及独立基线；主图并列Protenix、OpenFold、AtlasFold与Fresh96，未用Factor阳性替换Fresh96主要交互。

E2支持平均有用缓解：BR=+0.02973、TC=+0.01684、Native自身改善+0.01290；单旋转和TM-score边界紧邻报告。它没有恢复E1的预测假说，也没有建立新模型选择规则、完全消除方向差距或严格轨迹等价。E2补偿后的剩余差距事后描述只保留在MD／审计，不增加原正式检验。

E1共享重建预测失败、置换长链交互未建立、Dev8曲线未证明收敛等均保留。正文新增内容只使用完成记录，没有新训练、面板选择或执行锁改动。

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
