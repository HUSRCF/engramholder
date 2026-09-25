# EngramFold 证据整理仓库

**当前状态唯一入口：** [稿件状态](reports/manuscript_fill_20260922.md)。写作分支为 `main`，活跃来源锁 **v11**，共 **1,830** 个生成数值字段；入口区分已纳入证据、尚未纳入的独立工作及未完成复现交付。下列报告保留各自历史时点，不是远端队列的实时状态。

**最新记号修订：** [外积表示](reports/outer_product_notation_20260925.md)。decoder和残差展开统一使用列向量外积，并明确与实现一致的行优先展平。实验数值不变；编译及公式视觉检查通过。[当前PDF](build/feedback_v44_outer_product/paper.pdf)。

**新实验提案（未启动）：** [OpenFold注入时序与回收传播](reports/openfold_injection_schedule_plan_20260926.md)。按用户接受重训练、要求16并行和缩小规模的最新意见，推荐单旋转／三种子／三时序完整四格，共36组；纯训练估计7–10小时，含工程与评测约10–16小时，排队另算，跨轮梯度尚待实测。原24固定模型的推理方案另列；尚未提交GPU。

**最新独立结果审阅：** [同起点通道校准 A+B](reports/posttraining_calibration_review_20260925/interpretation.md)。33组续训、3,744次预测完成；本轮独立重建69项原比较及30项已有补充汇总。两项主要优势均未建立，OpenFold联合续训相对普通续训有次要平均收益。当前只进入审阅报告，尚未纳入正文、v11数值锁或匿名包；没有新GPU任务。

**最新补偿数值诊断：** [初值核对与短程定位](reports/compensation_numerical_diagnosis_20260925.md)。九对起点张量一致；同卡短程在第二步反传捕获差异，确定性cuDNN对照的两次八步路径一致。附录S.1区分这一局部定位与未解决的完整收益重复性；原204输入、1830分数字段不变。[当前PDF](build/feedback_v43_compensation_diagnosis/paper.pdf)供反馈。

**最新训练定义修订：** [源码核对与训练定义](reports/training_definition_revision_20260925.md)。补齐卷积编码器、Factor/G+层序、初始化及构造随机流，并明确普通旋转允许反射的QR采样规则。204输入、1830数值不变；构造核验、3项数值键测试、编译和局部视觉检查通过。[当前PDF](build/feedback_v42_training_definition/paper.pdf)供反馈，最终验收留到收尾。

**最新引用修订：** [引用修订与来源核验](reports/citation_revision_20260925.md)。补齐BLAST+、Holm、Adam、PDB、CATH、OFT与指数映射来源，四篇文献更新为正式会议版本；21条文献全部被引用。修正姓名和引用归属，区分CATH两类版本，并明确PDB原始下载日期未记录。204输入、1830数值不变；编译、引用检查及局部视觉检查通过。[当前PDF](build/feedback_v41_citations/paper.pdf)供反馈，最终验收留到收尾。

**前轮局部复审落实：** [94f1344复审回应](reports/review_followup_94f1344_response_20260925.md)。修正“未确立变化”的否定范围；主表用两种上标区分跨头／跨种子混合后端；三旋转总述限定核心四格，明确OpenFold预设Holm校正，摘要首次定义Factor／G+。1,830字段、204输入不变；3项数值键测试、编译和主表局部视觉检查通过。[当前PDF](build/feedback_v40/paper.pdf)供反馈，最终验收仍留到收尾。

**前轮逻辑审阅落实：** [六项修改与review5-2／review6-2](reports/logic_review_response_20260925.md)。开头明确固定训练预算；Fresh96与Fresh192同级且紧接观察面板发现；主文补偿表并列两次三项收益和区间；附录新增执行条件及未知差异表，复现声明同步九组完整重训练。v39编译与局部检查属于该轮历史快照。

**前轮结果整合：** [Fresh192与E2完整重训练](reports/fresh192_retraining_integration_20260925.md)。新目标主要Ψ=+0.02515；E2重训练旋转收益保留，但额外缓解T_C=+0.00065、区间跨零。两次执行分开，原1617字段不变。v38匿名包的27项测试及篇幅记录仅属于该历史快照，未包含本轮文字修改。

**前轮稿件与匿名材料同步：** [修订与同版本验收](reports/submission_sync_20260925.md)。摘要与引言统一为“训练后的旋转代价＋共享补偿”；Results依次呈现参数化比较、补偿干预、迁移与竞争边界。v10的1,617字段、121输入不变；v35匿名包通过23项测试，并从包内同一份源码编译PDF，正文9页、全文38页。正文全部页面和重点附录逐页检查，其余附录作总览检查。[当前PDF](build/submission_v35/paper.pdf)与[匿名包](build/submission_v35/supplement.zip)均为本机构建产物，不随Git上传。本轮没有启动新实验。

**前轮review6判断：** [独立验证与完整训练切片](reports/review6_response_20260925.md)。现有计划补可选R2b：单种子／单旋转、四臂各1536步的完整E2训练到评分切片，共6,144更新。它区别于32步smoke及原九组重训练，未自动加入默认预算、未启动。

**前轮review5落实：** [贡献定位与内部编号整理](reports/review5_response_20260925.md)。引言突出重训后的四格比较和Factor内补偿；正文及诊断表使用描述性名称。数值、来源、图和作者摘要保持不变；按最新指示，最终稿统一检查页数和更新匿名包。

**前轮旧评审落实：** [review3／review4与E3状态核对](reports/review34_response_20260925.md)。摘要补E2效应量，主表新增Fresh96独立行；小提琴与均值区间分列，E2表明确区间及舍入口径，CPU示例输出运行环境。1,522字段不变，主文9页；匿名包v30通过。

**已完成E3正式整合：** [锚点干预、完整分数与来源包](reports/anchor_intervention_integration_20260925.md)。Discussion与附录T现已纳入；1,824条分数的42项对比复算一致，预期平均帮助未获支持。E2与E3分别使用 $B_R^C$、$B_R^{\mathrm{anchor}}$。旧1,522字段完全保留；当时的v34包已由顶部v35同版本检查取代。先前[独立审阅](reports/e3_anchor_review_20260925.md)保留历史状态，不再作为待整合项。

**前轮图表呈现修订：** [Table 1／4、四位均值与交互小提琴图](reports/figures_tables_revision_20260925.md)。左图采用Set2配色；右图引入逐目标分布。168项均值仅改变显示精度，全部原值、效应区间与v9来源锁不变；均值／区间分列和Fresh96主表行在本轮补入。

**实施计划已更新：** [P192与E2重放](reports/protenix_fresh_e2_replay_plan_20260925.md)。P192、R2、R3均已完成并纳入；R1公共新特征完整重放仍待交付，不自动启动R2b或新增实验。

**本轮评审落实：** [E2正文、目标异质性与可运行干预](reports/e2_presentation_review_20260924.md)。新增独立E2小节、三行绝对分数与明确标为事后描述的目标分布；原1,518字段不变。[共享C示例](reproducibility/compensation/README.md)已运行，属于实际算子／优化器检查，非完整结构训练复现。

**前轮完成整合：** [E2结果、统计身份与两份评审落实](reports/e2_completed_review_20260924/interpretation.md)。E2平均旋转收益与差距缩小获支持，只有一个旋转单独建立缩小；E1预测失败保留。已完成的带符号置换及Dev8三节点曲线一并进入正文／附录和v9来源锁。

**适配器图与前轮评审：** [对71370ad评审的回应及已完成控制核对](reports/review_71370ad_response_20260924.md)。新增[适配器双支结构图](figures/construction.pdf)与[统一交互主图](figures/interaction.pdf)，主文保留Fresh96边界。该报告保留先入MD的历史时点；带符号置换及Dev8学习曲线现已进入v9。新Protenix面板仍仅为候选。

**此前已正式纳入：** [Protenix／AtlasFold完整四格v7整合](reports/diamondhill_fourcell_integration_20260924.md)。主表统一15行，Fresh96进入交互主图；A66主要终点与未校正后续交互分开。旧532字段及v6锁保持不变。

**E1历史初次审阅：** [共享重建未预测重训方向代价](reports/e1_prediction_review_20260924.md)。主要ρ=0.02381、精确单侧p=0.48839；E2按独立协议继续。当时先进入MD；现已正式纳入v8正文／数值锁。

**历史方法与统计范围修订：** [Atlas anchor分类与A66统计身份](reports/atlas_anchor_statistical_scope_revision_20260924.md)。方法表已区分静态写入、适配历史反馈、当轮LM条件；A66的主要量与未校正后续交互分别记录。主文编译仍为9页；本轮没有导入A66新数值。

**2026-09-24 已完成：** [Fresh96与训练交互变化](reports/fresh96_training_psi_integration_20260924.md)及[单目标OpenFold完整预测](reports/openfold_single_reproduction_20260924.md)。Fresh96的核心交互未建立；结果已进入摘要与主文。旧449数值不变，新增83项；[可执行预测入口](reproducibility/openfold_single/README.md)包含发布适配器、FASTA、CIF评分与环境说明。

**2026-09-24收尾证据：** [DiamondHill A66、Protenix Train96四格及三底座观测](reports/diamondhill_final_evidence_review_20260924.md)。A66与G12均完成统一评分；Protenix Train96及Train384／ESM2、ESMC的完整交互有正向区间，Atlas边界保留；几何／传播18/18已齐。当时先进入MD；现已将A66及Train96四格纳入v7。几何／传播数字仍是独立报告。

**机制小试首轮审阅：** [共享重建与OpenFold几何／传播](reports/shared_reconstruction_openfold_review_20260924.md)。六组共享重建留出误差约0.82–0.84，属于优于零输出的部分重建。该文保留当时OpenFold先完成的历史时点，完整三底座更新见上方收尾报告。

**架构分析：** [条件依赖能解释到哪一步](reports/architecture_dependence_analysis_20260924.md)。核对三底座实际路径，纠正Atlas首接口的历史反馈归类，区分局部可达性、下游读取与有限预算学习；该文件保留设计时点，执行结果见上方首轮审阅。

**补充实验设计：** [既有曲线／置换与共享旋转重建计划](reports/architecture_followup_plan_20260924.md)。共享重建已由原线程完成；原设计保留并追加状态，不重复提交既有任务或修改执行锁。

**串行研究设计与执行衔接：** [前瞻预测→补偿自由度／状态干预→跨底座验证](reports/prospective_orientation_series_20260924.md)。E1／E2／E3现已完成并纳入v10；E3主要假说未获支持，E4未启动。原设计保留历史身份。本写作轮没有提交或修改任务。该系列的Factor预测／补偿对比不替代Ψ的新目标确认。

[执行前评审已落实](reports/architecture_followup_review_response_20260924.md)：384→1536归因区间统一，补零输出参照和幅度／夹角分解，固定完整序列编码与单次旋转约定；不增加拟合任务，不改变既有执行锁。

[本轮收尾](reports/fixed_budget_status_followup_20260923.md)：在训练规模差值旁补充四格绝对均值均提高的事实；保留固定预算解释及未建立收敛的限制。

[历史：7c701ed盲审落实](reports/blind_review_7c701ed_response_20260923.md)：补短链数据定义、两层控制边界及核心补充指标；该轮来源锁v5，449字段，主文9页。

[历史：A组审阅落实](reports/esmc_review_response_20260923.md)：三套OpenFold交互并列主文，补齐模型引用及ESMC范围图；443数值不变，本轮实测主文9页。

**9月23日 OpenFold A组更新：** [ESMC完整四格及核验报告](reports/openfold_esmc_A_integration_20260923.md)。33组新训练与4,752次预测已完成；绝对质量提高，但ESMC下方向效应与头类型×旋转交互未建立。该次整合来源锁为v4（当前v10）；B组及DiamondHill后续矩阵未纳入本稿，此处不判断其实时进度。

**历史整合记录（2026-09-23）：** 完整推理四格与v4机制边界、近邻定位及复现范围说明已整合；该轮编译记录及视觉检查范围见相应报告。当前版本以顶部状态入口为准。

- [正式审读修改与数值键保护](reports/editorial_review_response_20260923.md)；[作者提交清单](notes/writing_branch_20260922/author_submission_checklist.md)。
- [9月23日稿件更新](reports/full_inference_manuscript_update_20260923.md)与[当前交付边界](reports/submission_priorities_20260924.md)。
- [最新PDF](build/iclr2027_conference.pdf)（本机构建产物，不随Git上传）；[表图生成脚本](scripts/build_paper_assets.py)与[数值来源映射](generated/cell_sources.json)。
- [当前章节与写作任务](notes/writing_branch_20260922/skeleton_files.md)：入口 `iclr2027_conference.tex`，正文在 `sections/`，附录在 `appendices/`。
- [历史主线与Introduction候选](notes/writing_branch_20260922/structure_and_opening.md)：参数化×方向交互 → 方法构造 → 迁移与边界。

- [完整推理机制干预](reports/mechanism_evidence/full_inference_cross_20260922/README.md)：720次同后端四格预测，匹配整体范数后原生来源优势仍在。
- [机制证据归档：v4](reports/mechanism_evidence/gradient_hpc3_v4_20260920/README.md)：正式协议、逐目标与精简逐条件数据、独立复算，以及后来完成的完整推理干预之历史候选；保留源码哈希差异和数值边界。
- [Protenix匹配G+最终结果](reports/protenix_gplus384_results_20260922.md)：3组训练与432次预测全部完成，补齐Train384两个面板的G+；[三项审阅意见](reports/reviewer_priorities_20260922.md)保留决策过程。
- [9月22日完成结果更新](reports/results_update_20260922.md)：OpenFold旋转G+交互、AtlasFold完整边界与ESMFold2最终评分。
- [证据审阅报告](reports/evidence_review.md)：核心问题、直接核对的数字、可支持与不可支持的解释，以及仍待补齐的记录。
- [数学对象与实现边界](reports/math_contract.md)：query/ESM信息来源、Full/Tangent、冻结与旋转位置、D=508.5逐项来源及同构造交互。
- [稳健性与暴露审计](reports/robustness_audit.md)：完整种子/旋转、目标差异分布、锁定时序、累计暴露限制；附[可复算脚本](scripts/audit_evidence_coverage.py)。
- [工作方式](notes/workflow.md)：先核验证据；当前已授权章节与占位写作。
- [可吸收旋转的 G+ 对照](reports/rotatable_gplus_control.md)：代码前提、四格交互、九组最小矩阵与历史计时；运行进度见该独立文件。
- [证据快照](evidence/)：已经拷入的结果 JSON，便于逐项核对；不包含全部原始 CIF。
- [未经用户审阅的旧草案](notes/unreviewed_writing_20260921/)：仅归档，不能当作正式稿。

仓库根目录为当前ICLR稿件入口，原模板另行备份。现有 source_*.md 是此前助手草稿的快照，不代表用户已认可其文字或论证。

此处整理不会改变实验配置、统计终点或训练队列。公开写作快照位于 `main`；旧本地分支 `writing/native-update-alignment` 仅为历史。旧草案仅供归档，最终Protenix G+已在MD报告、论文主表与配对比较中完成整合；该次历史整合使用v2来源锁，当前为v10。见[整合记录](reports/protenix_gplus384_manuscript_integration_20260922.md)。
