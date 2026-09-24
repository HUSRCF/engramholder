# EngramFold 证据整理仓库

**当前状态唯一入口：** [稿件状态](reports/manuscript_fill_20260922.md)。写作分支为 `main`，活跃来源锁 **v7**，共 **1,135** 个生成数值字段；入口区分已纳入证据、尚未纳入的独立工作及未完成复现交付。下列报告保留各自历史时点，不是远端队列的实时状态。

**本轮已正式纳入：** [Protenix／AtlasFold完整四格v7整合](reports/diamondhill_fourcell_integration_20260924.md)。主表统一15行，Fresh96进入交互主图；A66主要终点与未校正后续交互分开。旧532字段及v6锁保持不变。

**E1另行审阅：** [共享重建未预测重训方向代价](reports/e1_prediction_review_20260924.md)。主要ρ=0.02381、精确单侧p=0.48839；E2按独立协议继续。此结果先进入MD，未纳入v7正文／数值锁。

**历史方法与统计范围修订：** [Atlas anchor分类与A66统计身份](reports/atlas_anchor_statistical_scope_revision_20260924.md)。方法表已区分静态写入、适配历史反馈、当轮LM条件；A66的主要量与未校正后续交互分别记录。主文编译仍为9页；本轮没有导入A66新数值。

**2026-09-24 已完成：** [Fresh96与训练交互变化](reports/fresh96_training_psi_integration_20260924.md)及[单目标OpenFold完整预测](reports/openfold_single_reproduction_20260924.md)。Fresh96的核心交互未建立；结果已进入摘要与主文。旧449数值不变，新增83项；[可执行预测入口](reproducibility/openfold_single/README.md)包含发布适配器、FASTA、CIF评分与环境说明。

**2026-09-24收尾证据：** [DiamondHill A66、Protenix Train96四格及三底座观测](reports/diamondhill_final_evidence_review_20260924.md)。A66与G12均完成统一评分；Protenix Train96及Train384／ESM2、ESMC的完整交互有正向区间，Atlas边界保留；几何／传播18/18已齐。当时先进入MD；现已将A66及Train96四格纳入v7。几何／传播数字仍是独立报告。

**机制小试首轮审阅：** [共享重建与OpenFold几何／传播](reports/shared_reconstruction_openfold_review_20260924.md)。六组共享重建留出误差约0.82–0.84，属于优于零输出的部分重建。该文保留当时OpenFold先完成的历史时点，完整三底座更新见上方收尾报告。

**架构分析：** [条件依赖能解释到哪一步](reports/architecture_dependence_analysis_20260924.md)。核对三底座实际路径，纠正Atlas首接口的历史反馈归类，区分局部可达性、下游读取与有限预算学习；该文件保留设计时点，执行结果见上方首轮审阅。

**补充实验设计：** [既有曲线／置换与共享旋转重建计划](reports/architecture_followup_plan_20260924.md)。共享重建已由原线程完成；原设计保留并追加状态，不重复提交既有任务或修改执行锁。

**串行研究设计与执行衔接：** [前瞻预测→补偿自由度／状态干预→跨底座验证](reports/prospective_orientation_series_20260924.md)。原线程已按独立协议提交E1，E2保留其阶段门槛；本写作轮没有提交或修改任务。新系列主要研究Factor方向代价，不替代Ψ的新确认，尚不能计作完成结果。

[执行前评审已落实](reports/architecture_followup_review_response_20260924.md)：384→1536归因区间统一，补零输出参照和幅度／夹角分解，固定完整序列编码与单次旋转约定；不增加拟合任务，不改变既有执行锁。

[本轮收尾](reports/fixed_budget_status_followup_20260923.md)：在训练规模差值旁补充四格绝对均值均提高的事实；保留固定预算解释及未建立收敛的限制。

[历史：7c701ed盲审落实](reports/blind_review_7c701ed_response_20260923.md)：补短链数据定义、两层控制边界及核心补充指标；该轮来源锁v5，449字段，主文9页。

[历史：A组审阅落实](reports/esmc_review_response_20260923.md)：三套OpenFold交互并列主文，补齐模型引用及ESMC范围图；443数值不变，本轮实测主文9页。

**9月23日 OpenFold A组更新：** [ESMC完整四格及核验报告](reports/openfold_esmc_A_integration_20260923.md)。33组新训练与4,752次预测已完成；绝对质量提高，但ESMC下方向效应与头类型×旋转交互未建立。该次整合来源锁为v4（当前v7）；B组及DiamondHill后续矩阵未纳入本稿，此处不判断其实时进度。

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

此处整理不会改变实验配置、统计终点或训练队列。公开写作快照位于 `main`；旧本地分支 `writing/native-update-alignment` 仅为历史。旧草案仅供归档，最终Protenix G+已在MD报告、论文主表与配对比较中完成整合；该次历史整合使用v2来源锁，当前为v7。见[整合记录](reports/protenix_gplus384_manuscript_integration_20260922.md)。
