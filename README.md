# EngramFold 证据整理仓库

**当前状态唯一入口：** [稿件状态](reports/manuscript_fill_20260922.md)。写作分支为 `main`，活跃来源锁 **v5**，共 **449** 个生成数值字段；入口区分已纳入证据、尚未纳入的独立工作及未完成复现交付。下列报告保留各自历史时点，不是远端队列的实时状态。

[本轮收尾](reports/fixed_budget_status_followup_20260923.md)：在训练规模差值旁补充四格绝对均值均提高的事实；保留固定预算解释及未建立收敛的限制。

[7c701ed盲审落实](reports/blind_review_7c701ed_response_20260923.md)：补短链数据定义、两层控制边界及核心补充指标；活跃来源锁v5，449字段，主文9页。

[A组审阅落实](reports/esmc_review_response_20260923.md)：三套OpenFold交互并列主文，补齐模型引用及ESMC范围图；443数值不变，本轮实测主文9页。

**9月23日 OpenFold A组更新：** [ESMC完整四格及核验报告](reports/openfold_esmc_A_integration_20260923.md)。33组新训练与4,752次预测已完成；绝对质量提高，但ESMC下方向效应与头类型×旋转交互未建立。该次整合来源锁为v4（当前v5）；B组及DiamondHill后续矩阵未纳入本稿，此处不判断其实时进度。

**历史整合记录（2026-09-23）：** 完整推理四格与v4机制边界、近邻定位及复现范围说明已整合；该轮编译记录及视觉检查范围见相应报告。当前版本以顶部状态入口为准。

- [正式审读修改与数值键保护](reports/editorial_review_response_20260923.md)；[作者提交清单](notes/writing_branch_20260922/author_submission_checklist.md)。
- [9月23日稿件更新](reports/full_inference_manuscript_update_20260923.md)与[剩余交付优先级](reports/submission_priorities_20260923.md)。
- [最新PDF](build/iclr2027_conference.pdf)（本机构建产物，不随Git上传）；[表图生成脚本](scripts/build_paper_assets.py)与[数值来源映射](generated/cell_sources.json)。
- [当前章节与写作任务](notes/writing_branch_20260922/skeleton_files.md)：入口 `iclr2027_conference.tex`，正文在 `sections/`，附录在 `appendices/`。
- [已确认主线与Introduction候选](notes/writing_branch_20260922/structure_and_opening.md)：参数化×方向交互 → 方法构造 → 迁移与边界。

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

此处整理不会改变实验配置、统计终点或训练队列。公开写作快照位于 `main`；旧本地分支 `writing/native-update-alignment` 仅为历史。旧草案仅供归档，最终Protenix G+已在MD报告、论文主表与配对比较中完成整合；该次历史整合使用v2来源锁，当前为v5。见[整合记录](reports/protenix_gplus384_manuscript_integration_20260922.md)。
