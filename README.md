# EngramFold 证据整理仓库

**2026-09-22：已根据审阅意见形成带真实数据和图表的正文初稿。** 写作主分支是 `main`；本地未提交改动不代表远端已同步。当前状态以[填稿进度](reports/manuscript_fill_20260922.md)为准，历史报告保留各自日期。

- [最新PDF](build/iclr2027_conference.pdf)（本机构建产物，不随Git上传）；[表图生成脚本](scripts/build_paper_assets.py)与[数值来源映射](generated/cell_sources.json)。
- [当前章节与写作任务](notes/writing_branch_20260922/skeleton_files.md)：入口 `iclr2027_conference.tex`，正文在 `sections/`，附录在 `appendices/`。
- [已确认主线与Introduction候选](notes/writing_branch_20260922/structure_and_opening.md)：参数化×方向交互 → 方法构造 → 迁移与边界。

- [机制证据归档：v4](reports/mechanism_evidence/gradient_hpc3_v4_20260920/README.md)：正式协议、逐目标与精简逐条件数据、独立复算，以及完整推理干预的未启动候选；保留源码哈希差异和数值边界。
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

此处整理不会改变实验配置、统计终点或训练队列。公开写作快照位于 `main`；旧本地分支 `writing/native-update-alignment` 仅为历史。旧草案仅供归档，最终Protenix G+已在MD报告、论文主表与配对比较中完成整合；本次采用独立v2证据锁，保留旧锁。见[整合记录](reports/protenix_gplus384_manuscript_integration_20260922.md)。
