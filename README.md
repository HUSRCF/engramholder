# EngramFold 证据整理仓库

**2026-09-22：用户已确认主张并授权开始章节骨架。** 现有MD报告继续作为事实依据；待完成结果不预填。

- [当前章节与写作任务](notes/writing_branch_20260922/skeleton_files.md)：入口 `iclr2027_conference.tex`，正文在 `sections/`，附录在 `appendices/`。
- [已确认主线与Introduction候选](notes/writing_branch_20260922/structure_and_opening.md)：参数化×方向交互 → 方法构造 → 迁移与边界。

- [三项审阅意见与投入优先级](reports/reviewer_priorities_20260922.md)：哪些已补齐，Protenix匹配G+还缺什么；已启动，状态见对应报告。
- [9月22日完成结果更新](reports/results_update_20260922.md)：OpenFold旋转G+交互、AtlasFold完整边界与ESMFold2最终评分。
- [证据审阅报告](reports/evidence_review.md)：核心问题、直接核对的数字、可支持与不可支持的解释，以及仍待补齐的记录。
- [数学对象与实现边界](reports/math_contract.md)：query/ESM信息来源、Full/Tangent、冻结与旋转位置、D=508.5逐项来源及同构造交互。
- [稳健性与暴露审计](reports/robustness_audit.md)：完整种子/旋转、目标差异分布、锁定时序、累计暴露限制；附[可复算脚本](scripts/audit_evidence_coverage.py)。
- [工作方式](notes/workflow.md)：先核验证据；当前已授权章节与占位写作。
- [可吸收旋转的 G+ 对照](reports/rotatable_gplus_control.md)：代码前提、四格交互、九组最小矩阵与历史计时；运行进度见该独立文件。
- [证据快照](evidence/)：已经拷入的结果 JSON，便于逐项核对；不包含全部原始 CIF。
- [未经用户审阅的旧草案](notes/unreviewed_writing_20260921/)：仅归档，不能当作正式稿。

仓库根目录已按用户指示替换为ICLR章节骨架，原模板另行备份。现有 source_*.md 是此前助手草稿的快照，不代表用户已认可其文字或论证。

此处整理不会改变实验配置、统计终点或训练队列。当前分支为 `writing/native-update-alignment`；旧草案仅供归档，章节骨架仍有显式占位。
