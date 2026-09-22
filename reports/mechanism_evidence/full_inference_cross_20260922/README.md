# 完整推理：已学习残差来源 × 整体范数

2026-09-22 完成，2026-09-23 纳入论文证据。

- [最终结果与复核](results_review.md)
- [执行与前置修复记录](execution_review.md)
- [锁定协议](staging/protocol.md)、[执行锁](remote_v3/execution_lock.json)
- [原始720条结构分数](analysis/metric_records.json)、[四格与对比汇总](analysis/analysis.json)
- [独立核对脚本](verify_results.py)、[核对结果](independent_verification.json)
- [复制来源哈希](source_manifest.json)

运行 `python verify_results.py` 从当前目录中的评分记录复算36项格子/对比的目标数组、均值、种子/旋转边际和分层bootstrap区间；它不重跑折叠、不重新评分CIF，也不重新验证远端720份CIF本体。远端检查的范围由 completion_review.json 单独记录。

主要结论是：在所测两个实际学习范数下，原生来源残差仍产生更好的最终结构，整体范数不足以解释差距。方向来源包含归一化后全部内容和局部强弱分布；不等同于纯通道旋转。幅度/交互未建立稳定效应不等于严格为零。

24条目标已观察；本轮为Train384/Full，不是v4 Train24/Tangent的同模型续证。原始文件保留历史本机路径，不是匿名附件。
