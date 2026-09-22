# v4 已学习方向机制诊断：证据归档与复核

本目录于 2026-09-22 从实验仓库整理进论文仓库。**v4 已完成；这次只补齐证据并复算统计，没有新训练、GPU 前向或 CIF 评分。** 历史目录的旧 README 曾写 `active`，最终状态应以 [results.md](results.md)、完成审计和本次核对为准。

建议先读 [中文核对报告](audit_20260922.md)，再读 [原始结果报告](results.md) 和 [原始协议](protocol.md)。

| 文件 | 作用 |
|---|---|
| [execution_lock.json](execution_lock.json) | 原始目标、噪声、旋转、主终点与时间字段 |
| [checkpoint_inventory.json](checkpoint_inventory.json) | E2/E3 的模型、步数、几何构造及历史 checkpoint 哈希 |
| [analysis.json](analysis.json) | E1/E2 完整目标级汇总，含负结果、种子边际和关联 |
| [e3_analysis.json](e3_analysis.json) | E3 24 目标训练后状态的末轮残差保留效应 |
| [completion_audit.json](completion_audit.json) | 覆盖、冻结检查与全部 58 个主步长 oracle FD 例外 |
| [runtime_smoke.json](runtime_smoke.json) | 原始运行检查 |
| [core_conditions.json](core_conditions.json) | 864 条链×噪声记录中的必要标量，含 10,368 个学生方向条件 |
| [e3_conditions.json](e3_conditions.json) | 864 条 E3 模型×目标×噪声条件的必要标量 |
| [verify_summary.py](verify_summary.py) | 从精简记录复算原目标汇总和 bootstrap；只依赖 NumPy |
| [independent_verification.json](independent_verification.json) | 本次独立复算结果及适用边界 |
| [source_hash_audit.json](source_hash_audit.json) | 六个历史源码哈希与当前文件的比对，保留一个不匹配项 |
| [source_manifest.json](source_manifest.json) | 原报告和当前源码快照的来源与 SHA-256 |
| [raw_record_hashes.json](raw_record_hashes.json) | 精简数据所对应的 1,728 个原始条件文件哈希 |
| [code/](code/) | v4 与正式推理 hook 的当前源码快照；不是完整可运行模型包 |
| [candidate_full_inference.md](candidate_full_inference.md) | 历史候选；随后已完成，见文件顶部的正式结果链接 |

在此目录执行：

```bash
python verify_summary.py
```

原始 `results.md`、JSON 锁和汇总不改写。复制进来的协议包含历史 active 字样及原文件位置，属于当时记录，并不表示现在有运行队列。checkpoint 路径用于出处追溯，不保证在此仓库存在；这次未重新下载/散列全部模型文件。

本目录未加入论文主表的 `paper_sources.v2.lock.json`，也未更改正文数字或现有匿名 ZIP。它是新补齐的机制证据资料；正式写入 LaTeX 时需要另行显式升级输入锁。本目录含历史机器路径，不是匿名投稿附件。
