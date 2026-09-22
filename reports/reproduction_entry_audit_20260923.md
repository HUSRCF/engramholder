# 单目标完整预测入口：当前代码依赖核对

2026-09-23，只读审计现有正式路径；尚未进行独立目录的完整预测，不是完成报告。

| 组件 | 已读实现及发现 | 独立交付需要完成 |
|---|---|---|
| 固定Mini主干 | extension_runtime.py固定mini_default_v0.5.0、c4/s5、FP32、torch kernels、无MSA/模板/TF32 | 固定上游版本和底座获取方式，所有资产路径改由用户提供 |
| 正式推理入口 | evaluate_direction_extension.py依赖manifest/checkpoint锁、绝对checkpoint路径、特征缓存与系统名；confirmation角色要求96目标 | 提供明确单目标入口，不篡改原96目标正式协议去冒充一个独立命令 |
| ESM表征 | sequence_feature_cache.py读取预先存在的index与特征文件，并检查序列/模型/hash | 对输入序列提供同一ESM2层的生成路径；不能要求用户先拥有作者缓存 |
| 已训练adapter | 正式加载器验证模型hash、geometry、step及feature provenance，并使用checkpoint内的common decoder配置 | 提供可分发的adapter权重、配置和匹配验证；底座组件的获取/分发按实际条款处理 |
| 原生注入 | query_context与make_replay_module路径已在正式实验使用 | 将必要依赖闭包纳入入口，避免通过作者源码目录偶然成功 |
| 评分 | 正式评分锁列出参考面板、Cα-lDDT实现及固定TM-score工具 | 单目标参考与映射、相同口径命令和工具获取/构建方法，标签与推理输入分离 |

源码核对位置（实验仓库）：

- src/engramfold/experiments/extension_runtime.py
- src/engramfold/experiments/evaluate_direction_extension.py
- src/engramfold/experiments/sequence_feature_cache.py
- reports/full_inference_cross_20260922/remote_v3/scoring_prelock.json

当前paper仓库的算子快照与固定分数足以验证数学性质和表格，却不自动满足上述资产与输入流程。仅拼一条依赖作者目录的命令，或调用预缓存特征跑通，都不能宣称完成“从序列开始的独立复现”。

完成标准仍是一个固定公开目标、一个底座和一个adapter，在干净目录中实际完成序列→表征→完整CIF→固定对应评分，记录环境/资产/输出。它属于下一项工程交付，不需要新的科学矩阵，但不能在未运行前改写为已完成。
