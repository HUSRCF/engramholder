# 最新状态：预算扩展与统一评分全部完成

2026-09-26 14:59 HKT。24/24条续训达到3072步，9984/9984正式与Dev预测完成，零失败。Dev参考清单聚合问题已隔离修复，修复前后全部metric_records哈希相同。

[完成报告](review/report.md)：Ψ3072=+0.01820，95% CI [0.01176,0.02504]；配对K=−0.00695，95% CI [−0.01423,+0.00046]。189项独立统计复算通过。不追加训练。

---

## 以下为原执行计划与进度快照

# Protenix ESMC 四格预算扩展：执行记录

2026-09-26。依据[原方案](/home/husrcf/Code/engramholder/reports/protenix_fourcell_budget_plan_20260926.md)，只改变总步数，独立目录执行；未改论文正文。

- DiamondHill，原 Protenix／ROCm 环境，8 个 MI250 逻辑 GPU。
- Factor Native3／Rotated9、G+ Native3／Rotated9，共24条；恢复各自1536步权重与完整AdamW状态，正式全局1537–3072，总新增36864更新。
- 原Train384、ESMC-600M第36层1152维缓存、Full、D=508.5、旋转及损失全部沿用。恒定LR5e-5，不新增C。
- 唯一主终点Ψ3072；直接配对K=Ψ3072−Ψ1536为关键次要量。Fresh192已观察，不称为新盲确认；不据Dev选择checkpoint。
- 正式Fresh预测9408，Dev曲线576，旧／新入口工程预测100，总10084；工程恢复288次更新另外记账。

## 工程与恢复

全部24父状态已重新检查权重身份、AdamW moments/step、训练/开发清单及3072步schedule前1536项。继承全局数据顺序，每条训练链正式后半段再出现4次。

工程训练副本固定使用清单首条7hln_A与最长3pzi_A交替，共8步，再从保存的第4步另进程重跑后4步。这样覆盖最大训练长度的反传显存；工程顺序只在隔离副本中使用，正式恢复原schedule。loss、writer摘要、optimizer与RNG必须同机对应，完整24项通过后放行。

两条同样工程链进行25系统×旧/新入口回放，保持完整序列和4次注入，核验零初始化、完整残差旋转位置、冻结主干和坐标相对误差≤1e-3。标签不进入推理。

正式每128步保存原子断点，含writer、optimizer、全局步、数据顺序及Python/NumPy/Torch/CUDA RNG；固定输出2304/3072。历史1536缺完整RNG，因此不称为未中断历史轨迹的逐位延续。

瞬态HIP等错误最多2次重试，每次间隔120秒；系统身份、非有限值或恢复异常不自动放行。零梯度记诊断，保留原optimizer step，不变None、不跳目标。

## 独立评分

预测完成后再统一评分。Fresh192沿原固定参考mask、失败保留分母计零，Cα pair-lDDT主要，逐残基lDDT/TM-score次要；20,000次原四层目标bootstrap，seed20260926，两节点共用重抽样索引。输出四格绝对值、各臂预算收益、种子/旋转边际、目标向量和历史父模型重放差异。

旧Dev8原清单没有参考文件哈希/CA映射，执行准备阶段单独生成并绑定评分清单；序列输入清单不含这些字段。预启动审查版本保存于远端archives，不覆盖原研究。

远端根：`/media/PM982/engramfold/runs/protenix_fourcell_budget_20260926`。
状态以`status.json`、`engineering/complete.json`、`formal/*/progress.json`和`analysis/complete.json`为准。工程未通过不等于正式已启动；训练完成不等于完整预测和评分完成。
