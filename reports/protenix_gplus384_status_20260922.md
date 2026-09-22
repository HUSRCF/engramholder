# Protenix匹配G+：2026-09-22执行状态

**已完成：2026-09-22 13:01 HKT。** 三组1536步和432次预测全部成功，六项三指标对比独立复算通过。[完整结果与分析](protenix_gplus384_results_20260922.md)。以下保留启动过程；末尾旧“尚无结果”已被本状态替代。

用户批准后只增加Train384/1536 G+三个种子20260923/24/25。
4608正式updates、432正式预测；Native/Rotated/Query沿用匹配历史结果。

- DiamondHill原Protenix ROCm环境，torch 2.12.0a0+git78258b9 / HIP7.14.60850。
- 独立目录 `/media/PM982/engramfold/runs/protenix_gplus384_20260922`。
- 11:21HKT封存execution_lock，controller PID170623；三种子两步工程前缀已通过。
- 历史初始化、训练顺序、cache、PLM、主干、loss配置一致；零初始化、输出及encoder梯度正常。
- 七项既有InterfaceHead测试通过。pytest首次被上层pythonpath配置遮蔽，显式指定冻结源码后通过；不改训练代码。
- 短链及750残基工程检查全部通过。11:26:53 HKT已释放三个正式续训进程，从第3步继续；前两步计入1536总预算。
- 历史Native短链坐标完全复现；750链跨H100/MI250回放相对坐标L2=0.00032018，小于预先1e-3门槛，非逐位相同。新推理将使用DiamondHill原FP32路径，硬件差异应在结果中保留。
- 三个种子的第一步loss与历史Native精确一致，噪声水平一致；无二次初始化。
- c4/s5、FP32、seed101、sample0、完整序列、固定参考mask；不新增调参、旋转或模型。
- 硬件HIP进程错误最多额外两次120秒间隔重试，保留日志并从32步checkpoint续训。
- 统一评分程序已接入，全部432正式尝试后才评分，三指标与失败计零规则沿用。

主要新增比较固定为Confirm96-B Native−G+；Length48为预设次要比较。
目标内先平均三个匹配种子，再20,000次目标bootstrap；区间条件于已拟合模型。
两面板已观察，均为后续分析，不能重获盲确认身份。无论Native是否优于G+均报告。

[协议](/home/husrcf/Code/onestepfold/engramfold/docs/protenix_gplus384_v1.md)
[锁与工程记录](/home/husrcf/Code/onestepfold/engramfold/reports/protenix_gplus384_20260922)

启动时尚无最终结果；现已完成，见上方结果链接。本轮只更新MD证据，论文生成表和证据锁尚未接纳新增来源。
