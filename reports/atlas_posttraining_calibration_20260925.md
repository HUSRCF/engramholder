# AtlasFold 并行通道校准实验

本实验检验：已训练的 AtlasFold Native 适配器，追加同样512步时，只学输出通道变换C是否优于继续训练原头；以及改进是否真正超过未适配AtlasFold。

完整预定方案见 [PROTOCOL.md](/home/husrcf/Code/onestepfold/engramfold/reports/atlas_posttraining_calibration_20260925/PROTOCOL.md)，配置及源文件哈希见 [execution_lock.json](/home/husrcf/Code/onestepfold/engramfold/reports/atlas_posttraining_calibration_20260925/evidence/execution_lock.json)。本轮未修改论文、既有训练任务或系统记忆。

## 固定矩阵和预算

| 分支 | 三个起点 | 每个起点的额外训练 | 正式任务数 |
|---|---|---:|---:|
| C_only | ESM2 / Train96 / Native / step1536 | 512步；原头冻结；8001维C从恒等开始 | 3 |
| head_only | 相同三个checkpoint及其AdamW状态 | 512步；继续训练原头 | 3 |

三个种子为20260923、20260924、20260925。两分支使用相同的训练顺序后缀与逐步随机种子。Formal固定用step2048，不依据正式面板挑checkpoint。

在第一个种子上，用两分支×两个共同学习率倍率×64步做Dev8校准；只选一个共同倍率。Formal的原头学习率为1e-4×倍率、C学习率为1e-3×倍率。训练预算为3072正式更新+256校准更新+32工程检查更新。

测试集仍为原Confirm96-B的96条。重新预测Query、三个原Native、六个新分支共960条；另外32条Dev预测。两张MI250设备预计总耗时约5–7小时，计入训练、初始化、校准和评分；首次实测后可细化。

## 如何判断结果

主对照是C_only−head_only，同时必须展示C_only−Query。只超过原Native而仍低于Query，应表述为缓解适配器的负收益。不能仅凭该实验主张旋转机制成立，因为没有Rotated分支。

正式评分须等十个系统全部预测结束；失败目标保留为零分。报告三种子的各自均值，以及先平均种子、再以目标为单位的配对bootstrap区间。这是已观察面板的后续实验，区间以本次拟合的模型为条件；辅助比较不作为新的独立盲测结论。

## 部署和检查记录

运行根目录：

`DiamondHill:/media/PM982/engramfold/runs/atlas_posttraining_calibration_20260925`

使用HIP6/7，分别对应PCI8e/93和ROCm监控GPU4/5。控制器按“smoke → Dev校准 → 6个正式续训 → 10系统预测 → 统一评分”执行；工程检查失败会保留证据并停止，不自动改种子、损失、训练集或数值容差。

源码锁：`d47573f186c285ed4757181423b75f1165920e62d46bd96647656ce261a550b5`。

15项通道/优化器测试在实际Atlas环境通过，涵盖BF16微小残差恒等性、C与live anchors的梯度、原AdamW完整恢复、恢复后下一步更新与原优化器逐位一致。统计函数另已完成Dev选择与配对bootstrap的CPU检查。

首次启动错误使用fold解释器，因缺少einops在模型加载前停止，两分支均未进行更新。失败日志保留；随后通过`operations/runtime_supplement_v2.json`记录环境入口修正，使用历史Atlas解释器`folding_e2e_20260921/venvs/atlasfold-rocm/bin/python`。封存源码、实验矩阵和checkpoint未改变。

当前控制器PID为944183，日志`controller_v2.log`。实时状态查看`STATUS.json`；训练进度在`atlas/{stage}/{run}/progress.json`。最终结果写入`atlas/formal_analysis/analysis.json`和`atlas/COMPLETE.json`。

启动时已核实两卡独立占用；具体工程检查进度以最新STATUS/progress文件为准，不把控制器启动等同于正式训练已完成。

## 已核实的启动进度

两支均通过8步连续更新、4+4恢复检查和冻结参数验证，现已自动进入Dev8校准。两个训练目标的原路径前向差异及恢复后首个前向差异均为0。连续8步与4+4路径的最终参数相对差：C_only为0，head_only为0.0005410（约0.0541%）。后者按预定协议作为完整数值轨迹诊断保留；未修改既定放行条件。

真实Dev CIF解析、独立lDDT实现、TM工具调用，以及Python/Gemmi参考路径拦截的CPU集成检查均通过。此时尚无正式实验结果。

## 空闲资源扩容

用户指出可在另外六张卡释放后扩容。已核实旧Protenix队列结束、六卡空闲，并改用单独审计的operations/controller_expand_v3.py；封存科学协议及其哈希不变。仅停止旧controller父进程，原两个子任务完成了全部64步和8条Dev预测，均无失败，完成凭证已核验后接管。

新控制器PID952712、日志controller_v3.log，设备池为全部8张卡。正式6个续训可同时执行，随后10个模型预测任务最多8路并行。以实测每步约6.7秒估计，扩容时点起剩余约1.5–2.5小时；更早的5–7小时估计仅适用于两卡排队。正在运行的任务和已完成结果均未重置。
