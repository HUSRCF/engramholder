# 训练定义补齐与源码核对（2026-09-25）

本轮基于 `c83389f`，落实独立盲审的训练定义意见。只补充现有实现的说明，不修改训练配方、实验结论或数值来源锁。

## 论文改动

- 附录 A.1 完整定义输入投影、三层残差 Conv1d、全序列均值投影、Factor 与 G+ 的层序、宽度、LayerNorm/GELU、bias 与 dropout 边界。
- 明确默认参数初始化、末层权重和偏置清零、保留其他层初始化。解释被替换模块也消耗随机数：仅写同种子而省去历史构造顺序，不能复现初值。
- 说明 ESMC 先构造历史 480 输入 writer，再用独立 CPU 随机流替换输入投影，保留其余参数；措辞覆盖两种实际 RNG 实现，不声称所有设备 RNG 均不受影响。
- 附录 A.2 定义普通旋转的 CPU Gaussian、float64 NumPy QR、列符号修正、FP32存储、左右乘约定及共享范围。无 determinant 修正，允许 O(128) 中的反射；与均值保持、signed permutation 和 trainable C 分开。
- 方法正文增加到完整定义的交叉引用。

## 核验及范围

核心模块与多份历史执行锁匹配。相应 SHA256 见本轮验证 JSON；[旋转源码与 CPU 核验](training_definition_sources/rotation_definition_verification.md)说明匹配范围。

[初始化 CPU 核验](training_definition_sources/initialization_audit.json)检查实际构造、参数计数、同底座 Factor/G+ encoder 对齐及 Native/Rotated 参数对齐。该核验使用记录的 PyTorch 构建；它不等同于所有历史硬件环境的逐字节重现。原训练 checkpoint 的跨执行比较属于后续独立诊断，不在本轮论文定义中预写结论。

三份普通旋转矩阵的字节哈希与历史记录完全一致；第二份 determinant 为负。旋转采样保持全局 Torch/NumPy RNG 状态不变，且不依赖它们的种子。

## 验证与交付

v11的204个来源输入、1830个数值对象、全部生成表图和官方样式保持不变。3项现有数值键测试通过。LaTeX/BibTeX 编译通过，无未定义引用、重复标签、overfull 或最终编译警告；局部视觉检查覆盖方法交叉引用及新增定义。

[本轮反馈 PDF](../build/feedback_v42_training_definition/paper.pdf)属于此轮源码。最终页数预算、全稿视觉验收与匿名包仍统一留到收尾。
