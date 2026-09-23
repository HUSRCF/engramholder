# OpenFold ESMC Stage A execution v1

2026-09-22用户批准A先执行，随后准备B，不等A结果；本目录仅A。
Train384/1536，E-last新增G+R9，C-last新增F3/FR9/G3/GR9，共33新fits。
固定三个seed20260923/24/25、三个R20261001/02/03。复用历史E-last15完整模型。
保留冻结AF2 model_3_ptm、首个主Evoformer现场OPM、4recycle、原任务损失和推理seed。
LR1e-4，wd.01，原AdamW其余参数/clip1保持，最终1536，不新调参、不质量选checkpoint。
G+只旋转残差输出，原baseline不动；自由输出层参数重表示仍可吸收旋转，不要求AdamW等变。

ESMC固定父项目HF/code revision，600M/final post-LN/1152/BF16cache；PLM冻结。
旧缓存525条先验shard/张量，补11；选缓存中最短/最长训练链与最长评测链做无标签round-trip，
相对L2阈值.003提前固定。若任一不通过，统一在H100同环境提取全部536，不以结构质量选特征。
全部最终cache特征hash封存。旧ESM2仍FP32/480；更换PLM同时改变家族/宽度/精度，不叫纯规模因果效应。

变宽输入投影先构建旧480维adapter保留其余初始权重，再用seed+1000000独立流替换Linear1152→128。
同特征内F/FR/G/GR公共encoder一致；冻结head最后输出W/b为零。保存参数/初始摘要。

工程关：固定Train384最短与最长链；E-last历史F/FR/G+同路径回放、G+旋转吸收单元测试；
C-last四种头两步真实loss更新、零初始化精确回放、最后recycle梯度、冻结hash；另最长评测750完整前向。
工程误差沿用旧pair-relative1e-3/coordinateRMS.02；基线零输出要求torch.equal。
仅工程检查使用这些链，不据quality更改层/数据/科学配置。B将另建目录，不能串入A锁。

主要比较Confirm96的C-last Factor方向差。关键次要Psi(C-last)、PLM方向交互K、Psi变化J；
每目标内先平均三seed/三R，目标paired bootstrap20000 seed20260926，95%条件CI。
关键次要三项预锁双侧中心化bootstrap p=(1+#(|bootstrap_mean-mean|>=|mean|))/(20001)，Holm三项校正；
其解释须满足目标抽样假设，不是随机化因果检验。Length48及另外两个结构指标描述性次要。
原始N/G/各R/query绝对质量、seed/R边际、失败全保留。既有观察面板身份不变。

所有历史和新预测在统一三指标下聚合；不把预测重复当独立目标。
前期工程允许固定预算资源重试，保留旧失败；不修改父项目环境/已运行任务。
