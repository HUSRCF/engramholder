# 原生更新的数学对象与实现边界

本文件是方法核对笔记，不是论文正文。核对版本：2026-09-21；只读取已完成实验的代码与锁，不更改科学配置。数字来源及文件 SHA256 见[审计 JSON](coverage_robustness_audit.json)。

## 1. 输入相同到什么程度？

所有这里讨论的内部对照在推理时只取同一条 query 序列，不搜索 homolog，不读取评测结构作为输入。**这不等于四种系统获得完全相同的表征。**

| 条件 | 原主干 query 特征 | 额外冻结 ESM2-35M | 显式 query factors | 更新头 |
|---|---|---|---|---|
| Query-only | 有 | 不接入 | 主干内部自然存在，但没有新头 | 无 |
| Native Factor | 有 | layer 12，每残基480维 | 用作固定/现场残差 anchor | 预测逐残基因子增量 |
| Rotated Factor | 同 Native | 同 Native | 同 Native | 相同形式，输出残差固定旋转 |
| G+ | 同 Native | 同 Native | 显式送入通用 pair 头 | 自由 affine 输出 |
| 旧 Generic | 同 Native | 同 Native | **不显式送入 pair 头** | 通用 pair 头 |

因此，Native−Rotated 是最紧密的同构造、同新增输入比较；Native−G+ 控制了额外 PLM 与显式因素访问，但不是相同函数族。各适配器相对 Query-only 的总改善，混合了额外表征与更新路径的作用，不能全部归给方向。

AtlasFold 的原生 AtlasLM-3B 被保留，另外才加 ESM2-35M；AF2/OpenFold 没有原生 PLM；Protenix 此处适配 default checkpoint，官方 Mini-ESM 使用更大的原生 PLM，单列系统参照，不能当完全同表征对照。

## 2. 预测器究竟预测什么？

对序列 \(x=(x_1,\ldots,x_L)\)，设 \(e_i=\operatorname{ESM}_{\rm frozen}(x)_i\in\mathbb R^{480}\)。可训练共享编码器得到 \(h_i=E_\theta(e_{1:L})_i\)，最终输出头产生

\[
(\delta a_i,\delta b_i)=f_\theta(e_{1:L})_i.
\]

Protenix/OpenFold 的因子宽度各32，pair通道128。当前 Factor 预测器从 ESM 特征产生增量；query factors 作为组合算子的 anchor，**不是把真结构或 teacher 因子作为额外预测器输入**。

用 \(k\) 标识实际注入位置，\(t\) 标识 recycle。现场 query-only 特征经原生归一化/投影得到 \(a_{k,t},b_{k,t}\)，原始更新为 \(U_{k,t,0}\)。一般形式是

\[
\Delta U_{k,t}=G_{k,t}(a_{k,t}+\delta a,b_{k,t}+\delta b)-G_{k,t}(a_{k,t},b_{k,t}),\qquad
U^{(R)}_{k,t}=U_{k,t,0}+\mathcal R_k\Delta U_{k,t}.
\]

这里 \((\mathcal R\Delta U)_{ij}=R\Delta U_{ij}\) 是每个 pair 的128通道列向量变换，不是旋转残基坐标，也不混合 residue pair 索引。原生条件 \(R=I\)。固定旋转只作用于残差，baseline、主干权重、参考标签和真实空间坐标都不旋转。

**Protenix 特例：**当前接口使用缓存的 query anchor/Uq；\(G_D\) 的残差归一化常数与 depth=1 的原始 query baseline 分开，因此不能把 \(U_q\) 写成 \(G_D(a_q,b_q)\)。

**OpenFold 特例：**在首个 Evoformer block 的 OPM 调用现场取因子；先前 MSA/pair 更新与 recycle 会改变现场状态。权重冻结不代表 anchor 数值恒定，也不代表阻断原生对输入的梯度。不能把 Protenix 的缓存 anchor 直接用于所有层与 recycle。

## 3. Protenix 的完整与一阶构造

冻结 decoder 定义为

\[
G_D(a,b)_{ij}=\frac{D\,W\operatorname{vec}(a_i\otimes b_j)+b_{\rm out}}{D+\epsilon},\qquad
\widetilde W=\frac{D}{D+\epsilon}W,\quad\epsilon=10^{-3}.
\]

于是差分中的输出 bias 恰好抵消：

\[
\Delta U^{\rm full}_{ij}=\widetilde W\operatorname{vec}\left(\delta a_i\otimes b_{q,j}+a_{q,i}\otimes\delta b_j+\delta a_i\otimes\delta b_j\right).
\]

一阶（tangent）只保留前两项：

\[
\Delta U^{\rm tangent}_{ij}=\widetilde W\operatorname{vec}\left(\delta a_i\otimes b_{q,j}+a_{q,i}\otimes\delta b_j\right).
\]

两者使用同一因子预测器、同样参数数目；删除二阶项不等于减少参数。`NativeGeometryHead` 直接算上述双线性展开并将 \(RW\) 融合进权重，没有必要先计算两个近似的大 decoder 输出相减。

来源：[interface_heads.py](../../onestepfold/engramfold/src/engramfold/models/interface_heads.py)、[native_geometry.py](../../onestepfold/engramfold/src/engramfold/models/native_geometry.py)。这些是本机实验仓库链接；跨机器审阅可用审计 JSON 中记录的源码路径/哈希核对。

### D=508.5 从哪里来？

本次不是仅引用旧稿：从[原始缓存索引](../evidence/protenix_cache_index.json)的 `train_target_ids` 读取24个 `teacher_depth`，排序为：

```text
114,489,490,492,493,502,502,504,504,506,507,508,
509,510,511,511,512,512,513,513,513,513,513,513
```

中间两个为508与509，中位数为508.5，与 `fixed_depth` 一致。它来自**旧 Train24 的训练侧 MSA 深度统计**，不是测试目标的 MSA 深度，不是长链或 Confirm96 的质量调参，也不是每次 forward 的真实 homolog 行数。

此后方向、Train96/384和Length48实验沿用该固定值，没有按新训练规模或评测链重新估计。task-only推理没有真实 homolog 输入，但历史上使用过这个训练侧 MSA 元数据常数，应如实披露，不能写“整个研发从未使用任何 MSA 信息”。它只用于残差的固定尺度；depth=1 的原始 \(U_q\) affine 仍完整保留。

OpenFold 使用该层实际 mask/有效行数分母和原生 epsilon；AtlasFold 的 difference/product 没有这个OPM深度常数。**跨底座复用原则，不复用508.5。**

## 4. 冻结边界与零初始化

冻结：ESM2、folding主干、原生因子投影/归一化、OPM decoder、固定旋转矩阵。可训练：新增序列编码器、增量预测头；G+另外包括node projection和pair MLP。通过冻结主干反传结构梯度，不等于更新主干参数。

只把最后输出 affine 的权重和偏置置零，不把整个编码器置零。因此 \(\delta a=\delta b=0\)，Full/Tangent残差都为零，任意旋转初始均回到原生 Query-only。G+最后输出 affine 同样全零，残差也为零。第一个更新只有最终输出层获得有效梯度、前面层随后开始学习，是该初始化的正常结果。

当前参数数目：Protenix/OpenFold Factor各373,824，G+各378,144；旧Generic378,272。近似匹配不是完全相等。AtlasFactor423,168、G+415,008；其完整结果尚未进入本报告。

## 5. G+与旧Generic为什么不能互换？

源代码的G+ pair输入是

\[
[z_i,z_j,a_{q,i},b_{q,j},\operatorname{clip}(i-j,-32,32)/32],\qquad
z_i=\operatorname{Linear}(\operatorname{LN}(h_i)).
\]

经过 Linear→GELU→自由 Linear 输出128通道残差，再加回相同baseline。旧Generic不含 \(a_{q,i},b_{q,j}\)，而且为匹配参数量使用不同hidden宽度。两者都能接受同一序列，不意味着它们是同一个“同信息”控制。

G+最后自由输出层使 \(W_R=R^TW,b_R=R^Tb\) 能吸收输出旋转，故函数集合不变；AdamW逐坐标二阶矩不保证对应的训练轨迹。新旋转G+实验的协议与进度由[独立方案](rotatable_gplus_control.md)记录，**本文件不把它算作已完成证据**。

## 6. 同谱具体控制什么？

在相同因子增量、相同 anchor 和预测器参数处，正交 \(R\) 保持残差Frobenius范数、decoder奇异值，以及局部writer Jacobian的 \(J^TJ\)。它不保持固定下游的 \(J_{\rm down}RJ\)，不保持AdamW优化轨迹；训练后不同模型的实际残差范数也不必相同。

尤其现场anchor依赖历史状态的AF2/Atlas路径，只能在匹配现场状态谈局部算子等距，不能推出整个recurrent系统参数Jacobian同谱。完整Factor也没有对任意输出旋转保持函数集合不变的通用保证；这不是“所有旋转都必然有害”的定理。

## 7. 同构造的数据规模交互必须单列

交互使用的是 Mini **Full、1536 updates、同训练种子/旋转族/学习率/损失**：

| 训练集 | Native | Rotated | Native−Rotated |
|---|---:|---:|---:|
| Train96 | 0.61653 | 0.59148 | +0.02505 |
| Train384 | 0.62572 | 0.58621 | +0.03951 |

逐目标计算差中之差得到 +0.01446，保存的配对95%区间为[+0.00350,+0.02586]。本次从原始score记录重算每个目标后与[原交互](../evidence/protenix_data_interaction.json)逐项吻合，见[核验](coverage_robustness_audit.json)。

**Mini Train24 tangent/384的+0.05710不参与这个交互。**用它与Full Train384相比同时改变了构造、训练集合和步数，不能当数据规模交互。即便同Full1536，也是在固定更新预算下改变集合组成及每链平均重复数（16与4），不是纯样本数量因果效应或普遍scaling law。

## 8. 底座配置差异，不能只用一条公式掩盖

| 项目 | Protenix Mini / Tiny方向研究 | AF2 / OpenFold正式研究 | AtlasFold正式矩阵（结果待收齐） |
|---|---|---|---|
| 实际注入 | 单个MSA block的OPM边界，保留其pair stack及后续主干 | 48个Evoformer中第一个block的OPM，每轮recycle | 首个LM stack的SequenceToPair式difference/product边界，每轮recycle |
| anchor | 缓存query因子/更新；不随着各模型现场状态重算 | 当前OPM调用现场因子，依赖上游更新与recycle历史 | 当前sequence/pair过程的现场因子；不是OPM缓存 |
| 归一化 | 残差固定D=508.5，独立保留depth=1 query baseline | 原生逐位置mask及pair有效行数+epsilon，保留输出affine | 原生归一化/投影，无MSA深度D；组合顺序为difference再product |
| 训练结构损失 | 4MSE+4bond+4smooth-lDDT+0.03distogram；protein-only现有bond mask为空时该项为0；confidence关闭 | 原生FAPE×1+distogram×0.3+supervised-chi×1；关闭confidence与masked-MSA辅助训练 | 0.4distogram+2(weighted MSE+smooth-lDDT)；关闭confidence训练 |
| 训练recycle梯度 | 4轮，只有最后轮保留任务反传 | 3次recycling迭代加初始共4次trunk，最后轮反传 | forward_train(num_recycles=3)，实际4次trunk；按锁定训练入口 |
| 结构推理 | 4轮/5步diffusion，seed101，1个sample | 4次trunk+原生structure module；禁用提前停止；非diffusion | num_recycles=4实际5次trunk；≤512残基20 diffusion步、513–1024为30步，seed1，1sample |
| 已完成主训练预算 | Train24/384 tangent与full；Full Train96/384各1536；见覆盖表 | Train96及Train384各1536，15组/集合 | Train96/1536，15组；部分产物不作结果 |

这些loss是各底座自己的训练目标，不能横向比较原始loss大小。推理步数也不是强制统一数字；受控比较要求每个底座内部配方匹配。实际耗时、峰值内存另行核对，不能从update数推定跨底座计算相等。

实现位置参考：[OpenFold runtime](../../onestepfold/engramfold/src/engramfold/experiments/openfold_adapter_runtime.py)、[现场OPM](../../onestepfold/engramfold/src/engramfold/models/live_opm.py)。审计JSON的源码hash是本次读取时的状态，不覆盖或替换各历史执行锁封存的训练源码hash；例如新G+旋转接入可能让工作区源码比旧实验锁更新。
