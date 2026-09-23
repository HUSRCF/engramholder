# 方向效应的条件依赖：架构能解释什么，下一步测什么

2026-09-24。写作基线 `e18e4e3`，已纳入证据以 v6 来源锁为准。本轮是固定源码、执行记录和数学构造的审计，**没有运行新训练、结构预测、权重谱测量或相关性搜索**。以下“假说”和“建议分析”均不是已取得的实验结果，也不是已经锁定的新实验协议。

## 1. 判断：可以增加实质内容，但不能仅靠架构故事解释结果

最值得补充的不是“不同模型层数不同，所以结果不同”，而是把“原生接口”拆成四个可审查的对象：

1. **可达更新集合**：给定现场因子，头究竟能产生哪些更新？
2. **冻结下游的读取方式**：这些更新如何被通道投影、归一化、attention 和结构模块使用？
3. **状态依赖**：后续 recycle 的更新是否随先前适配状态变化？
4. **有限预算学习**：共享预测器在给定特征、数据与优化器下，实际学到了哪种更新？

前三项可以从架构与算子中得到具体事实；最后一项及其与结构质量的关系必须测量。它们提供解释候选，尚不是一个能事先选出最佳 adapter 的规则。

最新结果也要求更新问题表述：

| 已完成比较 | 当前能支持的判断 |
|---|---|
| OpenFold Train96/ESM2，旧 Confirm96-B、Length48 | 两个已观察面板上的头类型×旋转交互得到支持 |
| 同一批 Train96 模型，Fresh96 | Factor 方向效应 +0.01240；唯一主要交互 +0.01014，区间 [−0.00318,+0.02347]，未建立 |
| OpenFold Train384/ESM2 与 Train384/ESMC | 两套特征在两面板上均未建立主要交互；不等于证明交互为零 |
| ESM2 的直接 Ψ384−Ψ96 后续分析 | Confirm96-B 为 −0.01914 [−0.03759,−0.00168]；Length48 为 −0.00698 [−0.01673,+0.00259] |

来源：[Fresh96与直接交互变化](fresh96_training_psi_integration_20260924.md)、[A组完整四格](openfold_esmc_A_integration_20260923.md)。直接变化是后续、未校正分析，不是单纯并列两次显著性。

**同一架构、同一批模型也会因目标面板不同而有不同的证据强度。**因此“48层会补偿旋转”“小数据需要原生偏置”都不足以解释当前结果。Fresh96与旧面板之间尚未直接检验效应差异，不能因前者区间跨零就断言它的真实效应较小。

## 2. 三条实际运行路径：不能把 live anchor 统称为历史反馈

| 属性 | Protenix Mini-default | AF2/OpenFold model_3_ptm | AtlasFold-260703 |
|---|---|---|---|
| 适配算子 | 单 MSA block 的 OPM | 第一个主 Evoformer OPM | 第一个 LM block 的 difference/product |
| 因子／pair通道 | 32+32／128 | 32+32／128 | 128+128／128 |
| anchor 来源 | 缓存 query 因子 | 当前 MSA 状态，经 pair-biased attention、column attention、transition 后取因子 | 当轮 masked AtlasLM 单点特征，经首个原生投影取因子 |
| 首接口是否读取适配后的历史状态 | 否 | **是** | **否**；首接口在独立 LM 分支内，尚未接收 recycled pair |
| 各轮 Factor 增量 δa、δb | 同一次推理中相同 | 同一次推理中相同 | 同一次推理中相同 |
| 各轮完整残差 | 同一个张量重复注入 | 随现场 anchor 变化 | 随该轮 MLM mask 改变的 anchor 变化，不是适配历史反馈 |
| 主要后续计算 | 保留1个 MSA pair-stack block → 16个 Pairformer blocks → diffusion conditioning/denoiser | 首 block 的 pair stack → 其余47个完整 Evoformer → 8个结构模块 blocks | 余下 LM stack → 原生 LM 投影并入回收状态 → 48个主 Pairformer blocks → diffusion |
| 原生 PLM 信息 | 此 default checkpoint 未启用原生 ESM；adapter 另加 ESM2 | AF2 本身无原生 PLM；adapter 另加 ESM2或ESMC | 保留 AtlasLM-3B、多层表示混合及 attention-derived pair 输入；adapter 另加 ESM2 |
| 正式推理 | 4轮 trunk，5步 diffusion | 4轮 trunk，原生结构模块 | 5轮 trunk，按长度20/30步 diffusion |

不同底座的 block 数不代表同一种计算，也不能直接当成“纠偏能力”排序。

### 2.1 Protenix：静态写入、动态读取

历史执行入口在 `runner.predict` 之前形成一次完整更新，`StaticOPMProvider` 在每轮返回同一张量。它加到当前 pair state 后，仍运行原 MSA pair stack 与 Pairformer。回收状态包含对上一轮 pair/single 的归一化与线性投影。

Mini 只有一个 MSA block，该 block 本来就是最后一层；上游实现只在非最后 MSA block 创建 MSAStack。所以当前静态性不是通过删掉一整串原本存在的 MSA attention 才制造出来的。

最后的 pair 表征还需经过 diffusion conditioning 的归一化、投影与 transition。冻结下游并没有随残差一起旋转，不能期待输入任意正交旋转后输出不变。[固定 MSA 源码](https://github.com/bytedance/Protenix/blob/d3b4db6a121dd4584edd85e93744239325a2b72e/protenix/model/modules/pairformer.py#L607)、[回收路径](https://github.com/bytedance/Protenix/blob/d3b4db6a121dd4584edd85e93744239325a2b72e/protenix/model/protenix.py#L234)、[diffusion conditioning](https://github.com/bytedance/Protenix/blob/d3b4db6a121dd4584edd85e93744239325a2b72e/protenix/model/modules/diffusion.py#L86)。

**D=508.5 不是把残差缩小508倍。**bias相消后实际乘数是 D/(D+0.001)，约0.999998；OpenFold有效单行pair则除以1.001。不能把508.5这个数字当成跨底座量级差异的原因。实际量级仍取决于权重、anchor与学到的增量。

### 2.2 OpenFold：增量不读现场状态，组合算子读；G+在不同位置读

每轮先将上一轮 MSA 第一行、pair与预测坐标的距离编码回收。第一个主 Evoformer 先做带pair bias的MSA row attention，再做column attention与transition，之后才到适配 OPM。[固定执行顺序](https://github.com/aqlaboratory/openfold/blob/be2ec1841f16c966c65ae0e7599ebbadc725757d/openfold/model/evoformer.py#L473)、[回收调用](https://github.com/aqlaboratory/openfold/blob/be2ec1841f16c966c65ae0e7599ebbadc725757d/openfold/model/model.py#L265)。

Factor 是

\[
(\delta a,\delta b)=f_\theta(e(x)),\qquad
\Delta U_t=\widetilde W(\delta a\otimes b_t+a_t\otimes\delta b+\delta a\otimes\delta b).
\]

固定一次推理的 θ 与 PLM features 后，δa、δb相同；变化来自 a_t、b_t。G+则把现场 a_t、b_t直接放进非线性 pair MLP，与节点特征共同决定输出。两种头对现场状态的响应组织不同，**不只是一个输出层冻结、另一个自由**。[实际 hook 与残差](../reproducibility/src/engramfold/models/live_opm.py)、[G+输入](../reproducibility/src/engramfold/models/interface_heads.py)。

训练的四轮都会数值上使用当前 adapter；前三轮 `no_grad`，只对最后轮反传。前三轮状态随θ变化，但其导数在该步被截断。这是封存训练配方的一部分，不是发现了实现错误；也意味着共同状态的局部等谱不能升级为完整回收模型的参数 Jacobian 等谱。

### 2.3 AtlasFold：需要纠正早期报告中的反馈归类

主 trunk 先构造回收 s/z，随后**独立**调用 `run_lm_embedder(batch, mlm_mask)`。该函数没有 s_prev/z_prev 参数。首 LM block 第一项是 `PairwiseProdDiff(s)`，pair→single 在它后面。因此当前首 hook 的因子不会由上一轮适配后的 pair 反馈改变。[固定源码：LM与回收分支](https://github.com/SeonghwanSeo/atlasfold/blob/8ab3aca0e18c8b814d5ca6756b2617a07d72c68d/src/atlasfold/model/model.py#L364)、[首block顺序](https://github.com/SeonghwanSeo/atlasfold/blob/8ab3aca0e18c8b814d5ca6756b2617a07d72c68d/src/atlasfold/model/network/block.py#L244)。

现场因子仍可能轮间不同：运行入口使用 MLM probability 0.15，每轮重新采样mask。训练用run seed+step，正式推理`seeds=1`对应固定列表[1]。配对输入和随机条件下，各臂同轮首anchor不因适配历史改变；下游状态仍会分叉。

这修正了[旧数学笔记](math_contract.md)把“AF2/Atlas”共同称作历史依赖anchor的句子。**运行时重算、随机条件变化、适配历史反馈，是三个不同概念。**正式论文目前“live”的简称不必判成错误，但以后应明确这个区别。

Atlas原生输入已有多层 AtlasLM 表示混合与 attention 投影得到的 pair 特征；额外 ESM2-35M 与“给一个没有PLM的主干增加新信息”不是同一种设置。其48个主 Pairformer blocks均不再启用single→pair，最后12个具有pair→single。不能把它描述成与AF2相同的48个OPM反复重写。[固定配置](https://github.com/SeonghwanSeo/atlasfold/blob/8ab3aca0e18c8b814d5ca6756b2617a07d72c68d/src/atlasfold/configs/atlasfold.py)、[stack构造](https://github.com/SeonghwanSeo/atlasfold/blob/8ab3aca0e18c8b814d5ca6756b2617a07d72c68d/src/atlasfold/model/network/trunk.py)。

## 3. 一个可以直接证明、但范围很窄的架构差别

以下是本轮从算子推导的数学观察，**不是已测得的权重秩、也不是新的结构效果实验**。固定一个有效残基对及其现场状态，暂时将两组因子增量视为自由变量；实际共享预测器还会施加跨残基、跨目标约束。

### 3.1 宽度32的OPM：单pair局部秩至多63

设 p=a_i+δa_i，q=b_j+δb_j。Full残差在这个pair上的微分为

\[
\mathrm dU=\widetilde W\operatorname{vec}(\mathrm dp\,q^\top+p\,\mathrm dq^\top).
\]

两个因子共有64个自由变量，但缩放方向 (dp,dq)=(p,−q) 给出零变化。若两个因子都为零，导数直接为零；只有一个为零时秩最多32。因此所有情形均有

\[
\operatorname{rank}J_{ij}^{\rm OPM}\le\min(128,2\times32-1)=63.
\]

在Full的非零学习状态，把p、q换成该状态的移位因子，结论仍成立；Tangent在固定anchor处也有相同上界。旋转保留该局部秩，但通常改变这片局部可达子空间在128维通道中的位置。

需要避免两个误读：

- 不是说整个蛋白只剩63个自由度，也不是说所有目标共享同一个63维线性空间。
- W是128×1024矩阵，其列空间即使满128维，也不能让一个pair上的**秩一因子外积**变成任意1024维输入。“decoder列空间满”不能排除因子构造的约束。

这给“受约束”一个具体的局部数学含义。它仍不证明Native子空间更适合任务；Protenix与OpenFold都有这个上界，故它也不能单独解释两者的结果差别。

### 3.2 Atlas差／积：单pair有可能局部覆盖128个通道

实际顺序为

\[
\Phi(p,q)=\begin{bmatrix}p-q\\p\odot q\end{bmatrix},\qquad
U=W\Phi(p,q),\quad W\in\mathbb R^{128\times256}.
\]

其输入 Jacobian 是

\[
J_\Phi=\begin{bmatrix}I&-I\\\operatorname{diag}(q)&\operatorname{diag}(p)\end{bmatrix}.
\]

若每个通道p_c+q_c非零，这个256×256矩阵可逆；若W同时满行秩，则WJ_Φ的行秩为128。这是一组**充分条件**；个别和为零不一定令投影后的行秩下降。

源码维数与顺序支持这个推导：[原生算子](https://github.com/SeonghwanSeo/atlasfold/blob/8ab3aca0e18c8b814d5ca6756b2617a07d72c68d/src/atlasfold/model/network/block.py#L26)、[实际残差展开](../reproducibility/src/engramfold/models/live_prod_diff.py)。

本轮没有检查真实W及anchor是否满足良好条件数。满秩也不等于各方向同样容易学习。整链只有256L个自由因子坐标，却同时组成128L²个输出，不能逐pair任意指定，更不能推出Atlas Factor具有G+那种整函数类旋转闭包。

**可以支持的新认识：不同底座中的“原生Factor”不是同一种强度或同一种形式的约束。**不能据此声称已经解释Atlas无收益，或证明非OPM不需要方向。

## 4. 什么条件下，原生方向才可能真的重要？

把共同状态下的局部链式导数写成

\[
J_{\rm pred,R}=J_{\rm down}\,\mathcal R\,A\,B,
\]

其中A是因子→pair writer，B是共享序列预测器对参数的导数，J_down是下游对注入的导数。这只是共同状态的局部表达；完整回收路径还包括状态反馈，正式训练又截断早期recycle梯度。

头内等谱要求固定现场anchor、writer参数及因子增量，只切换R；此时保留AᵀA，却不保留AᵀRᵀJ_downᵀJ_downRA，更不保证与任务误差的关系不变。这里比较的是同一线性化点的局部控制；在真实非零更新上切换R还可能改变下游导数的求值点。独立训练的两个checkpoint即使共享下游起点，A、B本身也可能不同，不能把局部等谱声称为它们之间的匹配。**“下游响应大小”与“响应是否朝正确结构变化”也不同。**即使响应度量各向同性，目标误差方向仍可能使任务对不同更新有不同偏好。

因此更合理的条件链是：

| 条件 | 它提供什么 | 缺少它或没有测到它时 |
|---|---|---|
| 可达函数集合不能自由吸收该旋转，或优化过程不等变 | 旋转有机会改变可学问题 | G+虽函数类闭包，AdamW仍可能带来差异；闭包不是训练等价 |
| 冻结下游对不同更新有不同的任务相关响应 | 原生与旋转可能产生质量差异 | 光看残差范数、秩或谱不能判断收益符号 |
| 现有特征和共享参数能在预算内学到有用修正 | 潜在几何优势有机会转为结构效果 | oracle可达性、非零梯度都不足以保证泛化 |
| 评测目标中存在相应需要，且效应可被当前评分辨别 | 差异有机会体现在目标均值上 | 目标分布、评分余量与估计不确定性仍可能改变结果 |

这是一份**检查条件的框架，不是已经验证的乘法定律或必要充分定理**。当前不能为新模型给出“满足某个阈值就用Factor”的建议。

相比泛泛谈“更多数据”，这里能明确指出：OpenFold训练规模变化没有改变算子宽度、注入位置和主干架构，改变的是训练集合覆盖、组成、每链重复次数及所学θ，进而可能改变现场anchor与实际残差。ESMC又同时改变特征家族、宽度、精度与投影参数量。不能用同一个“原生偏置强弱”故事吸收所有正反结果。

## 5. 哪些具体解释有根据，哪些现在不能写成结论？

| 候选解释 | 已有依据 | 能将它否定或收窄的观察 |
|---|---|---|
| OPM局部约束使方向错配更难绕开 | 单pair秩上界、跨pair共享、无通用旋转吸收保证 | 满足相同约束却效应不同；当前Protenix/OF及OF内部结果已说明约束不是充分解释 |
| 现场反馈改变学习到的残差作用 | OpenFold真实anchor读回收状态，Factor/G+读状态的方式不同 | 记录发现anchor变化或残差变化很小；或有限的固定anchor干预并未按预言改变条件差异 |
| 下游使两类有效更新更接近／更远 | 冻结投影和非线性确实不随R旋转 | 传播差异不对应实际结构变化；“更深自动纠偏”不能由block数获得支持 |
| Atlas新增信息价值有限、结构分数余量较小 | 原生AtlasLM、多层/attention输入；baseline约0.95；当前未建立适配收益 | 更新已明显改变内部表征却损害结构，或剩余错误与接口任务不对应；高baseline不能证明原因 |
| 1536步的学习阶段影响旋转敏感性 | 没有正式收敛曲线；Train96每链16次、Train384每链4次 | 固定Dev结构曲线较早稳定仍有差异；或四臂均有效学习但交互无同向变化 |

不能将“Native相对baseline有收益”当成“原生方向有价值”的同义词；Atlas缺少rotated G+，也不能把其Native−Rotated无优势写成核心Ψ的独立反证。RF3、OpenFold3、ESMFold2的系统分数同样不提供相应四格机制证据。ColabFold与OpenFold运行同一AF2权重，不增加独立架构数。

## 6. 若继续，优先做什么：一个已有checkpoint分析，加一个小型算子检查

以下是供讨论的有界方案；**没有自动提交任务，也不改变原实验终点**。

### 优先一：同架构固定Dev曲线，分清学习阶段与静态架构

先清点OpenFold两套ESM2正式四格的384、768、1536步checkpoint及配对schedule。必须包括旋转Factor与旋转G+，不能只看Native/G+。缺失中间checkpoint记为缺失，不为补图重训。

如果配套齐全，在同一个旧Dev8、固定四轮推理和随机条件下评测所有节点，并记录绝对质量、Δ_F、Δ_G和Ψ；不挑最优checkpoint替换正式最终结果。完整上限为2套训练集合×24模型×3节点×8目标＝1,152次模型—目标预测，另计共同基线。已存在且配置、输出可核对的Dev结果可以复用。

这还能给出一个有用但不纯粹的对照：Train96/384步与Train384/1536步都约4次全遍历。它改变了更新预算及数据集合，**不是“只控制epoch就隔离数据量”**；与原固定1536步比较放在一起，能检查简单的重复不足叙述是否与曲线相容。

判读应提前明确：

- 两个头的旋转差异均随阶段变化，不能只解释Factor。
- 若旋转臂在固定Dev上持续追近，支持阶段依赖候选，仍不证明最终收敛后的能力相同。
- 若曲线并不支持“较慢追近”，就结束这个解释，不顺势增加训练步数。
- 八个开发目标不足以建立普遍预测规律；这是开发诊断，不将它称为新确认。

### 优先二：算子几何与实际更新使用情况

先复用已保存的真实因子／更新；不足时只在固定旧Train/Dev样例提取，目标按ID规则选，不按效果选。

**几何检查只回答局部可达性。**对实际W与固定anchor计算J_ij的奇异谱、数值秩和条件数；Full还应区分初始anchor与已学移位因子。不按Native赢多少来定截断阈值，不以“满秩”代替良好条件数。每pair的子空间旋转差异可作为结构诊断，但既不是整链函数类判定，也不直接预测结构收益。

**在自然完整推理中记录更新如何被使用。**先不用微小有限差分或真实结构梯度，记录：每轮anchor、残差相对当前pair状态的大小，注入后／trunk末端的表征变化，以及最终预测距离变化。Atlas要分清外生MLM变化和适配历史；OpenFold则保留现场反馈。坐标比较优先采用预测内距离，避免刚体自由度影响。

Factor的Full可以拆为线性项L_t与二次项Q，并固定记录

\[
q_t=\frac{\|Q\|}{\|L_t\|+\|Q\|},\qquad
c_t=\frac{\|L_t+Q\|}{\|L_t\|+\|Q\|}.
\]

分母为零单列。它们表示二次项比重和抵消，不意味着Q更大就绕过原生方向约束。与Query的自然轨迹差异混合了状态变化；如后续比较共同现场状态下的u与Ru，应另列为干预，不冒充自然训练轨迹。

至少保留三种可能：更新本身小；更新非小但后续响应弱；响应明确但结构质量没有改善。**大响应不等于有用响应。**Atlas尤其值得做这个区分，不能从最终差值小直接倒推出head没学动。

### 本轮不推荐：继续寻找一个事后相关性指标

不要用三个底座的三个均值拟合“架构预测器”；不要挑一组与Δ_F相关、却拿来解释Ψ；不要在Fresh96上开发新规则后仍叫独立确认。旧oracle未预测最终逐目标收益的负结果继续保留。

若某个预先选定量在开发数据中形成清楚、可证伪的预测，下一阶段才谈独立验证；当前不保证会出现这样的量。新架构、插层搜索、8/32/128步共享更新及重做Atlas训练都不是本分析的自动后续。

## 7. 对当前稿件的实际增量

现在就能补强的是**定义与解释的精度**：明确原生算子的局部约束不同，区分现场计算与历史反馈，把作用链拆为可达性、读取、共享学习及任务收益。

现在不能补上的，是已经验证的“什么模型一定受益”规则。更具体的预测依据需要上述固定数据的测量；仅从源码得到一个符合结果的故事，不足以承担机制结论。

本轮先完善MD，并纠正旧数学笔记中的Atlas归类。没有将新的秩推导或假说自动写入摘要、贡献列表或Results；论文数值、来源锁与正式实验均保持原状。

## 8. 审计来源与边界

- **Protenix**：锁定完整推理执行文件与[执行锁](../evidence/full_cross_execution_lock.json)逐项匹配；上游protenix.py、pairformer.py、diffusion.py、triangular/layers.py与官方固定commit `d3b4db6a121dd4584edd85e93744239325a2b72e`及当前安装文件三方hash一致。Mini配置另由历史[cache index](../evidence/protenix_cache_index.json)核对；不声称未列入锁的配置文件有同样历史绑定。
- **OpenFold**：审阅归档commit `be2ec1841f16c966c65ae0e7599ebbadc725757d`、论文包的[运行入口](../reproducibility/src/engramfold/experiments/openfold_adapter_runtime.py)与live_opm/interface_heads。历史实现字节对照范围见[单目标复现审计](openfold_single_reproduction_20260924.md)，不把后增旋转G+代码冒称原始Train96全部源码。
- **AtlasFold**：历史`atlas_execution_lock.json`绑定上游commit `8ab3aca0e18c8b814d5ca6756b2617a07d72c68d`；本轮核对DiamondHill该checkout的6个核心文件与git HEAD一致，结合正式adapter入口和已归档配置重建调用关系。历史锁没有给这6个上游文件逐文件hash，当前checkout一致性不等于所有历史执行文件均有逐文件绑定；本轮没有重放完整Atlas预测。
- 局部秩结论为本轮代数推导，已按实际算子维数与顺序核对；实际权重谱、样本anchor、传播响应与其预测价值尚未测量。
- 只使用已纳入论文的结果判断效果；未查其他线程的新矩阵完成数，不将其算作支持或反证。
