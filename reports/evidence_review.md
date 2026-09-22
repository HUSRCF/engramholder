# 证据审阅报告：先理解结果，再决定论文怎么写

状态：供用户核对的中文研究报告，**不是论文正文、摘要或定稿承诺**。
本版在2026-09-21原有快照上补做覆盖、方法与稳健性审计；准确生成时间及输入哈希见[审计JSON](coverage_robustness_audit.json)，不代表远端队列的实时状态。
本轮报告工作不重新评分 CIF、不修改统计终点、不开展新实验。另行授权的训练进度由其独立方案记录；本报告只计入已经完整收集的结果。JSON 快照位于 `evidence/`。

本次新增：[可审查的数学定义](math_contract.md)、[种子/旋转/逐目标/时序与暴露审计](robustness_audit.md)、[复算脚本](../scripts/audit_evidence_coverage.py)。

**9月22日更新：** [新完成结果与完整统计](results_update_20260922.md)。OpenFold旋转G+交互成立；AtlasFold两个面板未建立原生方向优势；ESMFold2完整144评分完成。以下旧证据不改写，新结果以该更新为准。

**9月22日13:01后续更新：** [Protenix匹配Train384 G+已完成](protenix_gplus384_results_20260922.md)，432/432预测成功。下表两格补齐；这是既有面板上的后续比较，不改变原确认实验身份，亦不修改论文生成数据锁。

## 1. 先区分三个问题

| 问题 | 应看哪个比较 | 为什么不能互相替代 |
|---|---|---|
| 加入适配器是否有用？ | Native、Rotated、G+ 各自相对 query-only | 新 PLM 信息与适配结构一起改变，全部收益不能归给原生方向 |
| 同一受约束构造的方向是否重要？ | Native−已训练 Rotated | 旋转组也训练，控制因子形式与参数量；不等于控制整个训练优化问题 |
| 原生头是否是更好的方法？ | Native−显式因子访问 G+ | G+ 更灵活，方向对照赢了仍可能输给 G+ |

Native 为原生方向的 Factor；Rotated 为正常训练的固定正交输出旋转对照。
G+ 是显式接收相关 query factors 的通用头，与旧 Generic 不是同一个对照。
Train96/384 指训练蛋白数量；Confirm96-B 指评测面板，不能只用“96”称呼。

所有下面的区间都以目标为统计单位，条件于这些已拟合模型。
三个训练种子与三个旋转不是九倍独立蛋白。后续分析的区间未作多重比较校正。

## 2. 同一序列输入条件下的完整覆盖表：空格不借用其他配方填补

这里的“同信息”指不使用测试homolog/模板/结构输入，**不是所有系统具有相同中间特征**：Query-only没有新接入的ESM2-35M；Native、Rotated、G+均有这个额外表征，G+还显式接收query factors。严格控制方向应看Native−Rotated；相对Query-only的总增益不能单归给方向。官方系统原生PLM不同，另列系统参照。

每格为Cα pair-lDDT均值。**✓**表示已完整评分；“缺”表示本配置/面板没有完成且可匹配的结果，不填零。Native平均3种子，Rotated平均3种子×3旋转，G+平均3种子；原始基线无需重复当三个训练实例。

| Protenix 配方 / 评测面板 | Query-only | Native | Rotated | G+ | 说明 |
|---|---:|---:|---:|---:|---|
| Mini原Factor/旧Generic研究，Train24/384，Confirm96-A | ✓0.37587 | ✓0.56241 | 缺 | 缺 | 旧Generic ✓0.49725另列，不能充作G+ |
| Mini tangent Train24/384，Confirm96-B | ✓0.37931 | ✓0.53189 | ✓0.47479 | 缺 | 锁定方向主比较 |
| Mini full Train24/384，Confirm96-B | ✓0.37931 | ✓0.55060 | ✓0.50410 | 缺 | 同研究预设次要构造 |
| Mini full Train96/1536，Confirm96-B | ✓0.37931 | ✓0.61653 | ✓0.59148 | 缺 | **Train384交互的同构造参照** |
| Mini full Train384/1536，Confirm96-B | ✓0.37931 | ✓0.62572 | ✓0.58621 | ✓0.54942 | 匹配G+已完成；Native−G+ +0.07630 [0.05677, 0.09795] |
| Mini full Train384/1536，Length48 | ✓0.32122 | ✓0.52919 | ✓0.48667 | ✓0.44781 | 后续匹配G+；Native−G+ +0.08138 [0.05968, 0.10571] |
| Tiny tangent Train24/384，Confirm96-B | ✓0.31497 | ✓0.45606 | ✓0.43400 | 缺 | Tiny自己的query/decoder |
| Mini tangent Train24/384，均值保持旋转，Confirm96-B | ✓0.37931 | ✓0.53189 | ✓0.43130 | 缺 | 固定特殊旋转族，非普通旋转 |
| **单独G+研究**：Mini full Train24/384，Confirm96-B | ✓0.37931† | ✓0.55314 | 缺 | ✓0.50084 | 独立种子及各头选定LR，不能与上方拼成四格 |

† Query-only为同底座、面板、推理配方的既有基线复用，不属于G+研究的576次新预测。G+研究实际是6 checkpoint×96=576。

最后一行核对自[完整G+分析](../evidence/protenix_gplus.json)、[576条原始记录](../evidence/protenix_gplus_records.json)和[评测锁](../evidence/protenix_gplus_lock.json)：种子20260920/21/22，Factor与G+在同一Dev8受限搜索后各自选择5e-5和1e-4。方向主矩阵使用种子20260923/24/25与共同5e-5。因此0.55314不是0.55060的重复记法，0.50084也不是Train96/384的G+值。这个比较可以评价已选配方的实际效果，不能单独估计信息访问的因果贡献，也不是完全同超参数比较。

Protenix G+另有Train24/Dev8在768、1536步的开发预算曲线；**那些是训练/开发评分，不是Confirm96-B或Length48缺项的替代品**。没有同配置的四格就不计算对应头类型×旋转交互。

以上均值从现有逐目标记录核对；覆盖及分布复算见[完整审计](coverage_robustness_audit.json)。OpenFold四格在下一节完整列出。

### 2.1 哪些比较有直接区间证据？


| 比较 | 主指标平均差值 [95% CI] | 证据身份 |
|---|---|---|
| 原 Factor−Generic | +0.06516 [+0.04936, +0.08288] | Confirm96-A 原锁定主比较；[protenix_generic](../evidence/protenix_generic.json) |
| Mini 一阶 Native−Rotated | +0.05710 [+0.04249, +0.07274] | Confirm96-B 原锁定主比较；[protenix_direction](../evidence/protenix_direction.json) |
| Mini完整 Train24/384 Native−Rotated | +0.04650 [+0.03118, +0.06336] | 同方向研究预设次要比较；[protenix_direction](../evidence/protenix_direction.json) |
| Mini完整 Train96/1536 Native−Rotated | +0.02505 [+0.01378, +0.03730] | 已观察面板后续对照；[protenix_extensions](../evidence/protenix_extensions.json) |
| Tiny 一阶 Native−Rotated | +0.02206 [+0.01397, +0.03075] | 复用已观察面板，第二 checkpoint；[protenix_extensions](../evidence/protenix_extensions.json) |
| Train384 完整 Native−Rotated | +0.03951 [+0.02800, +0.05216] | 复用已观察面板，固定更新预算扩展；[protenix_train384](../evidence/protenix_train384.json) |
| 固定 Protenix Train384，Length48 | +0.04252 [+0.02899, +0.05765] | 对该实验预先锁定的新长链面板；[length48](../evidence/length48.json) |

Length48 实际目标长度为392–750；适配训练长度为152–383。它检验的是**适配器训练长度范围之外的完整输入**，不是基础模型从未见过长蛋白，更不是严格家族外推。

Length48 的绝对均值如下；Native−Rotated 是锁定主比较，其余这里用于描述系统位置。

| 系统 | Cα pair-lDDT |
|---|---:|
| Native | 0.52919 |
| Rotated | 0.48667 |
| Query-only | 0.32122 |
| 官方 Mini-ESM | 0.92139 |

**能支持：**该设置下，两类适配系统的平均分都高于 query-only，原生又有额外优势。
**不能支持：**任意长度泛化、恢复真实协变、长链上优于所有 Generic、或接近官方系统的部署竞争力。9月22日后续匹配G+已完成，支持优于本次G+；不能泛化为优于所有通用头。见[新增G+结果](protenix_gplus384_results_20260922.md)。

## 3. OpenFold：方向作用与方法竞争力必须分开看

以下各数值格均已完整评分，Native/Rotated/G+为同底座同新增ESM输入的正式配置；Query-only仅有原主干输入。数值从模型均值聚合，并从逐目标原始记录独立复算差值。
两个面板在开展 AF2 研究时均已观察，不能再次称为新盲确认。

| 训练集 | 面板 | Query | Native | Rotated | G+ | Native−Rotated [95% CI] |
|---|---|---:|---:|---:|---:|---|
| Train96 | confirm96 | 0.30191 | 0.50190 | 0.48209 | 0.49534 | +0.01981 [+0.00928, +0.03129] |
| Train96 | length48 | 0.26079 | 0.38872 | 0.37834 | 0.39849 | +0.01038 [+0.00403, +0.01705] |
| Train384 | confirm96 | 0.30191 | 0.50684 | 0.50381 | 0.51333 | +0.00303 [-0.00675, +0.01218] |
| Train384 | length48 | 0.26079 | 0.40165 | 0.39040 | 0.40868 | +0.01126 [+0.00527, +0.01734] |

来源：[Train96](../evidence/openfold_train96.json)、[Train384](../evidence/openfold_train384.json)。

这里至少有三个不同结论：

- 适配接入有效：三类适配器平均均高于 query-only。
- 方向效应有条件：Train384 的 Confirm96-B 未建立原生优势，三种指标区间均跨零；这也不等于证明等效。
- Factor 并非始终优于 G+：长链上的竞争边界有直接配对结果，不能只看 Native−Rotated。

| 面板 | Native−G+ | 平均差值 [95% CI] |
|---|---|---|
| confirm96 | train96 | +0.00656 [-0.00515, +0.01845] |
| confirm96 | train384 | -0.00648 [-0.01777, +0.00537] |
| length48 | train96 | -0.00977 [-0.01674, -0.00288] |
| length48 | train384 | -0.00703 [-0.01369, -0.00039] |

来源：[已有完整分数的后续配对分析](../evidence/openfold_paired.json)。
这些区间未校正多重比较；它们限制了“Native 普遍更好”的说法，不能解释成已经知道 G+ 学到了何种机制。

## 4. 扩大训练集合：两个底座的交互不一样

这里检验的是 `(Native−Rotated)384 − (Native−Rotated)96`，不是把两个区间相减。**Protenix两边都是Full、1536步**：Train96为0.61653−0.59148=0.02505，Train384为0.62572−0.58621=0.03951。Train24 tangent/384的0.05710没有参与交互，不能用它替代Train96完整构造。

| 底座／面板 | 交互 [95% CI] |
|---|---|
| Protenix / Confirm96-B | +0.01446 [+0.00350, +0.02586] |
| OpenFold / confirm96 | -0.01678 [-0.03214, -0.00261] |
| OpenFold / length48 | +0.00087 [-0.00728, +0.00863] |

来源：[Protenix 交互](../evidence/protenix_data_interaction.json)、[OpenFold 交互](../evidence/openfold_paired.json)。

**数据直接告诉我们：**同一固定预算下，扩大训练集合对两种方向的相对影响依赖底座与面板。
**尚未知道：**这种差别具体由注入位置、下游结构、优化、训练样本组成中的哪一项导致。
不能据此写成“AF2 能纠正旋转”或“Protenix 学到了更多进化知识”。1536步下，96与384条蛋白的平均重复次数也不同。

## 5. 系统参考不能算成方向复现

| 原始系统 | Confirm96-B | Length48 | 目前证明什么 |
|---|---:|---:|---|
| AtlasFold | 0.95676 | 0.95050 | 原始系统固定单序列设置的质量；[atlasfold_system](../evidence/atlasfold_system.json) |
| OpenFold3 / OpenBind-0 | 0.36091 | 0.30814 | 原始系统固定单序列设置的质量；[openfold3_system](../evidence/openfold3_system.json) |
| RF3 Benchmark | 0.41558 | 0.34672 | 原始系统固定单序列设置的质量；[rf3_system](../evidence/rf3_system.json) |

上述三个系统均有完整144条评分，但这些只是 baseline，不含各自完成的 native／rotation 训练矩阵。
9月22日：AtlasFold正式15组/2160预测与ESMFold2完整144评分均完成，见[新结果](results_update_20260922.md)。AtlasFold两面板未建立原生方向优势；ESMFold2仍只是系统参考。

这些系统的原生 PLM、预训练数据和采样预算不同，不是同计算量排行榜。
OpenFold 与 ColabFold 在这里运行同一 AF2 权重，属于实现／管线比较，不是两个架构。

## 6. 方法上的共性与不同：先把实验对象理解清楚

完整可审阅定义见[数学契约](math_contract.md)：逐残基ESM特征→可训练编码器→两组因子增量；冻结query factors与原生decoder组合出残差；仅旋转pair通道残差，再加回未旋转baseline。该笔记给出Full/Tangent展开、零初始化、G+实际输入、冻结边界及精确控制范围。

Protenix的D=508.5已从原Train24缓存的24个teacher_depth独立复算，中间两项508/509；它是训练侧历史MSA深度常数，不是测试MSA输入，也不会替换depth=1的原生query baseline。Train96/384沿用此值，不重新按测试质量选取。

共同原则是围绕冻结模型已有的算子构造零初始化残差，然后对残差施加固定输出旋转。
零初始化保证初始 baseline 一致；在共同参数和输入状态下保留谱与范数性质。
训练后参数和现场状态发生分叉，不能声称完整优化过程仍然等价。

| 项目 | Protenix | AF2 / OpenFold | AtlasFold（完整结果见更新） |
|---|---|---|---|
| 原生组合 | OPM | OPM | difference/product，非 OPM |
| 因子来源 | 缓存 query factors | 当前首个 Evoformer OPM 的现场因子 | 当前首个 LM-stack 算子的现场因子 |
| 因子宽度 | 32 | 32 | 128 |
| 原生 PLM | 适配的是 default checkpoint；额外 ESM2-35M | AF2无原生PLM；额外 ESM2-35M | 保留 AtlasLM-3B，另加适配器 ESM2-35M |
| N/R 可训练参数 | 373,824 | 373,824 | 423,168 |
| G+ 参数 | 378,144 | 378,144 | 415,008 |
| 归一化 | 残差D=508.5，独立保留原始query baseline | 本层原生 mask/row-count，不能复制508.5 | 原生输出投影，无OPM深度常数 |
| 推理调用 | 4 recycles / 5 diffusion steps | 4次trunk；structure module | 参数num_recycles=4实际5次trunk；20/30 diffusion steps |

这张表是既有配置审计摘要，详细源码与执行锁仍在实验仓库。
暂存的[配置记录](../notes/source_appendix.md)包含准确字段与哈希，但它的叙述是助手旧草案，不等于用户已经认可的论文文本。

### 6.1 均值之外，当前一致性实际到哪里？

本轮从原始记录按目标ID复算12项比较，差数组与原报告最大误差均<1e-12。完整三种子、三旋转、3×3单元、目标P10/P25/中位数/P75/P90及逐目标值见[稳健性审计](robustness_audit.md)，没有按效果删样本或重新选分组。

- Protenix Train384/Confirm96：72/96正向，中位数+0.02288；Length48：41/48正向，中位数+0.02288。均值更高说明正尾部有贡献，但不是仅由极少数目标支撑。
- 这里核查的七个Protenix方向比较，9个种子×旋转单元均为正；这只覆盖已测三种子/三旋转，不代表任意重训稳定。
- OpenFold Train384/Confirm96：50/96正向、一个种子边际负向、9个单元只有6个为正，均值+0.00303的跨零结论不能隐藏。
- OpenFold Train384/Length48：三个种子和三个旋转**边际**均正，但9个交叉单元仅8/9正。边际一致不能升级成每个模型实例一致。

目标bootstrap区间条件于固定拟合模型；它不覆盖有限种子估计、旋转分布、重复使用面板的方法选择、未认证的家族相关性。当前结果支持有条件的规律，不能通过换一种汇总称为普遍稳定。

锁定时序和累计暴露见同一审计：原Confirm96-B方向主研究之后的Tiny、Train384、G+、OpenFold都明确是后续研究；Length48对当时固定Protenix是新锁面板，但随后对OpenFold已观察。384训练+8开发+192旧确认的584条旧inventory不包括后来48条；加入后已列query总计632，仍不是全部历史teacher/选择暴露的完整认证。旧104条CATH审计有12未解决，新增288训练链未审计，不可宣称严格家族隔离。

## 7. 已完成的机制研究：保留边界，不换成正结论

这一节依据已经整理的[机制记录](../notes/source_appendix.md)做状态索引；本轮未重新加载梯度张量或复算有限差分。

| 研究 | 已有结果的范围 | 不应推出 |
|---|---|---|
| 逐目标局部梯度 oracle | 原生局部相容性存在平均优势，但未建立对训练后逐目标结构收益的正关联 | 已解释全部折叠泛化机制 |
| 最初共享微分 S0 | 部分响应接近数值可分辨范围，原门槛未通过 | 共享学习无效，或门槛已经通过 |
| Train8→Dev8 一次有限更新 | 未建立平均正迁移或原生优势；不是单纯沿用S0数值失败 | 所有共享适配都不能泛化 |
| 序列间隔分解 | Train384收益更多集中在≥24关系；固定分组为1–11/12–23/≥24 | 恢复协同进化、跨结构域改善或所有配置天然长程专长 |
| CATH隔离 | 历史注释与累计暴露仍不完整 | 已证实家族隔离或基础模型预训练隔离 |

## 8. 仍待补齐的是哪些记录？

- **AtlasFold完整适配结果：已完成。**15组最终2160评分齐全，已纳入9月22日结果更新，未建立原生方向优势。
- **Atlas首批恢复来源：已核验。**720项原工程失败与5组独立recovery保留，最终2160均成功，预测/数据/来源hash通过。
- **ESMFold2完整评分：已完成。**144均成功；缺失评分锁的补齐时间与继承算法记录保留，HIP重试不增加独立样本数。
- **新系统成本：**统一端到端耗时、峰值显存与初始化范围尚未齐，暂不作部署优势判断。
- **累计数据暴露：**Train384必须计入；旧104条注释审计不能代表全部暴露。

这些是证据记录的待办，不是新实验立项，也不要求为已有结果再增加通过门槛。

## 9. 用户理解后再讨论的三个取舍

1. 你更希望把研究定位在“一个适配方法”，还是“方向约束何时有效的受控实证研究”？OpenFold 的 G+ 结果使这两种定位有明显区别。
2. AtlasFold完整矩阵没有建立方向优势，应作为非OPM扩展的边界；是否仍投入其他接口，不应仅因本次未为正就自动触发。
3. 哪些技术细节你希望先进一步解释：现场anchor、同谱控制究竟控制了什么、还是目标bootstrap区间的含义？

这些问题留供你阅读后讨论；助手不提前代选叙事或生成正文。下一步仍是完善报告和回答疑问。

## 独立新增对照的进度入口

[旋转可吸收的 G+ 对照方案](rotatable_gplus_control.md)用于直接检验头类型×旋转交互。该实验9组训练和1296预测已全部完成，直接交互与完整四格见[9月22日结果更新](results_update_20260922.md)。
