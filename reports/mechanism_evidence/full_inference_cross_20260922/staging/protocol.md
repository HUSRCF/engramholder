# Authorized execution supplement, 2026-09-22

User subsequently released DiamondHill for this task. Fixed existing24-target
candidate adopted without changes. Execute all720 new predictions on the existing
MI250/fold environment; no historical baseline reuse. Eight logical GPUs available
at launch. 6h dispatch cap; HIP-only retries twice with120s delay, preserved logs.
Stop on undefined norms or engineering gate failure. No training and no outcome-
based selection. Bootstrap and metrics follow the candidate below. Candidate's
"not started" labels describe its earlier status; execution_lock is authoritative.

# Protenix 完整推理：方向来源 × 幅度来源交叉候选

状态：**2026-09-22 源码与记录核对完成；候选未锁定、未启动。**
本文件细化 reviewer 建议，不是新队列授权，不改变正在执行的 OpenFold A/B。
不新增训练。若选择实施，仅测试一套 Mini/Train384/Full/1536，不能将它
写成 Train24/Tangent/384 的 v4 同模型续证。

## 1. 从已有记录中确实知道什么

v4 正式结果及 completion_audit 已完成；旧 README 的 active 状态过时。
其 E2 主要 learned equal-norm loss contrast 为 +0.000021771，95%CI
[-0.000034906,+0.000077741]，未建立稳定优势；实际幅度 loss contrast
+0.041534 是另一干预尺度。E3 最后 recycle 保留效应也不是完整自由采样。
这些结果提供本候选的动机，不能推出新实验必然存在方向或幅度优势。

本地正式推理实现已核对：`evaluate_native_geometry.py` 在 `runner.predict`
前生成一次 update 并调用 `make_replay_module`。`StaticOPMProvider` 缓存的是
**完整的 Uq+δ**，每轮返回同一个张量；`InjectedMSAModule` 加到当前 pair state
之后继续执行原 pair stack。静态的是注入更新，下游状态仍逐轮变化。
`NativeGeometryHead.effective_weight` 已包含旋转 R 与 D/(D+eps)；不能再旋转一次。

## 2. 正确的四格与残差构造

固定目标 i、训练种子 s，配对该种子的 Native 与一个 Rotated-k 模型。
从各自最终 checkpoint 的实际部署路径得到完整 pair 残差 δN、δR，
保留旋转模型训练时固定 R；不能将三种旋转的更新先平均。

令 mN=||δN||F、mR=||δR||F，dN=δN/mN、dR=δR/mR。

| 格子 | 方向来源 | 幅度来源 | 注入的完整 update |
|---|---|---|---|
| NN | Native | Native | 原始 UN=Uq+δN |
| NR | Native | Rotated-k | Uq+(mR/mN)δN |
| RN | Rotated-k | Native | Uq+(mN/mR)δR |
| RR | Rotated-k | Rotated-k | 原始 UR=Uq+δR |

**形成 Full pair 残差之后再缩放。** 不缩放因子增量；Full 包含二次项。

实现时使用与原 `NativeGeometryHead.forward` 相同计算顺序：
`δ = bilinear(da, query_b+db, W_eff) + bilinear(query_a, db, W_eff)`。
不能以 tangent+quadratic 的重新结合或 `writer_output-Uq` 取代它，再默认
浮点结果完全相同。可通过只读捕获这份残差或独立 helper 验证实现。
**NN/RR 直接使用原 forward 完整输出，不先相减再相加。**
缩放=1的混合分支应作为工程回放检查，保留实际误差。

范数使用实际完整输入的所有有效 residue-pair/channel（包含对角线）；
若有padding，只用输入token有效性mask，禁止使用真实结构可观测CA mask。
单链完整序列无padding时即整个 L×L×C 张量的未加权 Frobenius 范数。
建议将已形成残差转FP64累加范数和计算标量，再按固定FP32注入路径使用；
记录cast后的实际范数、方向余弦及相对误差，不声称浮点下严格等范数。
不做逐通道重加权，不对norm ratio裁剪，不按GT翻转方向。

输入manifest只包含ID/完整序列/序列hash/长度分层。参考结构路径和reference mask
单独提供给评分进程；推理维持已有 evidence-read guard。标签不参与幅度、方向、
目标选择或采样输出选择。

## 3. 不应漏掉的解释边界

- “幅度来源=Native”不意味着数值一定更大；逐配对记录 mN/mR 与log比值，不能把
  Eamp无条件叫作增大幅度的效应。不可事后重排成高/低幅度来取代主四格。
- 方向来源包含独立训练学到的内容及通道组织差异，不是同一向量的纯旋转。
- 混合更新可落在训练未经历的状态分布中；严重退化只能按该干预解释。
- 配对中的NN在三个R中相同，可共用一次真实预测；这不产生三份独立Native样本。
- 所有结论限于该24条已观察目标和固定模型；这不是盲确认，也不识别训练机制。

## 4. 主要读数必须与条件效应一起展示

记 Y 为固定对应关系的结构分数。

`D_mN = YNN - YRN`：在 Native 实际幅度上的方向来源效应。

`D_mR = YNR - YRR`：在 Rotated 实际幅度上的方向来源效应。

`Edir = (D_mN + D_mR)/2`：唯一拟议主要终点，Cα pair-lDDT。

`A_dN = YNN - YNR`；`A_dR = YRN - YRR`。

`Eamp = (A_dN + A_dR)/2`；`I = D_mN-D_mR = A_dN-A_dR`。

必须校验 `Edir+Eamp=YNN-YRR`。这是一种四格对称分配，不是天然可加的机制贡献。
结果表同时列四格绝对分数、两个D、两个A、Edir/Eamp/I与mN/mR；
若两个D符号相反，正文优先说明条件性，不只展示对称平均。

每目标先对种子与三个R聚合，再对目标bootstrap。建议按四个固定长度层各6条
分层重抽样，20000次、seed20260926；保留种子/旋转边际及全部24目标。
这是候选统计规则，须在预测前写入正式锁，不能从新分数中选择bootstrap方式。
其余对比和逐残基Cα-lDDT、固定对应TM-score列为次要，区间不作为多重独立阳性。

## 5. 目标与预算：24不是从v4的合并manifest直接抽

核对发现：v4本地 `manifest.json` 有200条，包括其他用途目标；不能直接取它的前24条。
正确来源是 Train384 的 `observed_manifest.json`，确有 Confirm96-B 96条，
四个历史长度层各24。沿用实际标签128–191、192–255、256–319、320–384。
候选按 `sha256('full-inference-direction-amplitude-v1|'+target_id)` 每层前6条选取，
只读ID与既有层标签，不读分数。已生成可审查的候选输入24条，仍标为未锁定。
不根据范数、旧模型效果或新结构得分替换这些目标。

12个checkpoint均存在于既有Train384评测锁，全部Full/1536、三seed×四方向。
开始前再次核验实际文件hash、冻结Mini/ESM2、decoder D=508.5来源和旧推理配置。

- 0新训练；先计算288份残差（24×12）用于输入/范数工程审计。
- 432个混合预测（24×3×3×2）。
- 288个纯Native/Rotated预测（24×(3+9)）。
- **默认安排720次正式预测上限**，工程检查和有限重试另记。

只有checkpoint、特征、源代码、配置、采样条件、数值后端与原文件均可配对核实时，
才允许复用288个历史基线而减少为432。两个目标的坐标近似回放通过，不能单独证明
跨后端的96/24全体结构分数可互换。尤其若新运行转H100而旧预测在MI250，默认重跑
所选24条的纯基线，与混合四格处于同一固定后端。不混用有利的历史NN或RR。

## 6. 工程与停止规则

先只读审计全部288份残差。任一严格零范数：方向未定义，保留记录，停止整批候选
进入正式预测，不以epsilon构造方向，不换目标；此时需另行修订协议而非自动删样本。
非有限值或混合后溢出属于工程失败，同样不裁剪/放宽定义。无需微小loss有限差分。

随后以候选清单中预定最短/最长两条工程检查：原生NN/RR回放、共同query baseline、
混合前后范数/方向余弦、完整序列、恰好4次pair-stack调用、c4/s5/1sample、实际采样
随机条件或RNG消费一致、参数hash不变。source切换与张量构造不得消耗采样随机流。
同seed重设是必要记录，不单独当作随机张量完全相同的证明。

所有正式任务先锁目标、模型、统计、数值设置、错误重试上限，完成输出后统一评分。
真正推理失败沿用固定分母与计零；方向未定义与推理失败分开报告，不能将未定义干预
直接当零分。没有结果后加目标/加倍率/切分recycle的自动续接。

## 7. 当前决策建议

认可这是有信息增量的候选，优于继续微小梯度或共享更新步数搜索。
但它不填补新的架构/PLM证据；现有OpenFold A/B已实际运行，应先保障这些任务。
本次只完成候选核验与细化；未启动残差生成或720预测，也不改写原论文结论。
