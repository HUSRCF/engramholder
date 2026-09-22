# 2026-09-21 主文重写与结果接入规则

本轮是写作重组，不增加训练、目标或评价终点。已完成证据先进入正文，AtlasFold 正式适配继续运行，固定位置等待完整矩阵。

## 主论证

原生更新与冻结下游的方向相容性能够影响受约束适配；效应依赖底座、训练配方与目标分布。方向控制中的优势，不保证原生 Factor 优于显式因子访问的 G+。论文不声称完整解释泛化、普遍家族隔离或全面部署优势。

## 正文结构和排版预算

[ICLR 2027 官方作者指南](https://iclr.cc/Conferences/2027/AuthorGuidelines)规定主文至多 9 页，参考文献与附录另计。下表是编辑预算，**不是已排版页数**。当前环境有 Pandoc，但未发现 pdflatex/tectonic；HTML 预览不证明符合页限。最终使用[官方样式包](https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip)编译核验。

| 内容 | 计划页数 | 核心对象 |
|---|---:|---|
| 标题、摘要、引言、相关工作 | 1.50 | 问题和条件性贡献 |
| 统一方法、方向控制、配置表 | 2.00 | Eq.1–2、Table1 |
| 评价设计 | 0.75 | 面板时序、统计单位 |
| Protenix 结构证据 | 1.00 | Table2 |
| 跨底座及长度范围 | 2.00 | Table3、Atlas固定插槽 |
| 系统位置 | 0.75 | Table4 |
| 机制边界、限制、结论 | 0.75 | 正负结果一并解释 |
| 排版余量 | 0.25 | 长表、浮动体与换行 |
| 合计 | **9.00** | 需正式模板实测 |

若超页，先压缩重复文字和系统背景、把配置细节留在附录；不删除 OpenFold 的零跨区间、反向种子或 G+ 竞争结果。Table1 可在排版时转置或分上下两块，以保持可读字号，不能靠缩小到难读字体满足页限。

## 正文与附录分工

| 正文保留 | 附录保留的完整记录 |
|---|---|
| 围绕原生算子的统一残差；现场 anchor 区别 | loss、mask、精度、优化器、源码和权重哈希 |
| Protenix Factor−Generic 与方向控制 | 原四格、U 蒸馏、预算曲线、逐种子统计 |
| 完整长链固定模型验证 | 序列筛查版本、漏检限制、672正式与28工程记录 |
| AF2 全部四个方向结果及 G+ | 三指标、配对交互、校准和两实现差异 |
| 原始系统绝对质量 | 原生输入、采样预算、完整成本尚待汇总项 |
| 一段机制证据与负结果 | 梯度 oracle、旧 S0 未通过、有限步未建立迁移、数值定位 |
| 一段长程收益描述 | 固定1–11/12–23/≥24分组、pair权重和headroom |
| 方法不含条件记忆 | 早期 Engram/Dense 对照失败与研究时间线 |

## 尚待填入的固定位置

**AtlasFold：**Train96、1536步、3 native＋9 rotations＋3 G+。主比较 Confirm96-B native−rotation；Length48/G+ 次要。完整表包含两面板的四类绝对均值、配对区间、种子边际及失败覆盖。正、负或不确定结果使用同一位置，不因结果修改摘要承诺或继续增加校准。原始 AtlasFold baseline 已完成，可作为系统参考，不能冒充适配对照。

首批评测存在跨机器绝对软链接失效。必须合并独立 recovery 的有效最终记录，保留原 FileNotFoundError 账本及 hash 修复证据；不能把旧工程零分当最终模型质量，也不能无记录覆盖失败。

**ESMFold2：**仅在固定144全部完成、统一评分后填系统表；部分成功子集不报均值。HIP重试保留原记录且复用校验通过的输出，不把重试数当样本数。

**成本：**新系统完整耗时/显存尚未统一汇总，不写资源竞争胜出。OpenFold3 已锁定 YAML 的完整 resolved config 在提交 artifact 中补齐，不能借用 Protenix 五步预算。

## 核验与提交前收尾

- `revision_evidence_lock_20260921.json` 留存本次正文/附录/证据哈希；原 `length48_evidence_lock.json` 保留为旧版历史锁，不覆盖。
- 本轮核对新表数值来源、OpenFold目标级差值和聚合、摘要一致性；不重新评分 CIF、不重抽样选择区间。
- 当前主文约3.6k空白分词，不能据此断言9页内。正式模板编译、参考文献格式与最终分页仍待完成。
- 内部附录含机器路径和运行标识。匿名提交副本需移除身份线索，保留可复现的匿名相对路径；内部原记录不删除。
- 检查官方要求的 AI 使用声明。以下只作如实的作者核对草稿，不声称已经完成人工复核：

> AI assistants were used to assist with code development, experiment operations, analysis and manuscript drafting. The submission should identify their actual roles and the verification performed by the authors; all scientific claims remain the authors' responsibility.

声明应据实际参与范围定稿，不能写成“仅语言润色”。本次未向任何外部投稿系统提交、改写或发送文件。
