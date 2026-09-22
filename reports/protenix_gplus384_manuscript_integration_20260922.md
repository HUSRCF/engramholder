# Protenix 匹配 G+：论文整合记录（2026-09-22）

已将完成结果正式补入正文、生成表格和附录，未新增实验、改动主终点或重新评分 CIF。当前仍是作者待审阅的本地草稿，未 commit / push。

## 本次修改

- 主表补齐 Train384 的 G+：Confirm96-B **0.54942**，Length48 **0.44781**。Train96 未开展的匹配 G+ 仍为 `---`，不以旧 Generic 代替。
- 相邻配对表加入 Native−G+：**+0.07630 [0.05677, 0.09795]** 与 **+0.08138 [0.05968, 0.10571]**。
- Results 同时交代 78/96、43/48 个目标改善、三个配对种子均正，以及 G+ 本身相对 Query 的均值改善。
- 附录新增六项指标比较、中位数与完整运行审计；摘要、Introduction、Discussion、Conclusion 同步呈现 Protenix 与 OpenFold 的竞争结果差异。
- 明确 Confirm96-B 为此次后续比较主要终点，Length48 为次要终点；两者均为已观察面板。原 Length48 方向比较的首次锁定身份不变。
- 保留公共学习率/固定预算限制，以及长链历史 H100 与新 G+ MI250 的后端差异。有限回放不是全目标数值误差上界。
- 本次没有 Protenix 旋转 G+，不新增 Protenix 头×旋转交互结论；OpenFold 四格、AtlasFold 近零结果完整保留。

## 证据与复算

输入采用新增的 [v2 锁](../notes/writing_branch_20260922/paper_sources.v2.lock.json)，[迁移记录](../notes/writing_branch_20260922/gplus_lock_migration.json)确认旧来源哈希未改变；v1 锁与旧编译记录保留。

生成脚本读取原始目标分数，独立重建两个面板×三种指标的配对差值和 20,000 次 bootstrap（沿用 seed20260926），与锁定报告误差小于 1e-12。继续检查既有 348 个系统×指标均值与 144 个 OpenFold 逐目标交互。

[来源映射](../generated/cell_sources.json)现在包含 153 个数值字段。正文、图表共同从这些字段读取；没有手工改写生成数字。

## 排版与候选材料

- [新版 PDF](../build/iclr2027_conference.pdf)：正文结束第 **8 页**，含声明、参考文献和附录共 **13 页**。
- 无未定义引用或 overfull；8 项 underfull 警告。第 7 页两张主表已做视觉检查。
- 官方 sty/bst/math_commands 与提交版本一致，匿名模式保留。
- [v4 匿名材料候选](../build/anonymous_artifact_v4.zip)：71 个文件，匿名化后重新生成并核对 153 个数值完全相同；5 项算子检查通过。
- 候选包仍是分析与算子级复现材料，不是完整重训练包；不包含全部权重、结构数据或一键训练环境。
- [本轮验证记录](../notes/writing_branch_20260922/draft_validation.gplus.json)。

## 主张变化

新结果补齐了 Protenix 的显式因子访问通用头对照：在匹配配方下，Native 的优势同时相对 Rotated 与 G+ 成立。这强化 Protenix 方法竞争证据，但不使“Native 普遍优于通用头”成立。跨底座的条件性仍是正文中心，而非需要补救的例外。
