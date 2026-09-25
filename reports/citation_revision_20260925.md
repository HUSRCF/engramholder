# 引用修订与来源核验（2026-09-25）

本轮依据用户提供的引用审查意见，修订当前匿名稿。基线为 `9c9c17b`；没有更新科学结果、数值来源锁或实验协议。

## 已落实

- Candido 条目补齐 `Bryan Z. Wu`、bioRxiv 出处及独立 DOI 字段。AtlasFold 的团体作者原本已使用 `{Team KAIST}`，继续保留，并添加对应 v2 链接。TM-score 条目补第4期。
- LoRA、BOFT、PiSSA、ReFT 更新为正式会议条目。LoRA、BOFT 的显示年份分别为2022和2024，后两篇仍为2024；旧 citation key 保留以避免破坏历史引用。
- 新增并实际引用 BLAST+、Holm、Adam、PDB、CATH、OFT、指数映射正交参数化，共七条。参考文献由14条增至21条。
- OpenFold 引文紧跟模型名，并把发现明确归属于本研究。Related Work 区分 OFT 的正交变换和 BOFT 的 butterfly 参数化；补偿公式附近单独引用指数映射方法。
- 保留 C$\alpha$ pair-averaged lDDT 变体定义；首次 Results 辅助指标写全 residue-average C$\alpha$ lDDT。没有把 smooth-lDDT 训练损失误改成评估指标。
- 修正数据说明的语义歧义：每个面板对每个 PDB entry 至多保留一条链，而非每个面板仅保留一个 PDB。

书目元数据和官方来源见[逐条核验记录](citation_bibliography_sources_20260925.md)。BOFT 官方会议 HTML 与 PDF 的末作者字段不一致，采用正式 PDF 的署名，没有机械复制网页元数据。

## 数据来源的边界

[数据来源审计](citation_data_provenance_20260925.md)记录原始下载凭证、文件头及七个已核实 SHA256。

PDB 集合的原始下载日期未在已检查记录中找到。正文明确说明这一缺项，并区分样本初次发布日期截止条件和数据库快照日期；未根据文件mtime或内部复制日期补造下载日期。

CATH 审计使用两类来源：2026-09-19获取的 daily 文件（HTTP Last-Modified为2025-01-15，含 `v4_3_0` 与 `putative` 标签），以及2026-09-20获取的 v4.4.0 chain/unclassified 清单（文件头日期2024-12-16）。附录分别说明，不将其包装为单一版本或严格家族隔离证据。BLAST程序版本按已存输出统一为2.17.0+。

## 验证与交付

- 21个唯一书目key全部在活跃正文／附录中使用，无未定义或未使用条目。
- 正式会议年份及 Bryan Z. Wu 已在生成的 `.bbl` 和PDF中核对。
- 65份当前论文输入在独立CPU目录中完成 LaTeX/BibTeX 编译；无未定义引用、重复标签或overfull。保留一条 `h` 自动改为 `ht` 的普通浮动体位置提示。
- 现有数值键保护的3项测试通过；v11锁的204个输入哈希、全部1830个数值对象、生成表图及官方样式保持不变。
- 局部视觉检查覆盖 Related Work、参考文献、新增数据来源说明和补偿公式引用。未进行最终页数预算检查、全稿视觉验收或更新匿名补充包。

反馈PDF：[paper.pdf](../build/feedback_v41_citations/paper.pdf)。验证记录：[draft_validation.citations_20260925.json](../notes/writing_branch_20260922/draft_validation.citations_20260925.json)。当前PDF为本轮源码编译结果，旧匿名包仍是历史快照。
