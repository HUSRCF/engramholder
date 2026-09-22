# 正式审读修改与数值键保护（2026-09-23）

本轮不新增训练、结构预测、评分或科学比较；既有246个数值的原值逐一与上个提交一致，证据锁仍为v3。

## 已落实

- 贡献段第三条明确包含 norm-controlled full-inference test，并保留跨底座/规模/长度范围，不增加新的贡献编号。
- 结果末节改为 **Competitiveness and limits across predictors**，让Protenix的强G+对照与其他底座边界共同呈现。
- 主文加入混合残差是推理时干预、不必对应训练头输出或训练遇到状态的说明。
- 删除解释章节先后顺序的内部叙事，保留原研究与观察后续研究的身份。
- 压缩摘要，保留OpenFold交互数值、长度外推及整体范数干预；使用“global Frobenius norm alone does not explain the observed gap under this intervention”限定结论。
- AI/伦理声明中的“待作者审核”等工作备注移入[作者清单](../notes/writing_branch_20260922/author_submission_checklist.md)。这不表示审核已经完成；真实复现限制仍在正文声明中。

## 防止静默丢数值

LaTeX `\result` 宏对不存在的键显式发出 `PackageError`。生成脚本同时扫描入口、正文、附录与生成表格中的字面数值引用，缺失键会列出文件及行号；空扫描不能直接通过。

三项针对性测试覆盖正常引用/转义百分号/注释、拼错键定位和空扫描；全部通过。临时最小LaTeX文档故意引用不存在的键，编译按预期失败。正式稿编译成功。扫描不是通用TeX解释器；动态构造键的路径由LaTeX运行时报错保护。

[验证记录](../notes/writing_branch_20260922/draft_validation.editorial.json)。v7匿名分析候选包包含该测试，重新生成246个字段并核对原值不变，既有5项算子检查通过。

## 保留的未完成事项

单目标完整预测入口尚未验证；见[实际代码依赖核对](reproduction_entry_audit_20260923.md)。未将分析/算子复现升级为完整推理复现。作者还需确认声明、许可和匿名二进制内容。

按用户要求，本轮只要求编译无错误，不继续读取最终页数、查看图表或审查PDF视觉排版；由作者在Overleaf完成。没有宣称最终页数或匿名性已通过新的检查。
