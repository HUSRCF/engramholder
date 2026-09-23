# ESMC审阅落实（2026-09-23）

本轮为52dde3b之后的写作修订；不改变实验、统计终点、输入证据锁或443个数值字段。没有加入B组、Fresh96、SGDM或未完成DiamondHill结果。

## 关键修改

1. 正文新增三行交互表：Train96/ESM2、Train384/ESM2、Train384/ESMC，各含两个面板的完整区间。摘要、引言、Discussion和结论均限定旧正交互的配方；明确两个Train384条件均未建立交互，不归因于ESMC，也不声称三个交互彼此显著不同。
2. 明确旧数据规模差中之差是Factor方向差Delta_F的变化，不是Psi的变化。
3. 删除“separates feature quality”的因果措辞；主文同时披露PLM家族、宽度、缓存精度与可训练输入投影参数一起变化。没有增加同参数量、同精度或层搜索的新实验。
4. 范数小节改为直接观察的标题，并恢复两个条件效应：+0.04669、+0.03967。完整四格仍留附录。
5. 统一主表、图和附录的F/RF/G+/RG+；内部Stage A首次用描述性名称解释。历史区间的bootstrap流差异在新交互表和附录表注就近说明，历史证据锁不覆盖。
6. 范围图加入ESMC两个方向对比，旧OpenFold行标注ESM2；原交互图明确Train96/ESM2。图与表区分Delta_F和Psi。
7. 附录保留ESMC/Confirm96逐残基lDDT的未校正区间略高于零，不用它替换主终点，也不说所有指标都跨零。
8. 修正“a assessment”，断开过长贡献句。正文保留绝对质量主表；完整Native−G+配对表移至附录以减轻重复，所有比较仍可查阅。

## 模型引用

ESM2的BibTeX与完整作者列表来自[Meta官方ESM仓库](https://github.com/facebookresearch/esm#citation)；ESMC来自[官方ESM仓库引用段](https://github.com/evolutionaryscale/esm#citations)，与[官方模型页面推荐文献](https://www.evolutionaryscale.ai/blog/esm-cambrian)一致。未自行拼接作者名单。bioRxiv网页抓取返回403，因此元数据取官方仓库提供的BibTeX，访问源与哈希保存在 `notes/writing_branch_20260922/reference_sources_esmc_20260923.json`。

## 检查与边界

- 本轮重新运行表图生成：78项A组对比重建通过，443个数值及来源与提交52dde3b完全相同；v4输入锁不变。
- 数值键3项测试通过。官方样式文件未变。
- Tectonic编译成功，无未定义引用或overfull；本轮实际主文止于第9页，不沿用旧版页数。
- 已查看新增主文表所在第7页、结论第9页与更新范围图，无可见截断或重叠；不是对全PDF的逐页匿名/视觉认证。
- 完整单目标“序列→特征→CIF→评分”干净环境入口仍未完成；这次不修改原线程模型运行环境或将分析包说成端到端复现。
- 未提交或推送。新版PDF在 `build/iclr2027_conference.pdf`。

匿名分析包 `build/anonymous_artifact_v10.zip` 已按当前文字重新生成：443字段重建一致，5项算子检查通过；仍不包含完整重训练/单目标端到端验证。
