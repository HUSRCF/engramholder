# OpenFold A组：ESMC最终层扩展及论文整合（2026-09-23）

本报告只纳入已完成的OpenFold Train384/1536 A组；不纳入仍在运行的B组、DiamondHill矩阵、Fresh96或SGDM结果。原线程的训练任务与配置均未修改。

## 直接核验

33个新训练实例：ESM2旋转G+九组＋ESMC完整四格24组；新增4,752份预测，失败0。复用历史15个ESM2适配器及基线，合并7,200条评分。脚本 `scripts/verify_openfold_esmc_A.py` 从评分数组重建78项对比，核验逐目标值、均值、区间、种子/旋转边际及Holm校正。没有重新评分CIF。

## 完整主指标表

| 面板 | 适配器特征 | Query | Native | Rotated Factor | G+ | Rotated G+ |
|---|---|---:|---:|---:|---:|---:|
| confirm96 | E_last | 0.30191 | 0.50684 | 0.50381 | 0.51333 | 0.50998 |
| confirm96 | C_last | 0.30191 | 0.67031 | 0.65780 | 0.67619 | 0.67117 |
| length48 | E_last | 0.26079 | 0.40165 | 0.39040 | 0.40868 | 0.40445 |
| length48 | C_last | 0.26079 | 0.48880 | 0.48158 | 0.50527 | 0.50128 |

## 不能省略的区间

| 面板 | 特征 | 对比 | 均值 | 95%区间 |
|---|---|---|---:|---|
| confirm96 | E_last | factor_rotation | +0.00303 | [-0.00656, +0.01218] |
| confirm96 | E_last | gplus_rotation | +0.00335 | [-0.00480, +0.01112] |
| confirm96 | E_last | interaction | -0.00032 | [-0.01194, +0.01157] |
| confirm96 | E_last | native_minus_gplus | -0.00648 | [-0.01772, +0.00548] |
| confirm96 | C_last | factor_rotation | +0.01251 | [-0.00019, +0.02578] |
| confirm96 | C_last | gplus_rotation | +0.00502 | [-0.00616, +0.01555] |
| confirm96 | C_last | interaction | +0.00749 | [-0.00849, +0.02388] |
| confirm96 | C_last | native_minus_gplus | -0.00588 | [-0.01997, +0.00905] |
| length48 | E_last | factor_rotation | +0.01126 | [+0.00531, +0.01734] |
| length48 | E_last | gplus_rotation | +0.00424 | [-0.00069, +0.00929] |
| length48 | E_last | interaction | +0.00702 | [-0.00021, +0.01451] |
| length48 | E_last | native_minus_gplus | -0.00703 | [-0.01357, -0.00052] |
| length48 | C_last | factor_rotation | +0.00722 | [-0.00318, +0.01766] |
| length48 | C_last | gplus_rotation | +0.00399 | [-0.00353, +0.01226] |
| length48 | C_last | interaction | +0.00323 | [-0.00965, +0.01618] |
| length48 | C_last | native_minus_gplus | -0.01647 | [-0.02919, -0.00451] |

## 论文应如何解释

- ESMC提高Native与G+的绝对质量，但两个面板的ESMC Native−Rotated区间均跨零；不能称为ESMC下方向效应复现成功，也不能称为等效或消失。
- ESMC下头类型×旋转交互未建立；相对ESM2的方向差变化K、交互变化J也未建立。Confirm96关键次要Holm调整p：Psi_C=0.73786、K=0.62547、J=0.73786。
- G+在ESMC下两个面板均值更高，Length48 Native−G+的未校正次要区间全负。不能说扩大PLM便让原生头成为更优方法。
- 原正交互来自OpenFold Train96；A组为Train384。不能把二者差异归因于PLM，也不能覆盖原Train96结论。
- ESMC最终层替换同时改变PLM家族、输入维度、缓存精度及投影参数数目，不是纯参数规模效应；A组不回答中间层/MLC选择。
- ESM2旧指标的点估计保持不变；A组统一bootstrap种子20260926，重算区间与历史不同随机流会有微小差别。旧表不覆盖。

## 稿件修改

- 主文新增ESMC四格绝对分数表及有边界的结果段；修改默认ESM2输入描述。
- 摘要、引言、讨论、结论同步保留这一不支持条件，不增加成功复现主张。
- 详细三指标四格对比、PLM直接交互、校正规则与输入参数变化放附录。
- 范数交换详细四格移入原附录小节，主文保留排除整体尺度解释的短段；不再让它承担纯方向机制解释。
- 新增v4来源锁及迁移记录；保留v1–v3，所有旧输入哈希和旧数值不变。
- 不提交或推送。

## 验证与交付

官方评分脚本与协议原文哈希均与执行锁一致。既有246数值字段及来源保持不变；新增197字段。数值键3测试、原算子5测试通过；v8匿名分析包443字段重建一致。Tectonic编译成功，无未定义引用和overfull警告；不替代作者在Overleaf的视觉/页数检查。详见 `notes/writing_branch_20260922/draft_validation.esmc_A.json`。
