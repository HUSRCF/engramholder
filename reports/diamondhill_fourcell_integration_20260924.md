# Protenix／AtlasFold 完整四格正式整合（来源锁 v7）

2026-09-24；用户授权将已经完成并审阅的 A66 与 Protenix Train96 结果纳入正式稿。
写作基线 `ff1ff64`；这次不启动训练、不修改实验锁、不重新预测或评分 CIF，也不查询实时 GPU 队列。

## 1. 本轮新增与叙述变化

- 主表统一为15行：Protenix 5行、OpenFold 6行、AtlasFold 4行，明确列出 Query、F、RF、G+、RG+及训练规模、特征来源和面板。显示的所有格子已完成；Protenix Train96 Length48仍未运行，交互表标为 `---`。
- 交互表统一8种配置，保留 Protenix 正向区间、OpenFold Train384和AtlasFold未建立的区间；OpenFold原始Train96正结果明确限定配方。
- Fresh96进入交互主图，与已观察Confirm96-B/Length48并排；摘要、引言、讨论、结论继续明确新目标主要交互未建立，不能用Factor的正向对比替代。
- 摘要及正文纳入Protenix Train96第二底座交互、Train384 ESMC的方向效应和后续交互。ESMC提高Protenix/OpenFold绝对质量，但其直接方向效应和交互变化未建立；未写成PLM规模因果效应。
- AtlasFold两种新增特征的完整四格已完成，不再用“缺旋转G+”解释不能判断。仍未建立对应Factor优势或有用适配；原生AtlasLM-3B保留。
- 原方向范围图移入附录并加入Protenix/AtlasFold ESMC四行，释放主文空间给完整主表；原历史确认身份保留。

## 2. 统计身份不可合并

| 研究 | 既定主要量 | 其余量与面板的身份 |
|---|---|---|
| 原OpenFold Train96/ESM2四格 | Confirm96-B Ψ | 已观察面板的后续研究；Length48次要 |
| Protenix Train96 G3+GR9 | Confirm96-B Ψ | 已观察面板后续研究，不是新目标确认 |
| A66的每个底座 | Confirm96-B ESMC Factor−Rotated | 新Ψ、Native−G+、其他PLM对比及Length48等保留后续身份 |
| OpenFold Fresh96 | 固定Train96模型的Ψ | 新目标主要检验未建立；不被上述结果覆盖 |

A66没有统一多重校正，也未继承OpenFold A组的三项Holm检验。完成完整四格不意味着所有对比都成为预注册主要结果。所有区间条件于已有拟合模型；种子／旋转不作为新增独立蛋白。

## 3. 数据与独立复算

原始新增工作量为78个训练实例、10,656次预测；本写作轮计算量仅为CPU分数复算和排版。
从执行仓库按字节复制14份JSON，另将既有Length48面板清单纳入来源锁。v6的46份输入全部保持原哈希，v7合计61份输入；旧锁未覆盖。

- A66：14,112条完整模型—目标记录，其中9,504条新预测，4,608条历史复用。
- Protenix Train96：2,400条完整记录，其中1,152条新预测。
- 新脚本 `scripts/verify_diamondhill_fourcells.py` 重新检查唯一身份、完整覆盖、目标成员、有限数值、来源绑定、目标内聚合、配对bootstrap、种子／旋转边际及存在的3×3单元。
- 216项A66及18项Train96的三指标对比重建通过，共1,596次数值数组／标量比较，最大误差约2.22e−16；没有重评分CIF。
- 20,000次bootstrap沿用各自锁；A66 Length48沿用三层分层抽样。旧锁定区间不覆盖，新表区间可能因历史抽样流／规则不同而略异，点估计与原目标分数不变。
- 新增603个生成数值字段，合计1,135；旧532个字段（含来源映射）保持完全相同。

来源：[v7锁](../notes/writing_branch_20260922/paper_sources.v7.lock.json)、[显式迁移记录](../notes/writing_branch_20260922/diamondhill_lock_migration.json)、[导入清单](../notes/writing_branch_20260922/diamondhill_source_import.json)。新表与所有数值由脚本生成，无手工修改数值输出。

## 4. 后端与解释边界

- Protenix ESMC完整四格为MI250；其历史ESM2 Length48 Factor/Query来自H100，G+来自MI250，跨头与跨PLM涉及对应后端限制。
- Atlas ESMC第三训练种子的八个模型为H100，其他两个ESMC种子和所有ESM2臂为MI250。后端与种子混合，不能称为独立后端复现；该种子跨PLM比较也改变后端。
- 附录保留Atlas梯度断言失败、恢复、饱和事件及诊断规则修订。最终预测均成功不等于从未发生工程失败。
- 不把两底座不同训练规模／接口／损失当作单一架构因果干预。不把大PLM、局部秩或大预测变化解释为已验证的效应预测器。
- 六组共享重建、18次几何／传播及其他研究仍独立保留；本轮没有把那些诊断数字或新E1结果并入论文结论。

## 5. 验证与交付

验证细节见[本轮验证记录](../notes/writing_branch_20260922/draft_validation.diamondhill_v7.json)。
数值字段检查、现有三组CPU测试及匿名包的算子检查通过。CPU LaTeX编译主文9页、全文29页，无未定义引用和overfull；官方模板、匿名模式未改动。
已检查交互主图、完整主表、交互表、结论及新增附录方法／表格的实际PDF页面；这不是全稿逐页人工验收。
最新本地PDF为 `build/iclr2027_conference.pdf`，匿名包更新到 `build/anonymous_artifact_v16.zip`。匿名包数值重建和检查通过；全矩阵推理／训练复现仍未建立。
