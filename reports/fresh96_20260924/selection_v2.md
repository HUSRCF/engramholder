# Fresh96 selection v2：在新预测前修正已知的全局比对误排

2026-09-23。核心授权不变：固定OpenFold Train96四格24模型＋Query，96个未评分目标，2400预测，零训练。此文件覆盖初始protocol.md的“全局比对”选择段；初始协议、两次选择失败与模型锁全部保留，不能称v1通过。

## 修订原因

初始计划遗漏了项目已完成的Length48筛查校准：全局比对以非gap列计算identity，对长短序列可能通过大量gap碎片化产生虚假的高identity。此次加入已观察长链和开发链后，首层2048候选0接受，按哈希顺序前8候选的8个组成保持shuffle负对照也全部被旧规则排除。第一次还误纳入256条synthetic-null为真实query，已纠正并保留父PDB排除；第二次在移除它们后仍0/2048。

这些是筛查工程证据，不是折叠分数；尚无新面板或新目标预测。改用项目已验证、用于Length48 v2的固定BLAST规则，不搜索新的阈值，不按任何adapter表现改数据。此前校准0/256 null、288/288指定嵌入阳性；四对CATH关系漏检继续披露，不称家族隔离。

## 固定算法

NCBI BLASTP2.17.0，binary hash必须匹配此前calibration lock 023fc7b802e719b7b0204a07cc2c46e3295af2e14654e445f0788242bf85e951。task blastp、BLOSUM62、gap-open11/extend1、word3、SEG yes、soft_masking true、comp_based_stats2；检索E≤1e-3、max_target_seqs为全部库大小、无max_hsps截断。

任一单HSP同时满足：E≤1e-5；identity=nident/alignment_columns≥.30（包含内部gap）；paired≥50；paired/coverage_denominator≥.70。已知真实query与panel用min(query/reference length)，其他历史teacher行用candidate length。不合并HSP。保存qseq/sseq、端点、完整长度并重算一致性。

排除库包含原15605参考序列并补齐累计真实query；已实际使用的query/PDB与未使用候选池区分。合成shuffle不作为真实蛋白暴露，原开发父PDB仍全部排除。新库规模不同于历史校准，记录此限制，不推断生物学误差率。使用完整原库＋已接受panel序列作第二个库核对panel近重复；仅对panel entries应用该步排除，不以增大库后的E值修改原暴露判断。

## 未改变的选择／分析设置

96目标，四层128–191/192–255/256–319/320–384各24；所有原资格、完整输入、PDB/实体映射、≥90%参考CA、旧项目暴露排除保留。固定SHA256('openfold-interaction-fresh96-v1|target_id)次序，每层最多2048候选、CPU选择3小时上限。v1没有合格目标，不继承任何暂选目标。v2配额不足停止，不继续换阈值。

24checkpoint及Query锁不变，原生／旋转／G+/GR主终点Psi不变。工程回放和预测精度不受筛查修订影响，继续保留其初始preparation lock；正式execution lock同时引用初始工程锁和本v2选择锁，不能混称同一版本。所有正式尝试之后统一评分。新面板将报告为“按BLAST v2筛查、对可审计项目记录未评分的新目标验证”，不写成完全相同的历史Confirm96-B筛查，也不宣称家族／基础模型隔离。
