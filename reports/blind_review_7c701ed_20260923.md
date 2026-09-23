**继续盲审：论证、对照与数据定义**

审阅对象：X570 `/home/husrcf/Code/engramholder`，提交 `7c701ed`，读取时工作树干净。该提交继承 `7b3886d` 的正式文字修订。原始审阅快照保存在另一台会话主机的 `/data/zhuoxu/hipfire/Engramfold/review/x570-engramholder-7c701ed/`；本文件为论文仓库中的可查阅副本。本轮未修改论文正文或实验，未启动训练、预测或读取新的评测面板。原始审阅笔记保存在当前会话工作目录，不写系统记忆；用户指出在论文仓库找不到报告后，另提供X570仓库 `reports/` 下的副本。

上一轮主要展示问题已处理：三套 OpenFold Ψ 并列、摘要与结论明确 Train96/ESM2、同 ESM2 的 Train384 边界进入正文、删除“separates feature quality”、范数交换不再承担纯方向机制解释。下文不把这些问题继续列为未解决缺陷。

**总体判断**

当前稿件可作为一项有边界的受控经验研究。未发现足以推翻四格比较的数学硬伤，也没有发现本轮可据以认定的数据泄漏或指标错误。模拟盲审仍倾向 Weak Reject，置信度中等；主要影响接收的因素是中心交互缺乏独立新目标确认，以及还未形成足够具体、可以区分解释的知识增量。负结果应被保留，不因其未显著而扣分。

“尚未唯一识别机制”本身不是否定经验论文的充分理由。也不应强迫作者同时完成全部优化器、全部底座和所有消融才承认现有观察成立。优先补强最核心证据，比不断增加零散实验更有价值。

**1．OpenFold 的追近模式不能推广成统一的“小数据归纳偏置”解释**

位置：[结果第223行](../sections/04_results.tex#L223)、[附录第176行](../appendices/implementation_and_evidence.tex#L176)。

这里比较的是 Factor 内部的 Δ_F=Native−Rotated，不能与四格交互 Ψ 混写。各行均为 Full、1536更新：

| 底座/面板 | Train96的Δ_F | Train384的Δ_F | 384−96，95%条件CI |
|---|---:|---:|---|
| OpenFold / Confirm96-B | +0.01981 | +0.00303 | −0.01678 [−0.03214, −0.00261] |
| Protenix / Confirm96-B | +0.02505 | +0.03951 | +0.01446 [+0.00350, +0.02586] |
| OpenFold / Length48 | +0.01038 | +0.01126 | +0.00087 [−0.00728, +0.00863] |

这些区间来自现有后续分析，未联合做多重比较校正。本轮未增加统计检验。

因此，“原生方向在小数据时有利，数据更多后旋转头追上”只能作为 OpenFold/Confirm96-B 的局部解释假说。Protenix 在同样固定更新预算下表现相反；OpenFold 长链也没有建立追近。不能为追近和差距扩大分别给出都归功于原生偏置的事后故事。

Protenix 的绝对均值为 Native 0.61653→0.62572、Rotated 0.59148→0.58621；旋转组下降这个单项的区间包含零，不能将每组均值变化分别宣称为已建立效应。可支持的是已有直接分析中的差距变化。

这一事实不否定一般的归纳偏置，也不推翻原生方向在部分设置下有益。它要求将问题表述为：不同接口、数据覆盖和有限训练预算怎样共同决定原生方向的收益。跨底座的接口、损失和原生表征同时改变，不能进一步归因于某个架构组件。

建议讨论句：

> The opposite fixed-budget changes under the tested Protenix and OpenFold recipes do not support a shared monotonic account in which additional training examples consistently remove the native-orientation advantage.

此前对话中关于“数据增加使旋转头追近”的解释，应明确限定到 OpenFold 的短链结果，不作为整篇论文已建立的机制。

**2．需要区分三种假说，防止“有条件的归纳偏置”变成无法证伪的概括**

位置：[引言第55行](../sections/01_introduction.tex#L55)、[讨论第34行](../sections/05_discussion.tex#L34)。

| 解释 | 真正要检验什么 | 当前证据缺少什么 |
|---|---|---|
| 样本效率 | 达到预定质量需要多少不同训练蛋白 | 不同集合在明确训练充分程度下的学习曲线 |
| 有限优化预算 | 固定数据下，各头随更新步数如何变化 | 1536步之后的配对结果及Dev曲线 |
| 训练组成 | 新增蛋白的分布是否改变某些头的相对收益 | 相同规模的独立子集或已定义的组成分析 |

当前 Train24⊂Train96⊂Train384；嵌套控制了旧样本被替换的问题，但增加样本同时改变了组成和每条样本的重复次数。固定1536步时，每条目标分别出现16次和4次。旧训练loss末段仍下降，不能假定训练均已收敛。详见[已有日志核查](openfold_training_budget_review_20260923.md)。

现稿第230–232行已经正确避免样本效率定律。建议将这一限制与上面的相反变化放在一起，解释为何暂不选择单一机制。若下一阶段专门回答训练预算问题，四种头应统一延长；不能只继续训练 Native，也不能按测试Ψ是否转正选停点。

这是一项解释层面的重要未决问题；不要求为了现有条件性经验结论完成全部三条研究路线。

**3．数学论证基本成立，但“同谱”和“更灵活”两个词需要更精确**

位置：[引言第20行](../sections/01_introduction.tex#L20)、[引言第50行](../sections/01_introduction.tex#L50)、[构造第59行](../sections/02_construction.tex#L59)、[构造第65行](../sections/02_construction.tex#L65)。

“matching parameter counts, operator spectra and initial functions”并列，容易让读者误以为 Factor/G+ 也匹配了谱。实际控制范围应拆开：

| 比较 | 已控制 | 未因此控制 |
|---|---|---|
| 同一头 Native/Rotated | 共同参数、输入、anchor处的局部writer谱；零残差初始函数 | 训练后的残差范数、完整recycle Jacobian、AdamW轨迹 |
| Factor/G+ | 相同初始函数、相同任务输入来源、近似参数量及共同训练配方 | 局部Jacobian、初始梯度大小、有效学习尺度、架构组织 |

可直接替换为：

> Within each head, orthogonal controls preserve local writer spectra at matched states. Across heads, we match the initial function and approximately match parameter counts; these controls do not equate their optimization geometry.

Factor 的限制应解释为：逐残基增量在所有 residue pairs 间共享，通过固定预训练decoder生成联合更新。不能只从因子宽度32推断整个pair×channel矩阵秩≤32；现稿没有做出这一错误主张，应保持。

G+ 的自由末层证明了输出旋转闭包，但没有证明它的函数类包含 Factor。其逐对MLP、GELU、节点瓶颈、相对位置及anchor进入位置均不同。因此“更灵活”最好改为明确性质：具有自由输出层、函数类可吸收输出旋转。不能将其当作已证明的表达能力或性能上界。

可补一句：

> G+ is a comparator with a provable output-rotation closure property, not a demonstrated superset or performance upper bound of Factor.

这些澄清不会削弱已测交互的有效性，而是明确它反映两个架构整体的差异，不能唯一归因于是否固定最后decoder。GELU在自由末层之前，不破坏吸收证明；现场anchor依赖recycle也不破坏理想实数下逐次保持相同输出的函数对应。数值路径仍需要与精确算术区分。

**4．数据定义是这轮找到的一个可直接修复的缺口**

位置：[设计第51行](../sections/03_design.tex#L51)、[附录第90行](../appendices/implementation_and_evidence.tex#L90)。

Length48 的来源、候选条件、分层和排除规则写得详细，短链 Confirm96-A/B 与训练集则主要介绍研究时序。盲审读者仍不知道这96条短链代表怎样的抽样总体。

本轮直接从 [split96](../evidence/protenix_split96.json) 与 [split384](../evidence/protenix_split384.json) 核对：Train24⊂Train96⊂Train384；Train384 与 Dev8、A/B两个确认面板无目标ID交集。ID不重叠并不等于序列或家族独立。

**依据用户纠正明确区分继承与新增：Train24和Dev8是继承的旧集合，不得将新增短链候选的“2021-09-30前发布”条件套到它们身上。Train96新增72条、Train384新增288条，以及Confirm96-A/B，应分别按各自协议描述筛选条件。** 因而不能把Train96或Train384整组统一标成满足新增候选的发布日期条件。继承集合的日期范围及其他资格条件只按其原始记录填写；未核对的内容保留未核对。

历史 B面板selector（实验仓库 `/home/husrcf/Code/onestepfold/engramfold/src/engramfold/data/native_direction_panel.py:15`） 的源文件哈希与split来源记录一致；代码定义四个长度层128–191、192–255、256–319、320–384，Confirm96-B每层24条，Train96继承Train24并在每层新增18条。排序使用不读取模型分数的固定哈希；排除重复PDB、已有近同源和已选目标近同源。上述为选择规则与来源核验，本轮没有重新运行原始候选选择。

Train384选择器（实验仓库 `/home/husrcf/Code/onestepfold/engramfold/scripts/select_train384.py:12`） 定义在已有Train96基础上每层新增72条，共288条，排除已知训练、开发与A/B目标。其具体候选筛选说明仍应引用各自执行锁；不应套用Length48的BLAST参数或把不同历史版本的筛选流程合并。

建议补一张六列数据表：**集合/来源分组、样本数、候选来源与结构条件、长度配额/实际范围、嵌套及排除关系、用途与首次评分身份**。分开列出继承的Train24、Dev8，Train96新增72条，Train384新增288条，以及A、B各自的面板；各用自己的记录，未知项保持未知。这里没有发现按测试质量挑选目标的证据，不能据说明不足指控泄漏。

该表可以直接改善科学可读性：读者应知道报告的均值对应一个经过结构质量筛选、按长度配额组成的面板，而不是自然蛋白总体的无条件平均。

**5．bootstrap和结构指标各有明确范围；有现成证据可以支持论文**

位置：[设计第67行](../sections/03_design.tex#L67)、[附录第131行](../appendices/implementation_and_evidence.tex#L131)。

当前以整条目标记录重采样，保留目标内seed/rotation配对是正确的。“条件于模型”仍不自动解决跨目标家族相关性；已有近同源筛查有帮助，但完整家族隔离未建立。应把区间解释为相应目标抽样假设下的条件不确定性，不能称为已认证的家族泛化区间。若已有簇映射可做有边界的敏感性分析；未知注释不能一律视为独立新家族。也不能仅凭未知预训练暴露就否定共享底座的匹配对照。

pair-lDDT 评估参考距离小于15Å的Cα对，没有最小序列间隔限制；目标内部按pair平均会给参考邻居较多的残基更大权重。它不独自证明全局拓扑、所有新接触或物理有效性。现稿并未提出这类过强主张，因此属于澄清事项。

已有 supplementary 结果反而值得更醒目：OpenFold Train96/Confirm96-B 的Ψ，在residue-average lDDT上为 **+0.01737 [0.00571,0.02996]**，在固定对应TM-score上为 **+0.01847 [0.00250,0.03528]**。将它们紧邻主结果列出，比再强调工程预测总次数更有说服力。这是同一批结构上的指标稳健性检查，不是三次独立复制。

**6．论文的贡献应按受控经验研究评价**

对论文最有利的几项事实应明确保留：

- 所有旋转条件都从零残差接受任务训练，避免将训练后破坏模型当作方向控制。
- G+函数类闭包给出有数学含义的对照；直接估计Ψ，而不借两个显著性结论代替交互。
- Protenix Train384 Native对匹配G+的优势为Confirm96-B +0.07630、Length48 +0.08138，且G+本身在均值上明显高于query-only；这说明原生构造在该配方下有实际作用。
- 最初锁定的Protenix方向与Length48研究提供了不同于后续观察面板的证据身份。
- 负结果、G+优势及局部机制未建立的事实被诚实保留。

Protenix Length48 Native 0.52919与官方Mini-ESM 0.92139的差距限制部署主张，但不单独否定内部适配行为研究。反过来，若机制未被识别，也不能临时将论文改成高性能方法来弥补贡献；当前缺少相应性能—成本证据。

最值得加强的主线是：**旋转敏感性需要结合头参数化判断；这一交互在明确配方下可观测，但其随数据覆盖和训练预算变化的方式并不跨底座一致。** 前半已有直接证据，后半提供需要解释的实证边界。不要把所有结果包装成统一样本效率机制。

**修订建议与优先级**

| 优先级 | 工作 | 当前是否需要新GPU实验 |
|---|---|---|
| P0 | 用跨底座相反变化约束归纳偏置解释，明确目前是局部假说 | 否 |
| P0 | 明确同谱控制的比较范围，以及G+闭包不等于包含Factor | 否 |
| P1 | 补短链面板与嵌套训练集的数据定义表 | 否，优先现有锁和选择记录 |
| P1 | 主结果附近展示已有residue-lDDT/TM-score交互；明确它们的证据身份 | 否 |
| P1 | 按已锁计划取得中心交互的独立目标证据 | 依原计划，本轮不运行 |
| P1 | 若选择解释数据效应，补匹配的训练预算/Dev曲线，而非先扩更多PLM型号 | 需要有界的新实验；另行按既有研究计划执行 |

本轮没有要求重新开展全部现有实验。已完成的条件性结论应保留，优先修复可由现有资料解决的说明缺口，再用最直接的独立证据支持论文中心。
