# Protenix Fresh192 执行协议

2026-09-25：作者明确批准新目标验证。既有方案：engramholder/reports/protenix_fresh_e2_replay_plan_20260925.md。该选择来自历史结果，非随机底座选择。

固定 Mini default / Train384 / Full / 1536，ESMC-600M final layer36、1152维；三种子20260923/24/25，三个旋转20261001/02/03，Factor和G+各Native3+Rotated9，加Query，共25系统。0新训练。

192完整新目标，128–191/192–255/256–319/320–384各48，截止2021-09-30，monomer_clean、单模型X-ray、≤2.5Å、20AA、参考CA≥90%。一PDB一目标；继承Fresh96最终BLAST v2原参数与单HSP判据，已知query coverage分母min长度，其他teacher分母candidate。固定rank SHA256('protenix-fourcell-fresh192-v1|'+target_id)，每层最多2048候选；3小时CPU筛选上限，不足即停、不放松。候选元数据不等于已评分目标。累积暴露刷新覆盖本机、DiamondHill、hpc3和Precision，显式补入Fresh96及E1/E2/E3；无完整历史/家族/基础模型隔离保证。

旧Confirm96清单首条+最长链(并列ID)共2条×25系统=50工程预测，不看质量选择。原坐标relative L2≤1e-3，full length、finite、4次注入、冻结SHA，零残差和完整残差后旋转。旧baseline仅用于工程，不混入新科学面板。

原c4/s5/sample1/seed101，torch kernels、FP32、关闭TF32/fusion，D沿checkpoint。ESMC保留历史H100 FP32计算→BF16缓存→加载FP32路径；新特征不改层或模型。正式folding全为DiamondHill MI250。只用序列构建推理特征；独立reference清单不得进入推理进程。4800正式预测全部到终态且身份核验后才评分。

唯一主比较Psi=(FN−FR)-(GN−GR)，先目标内3种子/3R聚合；Cα pair-lDDT原<15Å/0.5,1,2,4阈值与固定参考pair；20k次四层内同步目标bootstrap，seed20260925，等权层配额，百分位95%CI。正向支持下界>0。其他指标/组内与Query比较/seed,R,3×3/长度层为次要，不替换主终点。区间条件于固定24模型，不计训练/研究选择不确定性。新旧面板不池化。

个体瞬态失败最多2次同配置重试，间隔120秒；保存全部attempt，最终失败计零且保留分母。系统性源码/权重/特征契约错误阻止评分。不得换目标/裁剪/单臂改变配方。原子收据+进程锁恢复，不因心跳慢重复进程。全局执行锁在工程、面板、特征通过后封存；所有源码更正需保存旧版本与理由。
