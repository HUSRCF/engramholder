# 外积记号修订

依据用户建议，方法中双线性decoder定义及Full残差展开的四处向量张量积记号统一改为外积：`a_i b_j^T`、`delta a_i b_{q,j}^T`、`a_{q,i} delta b_j^T`、`delta a_i delta b_j^T`。

两因子视为列向量，`vec`按行优先展平，b索引变化最快；这与实际decoder的`[output, a-factor, b-factor]`权重reshape及einsum一致。没有修改算子、模型、训练或实验结论。

当前正文、附录和PDF均未残留张量积符号。编译无警告，公式所在第2页视觉检查通过；v11的204个来源输入及1830个分数字段保持不变。没有为纯记号修改添加测试或重跑实验。

[当前反馈PDF](../build/feedback_v44_outer_product/paper.pdf)。最终篇幅与匿名包仍留待收尾。
