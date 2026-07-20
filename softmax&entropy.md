# Softmax

在预测分类 model prediction 任务里，总共 $K$ 种预测可能 ($K$ categories)，模型会给每个 category 一个 score，得分越高，对象是这个 category 的可能越大：

$$\mathbb{R}^K \to (0, 1)^K \quad \text{where } K > 1$$

本质是把 score (logit, raw score) 映射到 0 到 1 的概率上，这里是伪概率。

自然地，直接取得分最高的分类为 1，其余为 0（这就是One-hot Encoding方法）
例如 $(0, 1, 0, \dots, 0)$。但是这样会显得太生硬，比如预测模型有 3 个可能：(猫, 狗, 兔)，给出 score 是 $(10, 110, 130)$。如果只取最高分 130 (兔) 为 1，其余为 0，那么会丧失模型也觉得这个对象很像狗的信息。

因此做指数函数的指数：
$\sigma : \mathbb{R}^K \to (0, 1)^K$, where $K > 1$

把 $\mathbf{z} = (z_1, \dots, z_K) \in \mathbb{R}^K$ 映射到 $\sigma(\mathbf{z}) \in (0, 1)^K$ 上：

$$\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

This ensures:
1. If $z_i \ll z_j$, then $e^{z_i} \ll e^{z_j}$
2. Normalization 归一化：所有 category 的 softmax 之和是 1
   $$\sum_{i=1}^{K} \sigma(\mathbf{z})_i = 1$$

---

# Softmax with Temperature

在上述 softmax 的每个 $z_j$ 下都加一个 $/t$：

$$y_i(\mathbf{x}|t) = \frac{e^{\frac{z_i(\mathbf{x})}{t}}}{\sum_{j=1}^{K} e^{\frac{z_j(\mathbf{x})}{t}}}$$

- $t = 1$ 就是正常的 softmax
- $t \to 0^+$ 时，会退化为 Hardmax，直接取得分最高项为 1，其余为 0
- $t > 1$ 把一些高 softmax 的 score 给到那些不那么高的；越温和，越能考虑到类别之间的联系，不会完全抹除低分的类别
- $t \to +\infty$，$y_i(\mathbf{x}|+\infty) = \frac{1}{K}$，所有类别的概率都一致，即 uniform distribution