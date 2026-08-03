# Neural Network Training Process

最开始 $W$ 都是随机的，形状永远是 $(m, x)$，$m$ 是特征数（也是 `input_shape=[m]`，这里不写前向传播里 $X$ 的真实 shape 即 `[batch_size, m]`），$x$ 是神经元数。  
$b$ 默认全是 0。

$$\text{1个 batch} \longrightarrow \text{1次前向传播} \longrightarrow \text{1次反向传播} \longrightarrow \text{1次更新参数}$$

---

### 1、前向传播（Forward Pass）

输入一个 batch 的数量，假设 3 个神经元，它们各自不同的初始 $W$ 算 3 套预测值。  
$\sigma$ 是 activation 激活函数。

**预测值：**
$$\hat{y} = \sigma(X W + b)$$

$$Y_{\text{[batchSize, x]}} = X_{\text{[batchSize, m]}} \cdot W_{\text{[m, x]}} + b_{\text{[1, x]}} 再过一下激活函数 $$

* $\sigma$ 为激活函数（如 ReLU, Sigmoid, Tanh 等）
* $X$ 为输入数据矩阵（$\text{Shape} = [\text{batch\_size}, m]$）
* $W$ 为权重矩阵（$\text{Shape} = [m, x]$）
* $b$ 为偏置向量（$\text{Shape} = [1, x]$），通过广播机制相加到 $X \cdot W$ 上

---

### 2、反向传播（Back Propagation）

比较预测结果和真实值 ground truth，计算 loss（比如 mae）；对 loss 求 $w$ 和 $b$ 的偏导：

* 对 $w_1, b_1$ 求偏导 $\implies$ 得到梯度 $g_{w1}, g_{b1}$
* 对 $w_2, b_2$ 求偏导 $\implies$ 得到梯度 $g_{w2}, g_{b2}$
* 对 $w_3, b_3$ 求偏导 $\implies$ 得到梯度 $g_{w3}, g_{b3}$

---

### 3、更新参数（一次性更新所有 $w, b$）

一个 batch 结束后，SGD 用统一的更新公式，并行更新这 6 个参数：

$$w_{\text{new}} = w_{\text{old}} - \text{learningRate} \times g_w$$

$$b_{\text{new}} = b_{\text{old}} - \text{learningRate} \times g_b$$

---

**总共更新次数：**
$$\left\lceil \frac{\text{sampleSize}}{\text{batchSize}} \right\rceil \times \text{epoch}$$
