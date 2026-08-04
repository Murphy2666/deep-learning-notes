# Neural Network Training Process

最开始 $W$ 都是随机的，形状永远是 $(m, x)$。  
$m$ 是特征数（也是 `input_shape=[m]`，这里不写前向传播里 $X$ 的真实 shape 即 `[batch_size, m]`），$x$ 是神经元数。  
$b$ 默认全是 0。

$$\text{1个 batch} \longrightarrow \text{1次前向传播} \longrightarrow \text{1次反向传播} \longrightarrow \text{1次更新参数}$$

---

### 1、前向传播（Forward Pass 本质是空间变换）
**(wx+b空间拉伸/旋转/平移-->激活函数σ(·)空间弯曲/折叠)**

输入一个 batch 的数量，假设 3 个神经元，它们各自不同的初始 $W$ 算 3 套预测值。  
$\sigma$ 是 activation 激活函数。

**预测值：**
$$\hat{y} = \sigma(X W + b)$$

$$Y_{[batchSize, x]} = \sigma\left( X_{[batchSize, m]} \cdot W_{[m, x]} + b_{[1, x]} \right)$$

* $\sigma$ 为激活函数（如 ReLU, Sigmoid, Tanh 等）
* $X$ 为输入数据矩阵（Shape = `[batchSize, m]`）
* $W$ 为权重矩阵（Shape = `[m, x]`）
* $b$ 为偏置向量（Shape = `[1, x]`），通过广播机制相加到 $X \cdot W$ 上

wx+b 不进 $\sigma$ 激活函数，就只是简单的对input data空间的线性变换，直线还是直线，平面还是平面\

比如二维的input space,($x_1$, $x_2$):\
$$z = w_1 x_1 + w_2 x_2 + b$$ 只能做linear transformation线性/仿射变换，fold,bend 折叠/弯曲需要靠激活函数

### 几何直观：ReLU 与 Sigmoid 如何改变二维空间

* **ReLU（沿 $z=0$ 折叠 / Folding）**
  * **几何直观：** 以直线 $y = x$ 为例，原本是穿过原点 $(0,0)$ 的平直直线；经过 ReLU 作用后，在 $x \le 0$ 区域被**拍平折叠**为 $y = 0$，而在 $x > 0$ 区域保持 $y = x$。其实沿y=0折叠
  * **本质：** 沿 $z = 0$ 将空间硬折叠。

* **Sigmoid（非线性弯曲与挤压 / Bending/Warping）**
  * **几何直观：** 把原本平直的 $x$ 轴，**弯曲拉伸**成一条 S 型的 Sigmoid 曲线。
  * **本质：** 将整个空间的远端向 $(0, 1)$ 区间内**弯曲并挤压**。

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
