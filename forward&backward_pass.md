# Neural Network Training Process

最开始 $W$ 都是随机的，形状永远是 $(m, x)$。  
$m$ 是特征数（也是 `input_shape=[m]`，这里不写前向传播里 $X$ 的真实 shape 即 `[batch_size, m]`），$x$ 是神经元数。  
$b$ 默认全是 0。

$$\text{1个 batch} \longrightarrow \text{1次前向传播} \longrightarrow \text{1次反向传播} \longrightarrow \text{1次更新参数}$$

---

### 1、前向传播（Forward Pass 本质是空间变换）
**(wx+b空间拉伸/旋转/平移-->激活函数σ(·)空间弯曲/折叠)**

输入一个 batch 的数量，假设 **3 个神经元**，它们各自不同的初始 $W$ 算 3 套预测值。  
$\sigma$ 是 activation 激活函数。

**预测值：**
$$\hat{y} = \sigma(X W + b)$$

$$Y_{[batchSize, x]} = \sigma\left( X_{[batchSize, m]} \cdot W_{[m, x]} + b_{[1, x]} \right)$$

* $\sigma$ 为激活函数（如 ReLU, Sigmoid, Tanh 等）
* $X$ 为输入数据矩阵（Shape = `[batchSize, m]`）
* $W$ 为权重矩阵（Shape = `[m, x]`）
* $b$ 为偏置向量（Shape = `[1, x]`），通过广播机制相加到 $X \cdot W$ 上

wx+b 不进 $\sigma$ 激活函数，就只是简单的对input data空间的线性变换，直线还是直线，平面还是平面

比如二维的input space,($x_1$, $x_2$):\
$$z = w_1 x_1 + w_2 x_2 + b$$ 只能做linear transformation线性/仿射变换，fold,bend 折叠/弯曲需要靠激活函数

### 几何直观：ReLU 与 Sigmoid 如何改变二维空间

* **ReLU（沿 $z=0$ 折叠 / Folding）**
  * **几何直观：** 以直线 $y = x$ 为例，原本是穿过原点 $(0,0)$ 的平直直线；经过 ReLU 作用后，在 $x \le 0$ 区域被**拍平折叠**为 $y = 0$，而在 $x > 0$ 区域保持 $y = x$。其实沿y=0折叠
  * **本质：** 沿 $z = 0$ 将空间硬折叠。
  * **多分类任务：一般hidden layers用ReLU，最后一层output layer用softmax**
  * **二分类任务：一般hidden layers用ReLU，最后一层output layer用sigmoid** \
    其实sigmoid是softmax的一种
    
$$P(\text{Class 1}) = \frac{e^{z_1}}{e^{z_1} + e^{z_2}} = \frac{1}{1 + e^{-(z_1 - z_2)}} = \frac{1}{1 + e^{-\Delta z}} = \sigma(\Delta z)$$

* **Sigmoid（非线性弯曲与挤压 / Bending/Warping）**
  * **几何直观：** 把原本平直的 $x$ 轴，**弯曲拉伸**成一条 S 型的 Sigmoid 曲线。
  * **本质：** 将整个空间的远端向 $(0, 1)$ 区间内**弯曲并挤压**。

---

### 2、反向传播（Back Propagation）

比较预测结果和真实值 ground truth，计算 loss（比如 mae）；对 loss 求 $w$ 和 $b$ 的偏导：

* 对 $w_1, b_1$ 求偏导 $\implies$ 得到梯度 $g_{w1}, g_{b1}$
* 对 $w_2, b_2$ 求偏导 $\implies$ 得到梯度 $g_{w2}, g_{b2}$
* 对 $w_3, b_3$ 求偏导 $\implies$ 得到梯度 $g_{w3}, g_{b3}$

$$
\begin{aligned}
\nabla_{w_i} L &= \frac{\partial L(y, \hat{y})}{\partial w_i} = g_{wi} \quad (i \in \{1, 2, 3\}) \\
\nabla_{b_i} L &= \frac{\partial L(y, \hat{y})}{\partial b_i} = g_{bi} \quad (i \in \{1, 2, 3\})
\end{aligned}
$$

---

### 3、更新参数（一次性更新所有 $w, b$）

一个 batch 结束后，SGD 用统一的更新公式，并行更新这 6 个参数：

$$w_{\text{new}} = w_{\text{old}} - \text{learningRate} \times g_w$$

$$b_{\text{new}} = b_{\text{old}} - \text{learningRate} \times g_b$$

为什么是 $-\text{learningRate} \times g_w$，学习率又是正数，为什么前面负号与 $g_{w_i}$ 相反的方向？
因为 $g_{w_i}$ 是 Loss 随 $w_i$ 增长的单位变化量（切线斜率）：

* 如果 $g_{w_i} > 0$，Loss 随 $w_i$ 增大而增大，要找 Loss 最低点，所以取 $-g_{w_i}$（向左走，减小 $w_i$）。
* 如果 $g_{w_i} < 0$，Loss 随 $w_i$ 增大而减小，要找 Loss 最低点，所以取 $-g_{w_i}$（即正数，向右走，增大 $w_i$）

上述“**计算梯度方向 $\to$ 沿该方向走一小步**”的完整过程重复多次:
<p align="center">
                              前向传播 $\implies$  计算 loss $\implies$ 反向传播求梯度 $\implies$  更新参数
</p>
直到 Loss 收敛到局部最小值点，就是**梯度下降算法 (Gradient Descent Algorithm)**：

 *"The algorithm for minimizing this function is to compute this gradient direction, take a step downhill, and repeat that over and over."*

  **注意**：找的Loss最低点是local minimum局部最小，几个坑可能有比现在更低的（global minimun）；\
  不考虑学习率的情况下，越接近谷底偏导绝对值越小

---

**总共更新次数：**
$$\left\lceil \frac{\text{sampleSize}}{\text{batchSize}} \right\rceil \times \text{epoch}$$
