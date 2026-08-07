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

最后得到的新空间，维度=神经元个数，就是新的representation\
推导详见linear_seperable.md

#### 为什么minst分类需要10个输出神经元而不是4个，这样 $2^4=16$ 可以cover 10个数字，比如0000=0, 0001=1, 1001=9
直觉上来讲，输出10个神经元，过一层softmax可以找到range在0-1之间分数最大的神经元；\
假设设计输出层4个神经元，用二进制表示预测结果\
我们知道图片是1或7，但不确定是哪个，1的二进制是(0,0,0,1)，7的二进制是(0,1,1,1);/
假设机器是完美的，因为这个不确定，输出层给出的预测分数会是(0,0.5,0.5,1)
这在二进制下可以是1,3,5,7；如果是10个输出神经元，1和7的分数会各为0.5，就不会有3和5的可能性/
所以输出层用10个神经元最好

---

### 2、反向传播（Back Propagation）本质是导数的chain rule

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

假设一个深度学习模型有$L$层，输出层有只有一个神经元，上一层$L-1$层也只有一个神经元，用mse做loss；
$L-1$到$L$层的权重和bias是：
$(w^{(L)},b^{(L)})$
C0是cost function也就是loss
链式传导：
$$\frac{\partial C_0}{\partial w^{(L)}} = \frac{\partial z^{(L)}}{\partial w^{(L)}} \frac{\partial a^{(L)}}{\partial z^{(L)}} \frac{\partial C_0}{\partial a^{(L)}}$$

各部分等于：

$$z^{(L)} = w^{(L)} a^{(L-1)} + b^{(L)} \quad \longrightarrow \quad \frac{\partial z^{(L)}}{\partial w^{(L)}} = a^{(L-1)}$$

<p align="center">
$$a^{(L)} = \sigma \left( z^{(L)} \right) \quad \longrightarrow \quad \frac{\partial a^{(L)}}{\partial z^{(L)}} = \sigma' \left( z^{(L)} \right)$$ 
（这里 $\sigma$ 激活函数可以是sigmoid可以是$tanh$可以是softmax等等）
</p>

$$C_0 = \left( a^{(L)} - y \right)^2 \quad \longrightarrow \quad \frac{\partial C_0}{\partial a^{(L)}} = 2 \left( a^{(L)} - y \right)$$

这是一个training example的，对于batch_size个example的，它们前向传导后再反向传导，调整一次参数。\
假设batch_size=n:\
对于n个样本的损失函数，是取平均：
$$C = \frac{1}{n} \sum_{k=0}^{n-1} C_k$$

对 $w^(L)$ 的求导：
$$\frac{\partial C}{\partial w^{(L)}} = \frac{1}{n} \sum_{k=0}^{n-1} \frac{\partial C_k}{\partial w^{(L)}}$$\

这是一个batch_size训完后，模型对于 $w^(L)$ 整体调整的方向

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
