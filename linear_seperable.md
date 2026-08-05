
# 向前传导几何变换、特征表示（Representation）
https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/#fnref2
---
### 1、向前传导本质是空间拉伸、平移、弯曲
假设这一层（可以是input layer）空间是 \
$$\mathbf{X} = (x_1, x_2, \dots, x_n)^T \in \mathbb{R}^n$$\
假设下一层（可以是hidden layer）有k个神经元 \
则权重 $\mathbf{W}$ 为 $k \times n$ 维，bias $\mathbf{b}$ 为 $k \times 1$ 维

<img width="512" height="279.5" alt="image" src="https://github.com/user-attachments/assets/6d2e4107-3d43-4585-8c9e-95052a5dac8e" />

下一层激活与新表示 \
$$\mathbf{a} = \sigma(\mathbf{W}\mathbf{x} + \mathbf{b}) = (a_1, a_2, \dots, a_k)^T \in \mathbb{R}^k$$
- $\mathbf{W}\mathbf{x} + \mathbf{b}$（线性变换 Linear）
  权重矩阵 $\mathbf{W}$: 旋转、缩放等
  bias $\mathbf{b}$: 平移
  
  **只有 $\mathbf{W}$ 和 $\mathbf{b}$ 没有激活函数的向前传导，无论有几层，只能简单线性变换**
  
- $\sigma(\cdot)$（非线性变换 Non-linear）
  激活函数 $\sigma$: 扭曲、挤压、折叠。


####  Eg. $\sigma$ 几何直观：ReLU 与 Sigmoid 如何改变二维空间

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

原来平面长这样
<img width="429" height="362" alt="image" src="https://github.com/user-attachments/assets/010d1ce2-6f37-48f7-ad4d-bf1d3fde1bea" />
