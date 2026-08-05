# 向前传导几何变换、特征表示（Representation）
https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/#fnref2

深度学习本质是传好几层，线性/非线性地改变空间，把input layer的扭曲流形（缠在一起的绳子），一次次映射为新的表示，直到最新表示下的特征变得线性可分（揉成平行直线决策边界划得开）\
- **理想同胚模式（完全不折叠/不压扁）**： $\mathbf{W}$ 非奇异+使用可逆的激活函数（如 $\tanh$ ）\
  等于在不剪断、不折叠压扁的前提下，对空间进行连续的拉伸、弯曲与解缠

- **实际工程模式（允许压扁与折叠）**: 比如ReLU 把负半轴的空间折叠并压扁到坐标轴上
  
<img width="288" height="280" alt="image" src="https://github.com/user-attachments/assets/7e874b64-a080-460e-ad79-bf6422b0ac65" />

一次次拉伸、扭曲空间，直到直线可以平分（linearly seperable, 二维里是直线，n维里叫决策边界，高维超平面hyperplane）

<img width="288" height="280" alt="image" src="https://github.com/user-attachments/assets/1d6f468c-eaa1-405b-808d-bd4058eeafe8" />

---
## 1、向前传导本质是空间拉伸、平移、弯曲
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

### 如果 $\mathbf{W}$ 是non-singular可逆的，且激活函数取可逆的单调函数，那这一层神经网络就是同胚Homeomorphism

<img width="352" height="55" alt="image" src="https://github.com/user-attachments/assets/637c01d5-23be-480d-a1a9-90605bac197d" />

同胚变换会保留拓扑性质，带手把的咖啡杯和甜甜圈同胚；不能撕扯、戳破\
  - 比如$$A,B⊂R2:$$\
  $$A={x|d(x,0)<1/3}$$\
  $$B={x|2/3<d(x,0)<1}$$ \
这里只能升维到至少3维($n>=3$)，直觉是也许可以让其中一个稍微下沉一点，拿个$(n-1)$维的超平面切一刀；\
如果还是二维到二维，采用可逆变换，那么每一层都是一个 $\mathbb{R}^2 \to \mathbb{R}^2$ 的同胚映射，不可能分开
<img width="186" height="186" alt="image" src="https://github.com/user-attachments/assets/f35cc555-99d1-46f3-b5ef-c8fe58ce8276" />

  
  

###  激活函数例子： $\sigma$ 几何直观：ReLU 与 Sigmoid 如何改变二维空间

* **ReLU（沿 $z=0$ 折叠 / Folding）**
  * **几何直观：** 以直线 $y = x$ 为例，原本是穿过原点 $(0,0)$ 的平直直线；经过 ReLU 作用后，在 $x \le 0$ 区域被**拍平折叠**为 $y = 0$，而在 $x > 0$ 区域保持 $y = x$。其实沿y=0折叠
  * **本质：** 沿 $z = 0$ 将空间硬折叠。
  * **多分类任务：一般hidden layers用ReLU，最后一层output layer用softmax**
  * **二分类任务：一般hidden layers用ReLU，最后一层output layer用sigmoid** \
    其实sigmoid是softmax的一种
    


* **Sigmoid（非线性弯曲与挤压 / Bending/Warping）**
  * **几何直观：** 把原本平直的 $x$ 轴，**弯曲拉伸**成一条 S 型的 Sigmoid 曲线。
  * **本质：** 将整个空间的远端向 $(0, 1)$ 区间内**弯曲并挤压**。


 
    
---

原来平面长这样
<img width="429" height="362" alt="image" src="https://github.com/user-attachments/assets/010d1ce2-6f37-48f7-ad4d-bf1d3fde1bea" />
