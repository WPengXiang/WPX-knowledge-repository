---
title: 相场理论的简单介绍
key: docs-phase-field-method-zh
---

## 介绍

在实际问题中常常需要使用数学的方法来处理物体的界面, 界面的数学表达方式有两种:

* 将界面看做没有厚度的表面, 即嵌入在三维欧式空间的二维表面
* 相场法, 引入相场变量(或相场函数), 用相场中薄的过渡层表示界面

相场法的优点:
* 界面方程变为定义在整个空间上的简单反应扩散方程, 不需要对界面进行特殊处理
* 表面张力自然存在
* 容易处理像挤压或合并界面之类的拓扑变化
* 可以很容易与各种物理方程相结合

## 基本的界面方程

**相场和能量**

假设在凝固过程中存在两种相, 例如液相和固相. 使用 $\phi=0$ 和 $\phi=1$
来表示这两种相, 分别记作 0 相和 1 相. 在相场法中,
两相之间的薄的内部过渡层使用变量 $\phi$ 来表示, 如下图.

<img src="figures/phase_field_description.png" alt="pfd" style="zoom:50%;" />

我们使用双稳态反应扩散方程的前解来描述相场 $\phi$.

首先定义双势阱 $f(\phi)$:

$$
f(\phi) = \frac{a^2}{2}g(\phi)+\Delta f \cdot h(\phi)
$$

其中, $g(\phi)$ 表示对称部分的势能, 例如:

$$
g(\phi) = \phi^2(1-\phi)^2
$$

$h(\phi)$ 表示倾斜部分的势能, 表示为:
* 类别 1
$$
h(\phi) = \phi^3(10-15\phi+6\phi^2)
$$

* 类别 2
$$
h(\phi) = \phi^2(3-2\phi)
$$

这时可以看出, $f(0)=0, f(1)=\Delta f$, 即 $f(\phi)$ 在 $\phi=0$ 和 $\phi=1$
处有局部最小值. 因此, 
* 当 $\Delta f <0$ 时 0 相为亚稳态(meta-stable)相, 1 相为稳定相. 
* 当 $\Delta f >0$ 时 1 相为亚稳态(meta-stable)相, 0 相为稳态相. 

如下图所示. (注意: 当采用第二类
$h(\phi)$ 时, 必须保证 $| \Delta f | <\frac{a^2}{6}$, 才能确保 $f(\phi)$ 在
$\phi=0$ 和 $\phi=1$ 处有局部最小值.)

<img src="figures/f.png" alt="f" style="zoom:50%;" />

如果图中 $\phi=0$ 和 $\phi=1$ 分别对应液相和固相, 则温度 $T$ 与 平衡温度$T_e$
之间的关系分别为 图 a) $T<T_e$, 图 b) $T=T_e$, 图 c) $T>T_e$.

利用 $f(\phi)$ 我们可以给出 Ginzburg-Landau 型能量公式:

$$
E[\phi] = \int_{\Omega}\left[\frac{\varepsilon^2}{2} | \nabla \phi|^2
+f(\phi) \right] \mathrm d V
$$
这里的 $\varepsilon$ 是一个小的正参数, $\Omega$ 为物质的区域.

**Allen-Cahn 方程**

考虑梯度系统 

$$
\tau \frac{\partial \phi}{\partial t} = - \frac{\delta E}{ \delta \phi},
$$

可以得到界面公式:

$$
\tau \frac{\partial \phi}{\partial t} = \varepsilon^2 \nabla^2 \phi -f'(\phi).
$$

对于第一类 $h(\phi)$:

$$
\tau \frac{\partial \phi}{\partial t} = \varepsilon^2 \nabla^2 \phi 
+2a^2\phi(1-\phi)\left(\phi-\frac{1}{2}+\frac{15 \Delta f }{a^2}
\phi(1-\phi)\right).
$$

对于第二类 $h(\phi)$:

$$
\tau \frac{\partial \phi}{\partial t} = 
\varepsilon^2 \nabla^2 \phi +
2a^2\phi(1-\phi)\left(\phi-\frac{1}{2} + 
\frac{3 \Delta f }{a^2}\right) (\text{其中要求} | \Delta f | <\frac{a^2}{6}).
$$

该方程是一个双稳态 ($\phi=0$ 和 $\phi=1$) 反应扩散方程, 称为 Allen-Cahn 方程. 

选取参数 $\varepsilon$ 和 $a$, 使得 $\varepsilon \ll a$, 只要 $\phi$ 
在空间中改变不剧烈, 则上面两类方程右端项中带 $2a^2$ 的项占主导地位. 
$\phi$ 倾向于处处到 0 或都处处取 1. 如果 $\phi = 0$ 和 $\phi = 1$
的区域同时存在, 它们之间必须存在一个边界,
那么两类方程中的第一项的作用就不能忽略. 
实际上, $\phi$ 在边界区域光滑的变化, 形成一个薄薄的内部过度层,
这是两项平衡的结果. 我们称这个过渡层 "front" 或者 "interface".


$$
E[\phi_0] = a\varepsilon \int_0^1\phi_0(1-\phi_0)\mathrm d \phi_0 =
\frac{a\varepsilon}{6}
$$


## 参考文献

1. Programming Phase-Field Modeling.
1. [Phase Field methods: From fundamentals to applications](https://www.youtube.com/watch?v=FTiBq1o-8e4).
1. [GIAN-Phase Field Modelling](https://www.youtube.com/watch?v=3Nt5hS8S2qY).

