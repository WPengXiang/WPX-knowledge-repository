# 经典水平集方法及其离散

考虑区域 $\Omega \in \mathbb R^n$ ,
令 $\Omega = \Omega_1 \cup \Omega_2$, 其中 $\Omega_1$ 和 $\Omega_2$ 代表 
为两种不同的材料, 则材料之间的界面可以表示为所有闭子域的交集

$$
\Gamma = \{ \boldsymbol x : \boldsymbol x \in \bar\Omega_1 \cap \bar\Omega_2 \},
$$

我们考虑用符号距离函数来作为水平集函数

$$
\begin{aligned}
d(\boldsymbol x ) =
\begin{cases}
\min_{\boldsymbol x_i \in \Gamma}
\| \boldsymbol x -\boldsymbol x_i\| , x \in \Omega_1 \\
-\min_{\boldsymbol x_i \in \Gamma}
\| \boldsymbol x -\boldsymbol x_i\| , x \in \Omega_2 \\
\end{cases}
\end{aligned}
$$

界面 $\Gamma$ 用水平集的函数隐式来表示

$$
\Gamma = \{\boldsymbol x: \varphi(\boldsymbol x) = 0\}
$$

假定求解区域  $\Omega$ 上有一个速度场 $\boldsymbol u$ 驱动界面 $\Gamma$ 的演化，
其对应演化方程为

$$
\frac{\partial \varphi}{\partial t}+\boldsymbol u \cdot \nabla \varphi = 0
$$

$\qquad$ 数值求求解经典的水平集演化方程, 水平集函数 $\varphi$ 不会一直是符号距离函数.
为了将 $\varphi$ 重置为符号距离函数, 可以求解下面的方程(称为**重置方程**) 

$$
\frac{\partial \varphi}{\partial \tau}(\boldsymbol x,\tau) + 
sign(\varphi_0)(| \nabla \varphi(\boldsymbol x,\tau)|-1) = 0,
$$

其中 

$$
\varphi_0 = \varphi(\boldsymbol x, t_n)
$$

为水平集演化方程第 $n$ 个时刻的解. 而 $\tau$ 是一个虚假的时间变量. 
通过求解上面的**重置方程**, 即可得到想要的符号距离函数.

## 数值离散

首先对运输方程进行离散, 时间上采取 Crank-Nicholson 方法, 即

$$
\frac{\varphi^{n+1}-\varphi^n}{\Delta t} + 
\boldsymbol u \cdot \nabla (\frac{\varphi^{n+1}+\varphi^n}{2})
=0,
$$

空间上用有限元来离散, 得到

$$
(\varphi^{n+1},w) + 
\frac{\Delta t}{2}(\boldsymbol u \cdot \nabla \varphi^{n+1},w)
=
(\varphi^{n},w) - 
\frac{\Delta t}{2}(\boldsymbol u \cdot \nabla \varphi^{n},w),
$$

之后离散重置方程,为了解决数值不稳定性,我们在重置方程后面加一个人工粘性项 $\alpha \nabla^2 \varphi$,时间上采取Eular
$$
\frac{\varphi^{n+1}- \varphi^{n}}{\Delta \tau} + sign(\varphi_0)(|\nabla \varphi^n|-1)+\alpha \nabla^2 \phi^n
=0
$$

空间离散还是用有限元,最后得到
$$
\frac{1}{\Delta \tau}(\varphi^{n+1},w) = 
\frac{1}{\Delta \tau}(\varphi^n,w) -
(sign(\varphi_0)(|\nabla \varphi^n|)-1,w)
- \alpha(\nabla \varphi^n, \nabla w)
$$

