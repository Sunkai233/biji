# NSFnets（Navier-Stokes流动网络）：用于不可压缩Navier-Stokes方程的物理信息神经网络

**作者**：金晓伟^{a,b}，蔡胜泽^c，李辉^{a,b,*}，George Em Karniadakis^{c,*}

**单位**：

^a 哈尔滨工业大学，智能防控与土木工程减灾实验室，工业与信息化部，哈尔滨 150090，中国  
^b 哈尔滨工业大学，结构动态行为与控制实验室，教育部，哈尔滨 150090，中国  
^c 布朗大学，应用数学系，普罗维登斯，RI 02912，美国  

**摘要**

我们采用物理信息神经网络（PINNs）来模拟从层流到湍流的不可压缩流动。我们通过考虑Navier-Stokes方程的两种不同形式进行PINN模拟：速度-压力（VP）形式和涡量-速度（VV）形式。我们将这些专门针对Navier-Stokes流动的PINNs称为NSFnets。解析解和直接数值模拟（DNS）数据库为NSFnets模拟提供了适当的初始和边界条件。空间和时间坐标是NSFnets的输入，而瞬时速度和压力场是VP-NSFnet的输出，瞬时速度和涡量场是VV-NSFnet的输出。这两种不同形式的Navier-Stokes方程以及初始和边界条件被嵌入到PINNs的损失函数中。VP-NSFnet无需为压力提供数据，压力作为隐状态通过不可压缩性约束获得，而无需拆分方程。NSFnets模拟结果在损失函数收敛后具有良好的准确性，验证了NSFnets可以有效地使用VP或VV形式模拟复杂不可压缩流动。对于层流解，我们展示了VV形式比VP形式更准确。对于湍流通道流动，我们展示了NSFnets可以在 $Re_\tau \sim 1,000$ 时维持湍流，但由于训练成本高，我们仅考虑了通道域的一部分，并对DNS数据库提供的边界施加速度边界条件。我们还对损失函数中数据/物理部分的权重进行了系统研究，并探索了一种动态计算权重的新方法，以加速训练并提高准确性。我们的结果表明，通过适当调整损失函数中的权重（手动或动态），NSFnets在层流和湍流中的准确性可以得到改善。

**关键词**：PINNs，DNS，湍流，速度-压力形式，涡量-速度形式，自动微分

{

#### 通俗解释

研究人员想用一种叫 **PINNs（物理信息神经网络）** 的方法来模拟空气或液体流动，尤其是从**层流**（流体运动平稳、有序）到**湍流**（流体运动混乱、充满涡旋）的复杂情况。

他们主要尝试了 **Navier–Stokes方程** 的两种数学表达方式：

1. **速度–压力（VP）形式**
   - 输入：空间和时间坐标
   - 输出：速度场 + 压力场
   - 特点：不需要额外提供真实的压力数据，因为压力可以通过“不可压缩性约束”（流体不能被压缩的数学条件）自动推导出来。
2. **涡量–速度（VV）形式**
   - 输入：空间和时间坐标
   - 输出：速度场 + 涡量场
   - 特点：用涡量代替压力，避免了压力难测量的问题。

这两种版本的PINNs被称作 **NSFnets**。

------

#### 他们做了什么实验？

- **层流情况**
  - 发现 VV 形式比 VP 形式表现更好，更接近真实解。
- **湍流情况**
  - 在比较高的雷诺数条件下（$Re_\tau \sim 1000$），NSFnets 仍然能维持湍流的特征。
  - 但因为训练成本太高，只能在通道流的一部分区域里进行模拟，并且在边界施加了速度条件（借助 DNS 数据库的数据）。

------

#### 他们还研究了什么？

在 PINNs 中，损失函数里通常有两部分：

- **数据损失**（和真实数据比对）
- **物理损失**（约束方程必须满足物理规律）

不同任务需要不同的权重。研究人员：

- 系统地试了不同的手动权重分配。
- 还提出了一种“动态权重”计算方法，让训练更快、精度更高。

------

### 总结

- **NSFnets** 可以用来模拟复杂的不可压缩流动。
- **VV形式更适合层流**，而 **VP形式也能处理湍流**。
- **通过调整或动态分配损失函数权重**，能进一步提升模型的精度和训练效率。

}

---

## 1. 引言

在过去五年中，研究人员尝试将神经网络（NNs）集成到不可压缩Navier-Stokes方程的求解中，采用了不同的方法。对于湍流，最常见的方法是推导数据驱动的湍流闭合模型。例如，Ling等人[1]提出了一种数据驱动的雷诺平均Navier-Stokes（RANS）湍流闭合模型，通过将伽利略不变性嵌入深度神经网络，展示了更高的雷诺应力预测准确性。类似地，Wang等人[2]使用随机森林回归预测RANS预测的雷诺应力与DNS数据的偏差，从而高精度地预测雷诺应力。Jiang等人[3]开发了一种新颖的RANS应力闭合模型，结合机器学习辅助参数化和非局部效应，旨在减少结构和参数误差，并更准确地描述雷诺应力的各向异性。对于各向同性湍流的大涡模拟（LES），Zhou等人[4]开发了一种基于单隐藏层神经网络的数据驱动子网格尺度模型。此外，还研究了一些流体力学中的降阶模型（ROMs）或快速预测模型。例如，卷积神经网络（CNNs）被用来构建圆柱尾流的预测模型[5]，时间卷积神经网络用于建立圆柱尾流正交分解（POD）模式系数的预测模型[6]。双向递归神经网络基于少量速度测量预测圆柱尾流的POD系数[7]，获得了比扩展POD方法[8,9]更准确的结果。此外，深度学习技术还被应用于粒子图像测速（PIV）以分析湍流边界层的实验室数据[10]。关于通过引入各种机器学习技术在流体力学中取得的进展的全面总结可以在[11,12]中找到。

我们采取了不同的路径，利用神经网络的通用逼近属性，结合自动微分，使我们能够开发无需生成网格的Navier-Stokes“求解器”。这些求解器易于实现，特别适用于多物理和逆流体力学问题。特别是，Raissi等人[13,14,15]首次引入了物理信息神经网络（PINNs）的概念，用于求解涉及多种不同类型PDE的前向和逆问题。这种方法还被用于模拟涡激振动[16]以及处理不适定逆流体力学问题，这被称为“隐藏流体力学”框架[17]。上述工作中考虑的流动是相对低雷诺数的层流，由不可压缩Navier-Stokes方程的速度-压力（VP）形式描述。一个基本问题是，PINNs能否直接模拟湍流，类似于使用高阶离散化的直接数值模拟（DNS）[18,19]。另一个重要问题是，是否存在另一种Navier-Stokes方程形式，例如涡量-速度（VV）形式，可能实现更高精度或更有效的训练。

在本研究中，我们通过使用二维和三维流动的解析解以及DNS湍流通道流数据库[20,21,22]，系统地解决了上述两个问题。特别是，我们通过考虑Navier-Stokes方程的两种形式进行PINN模拟：VP形式和VV形式，我们将这些针对Navier-Stokes流动的PINNs称为NSFnets。对于VP-NSFnet，输入是空间和时间坐标，输出是瞬时速度和压力场。对于VV-NSFnet，输入同样是空间和时间坐标，输出是瞬时速度和涡量场。我们使用自动微分（AD）[23]处理Navier-Stokes方程中的微分算子，这比数值微分具有更高的计算效率。然而，它不需要网格，并避免了经典的人为色散和扩散误差。此外，通过AD，我们微分的是神经网络而不是数据本身，因此可以处理噪声输入或有限正则性的解。使用PINNs采用VP和VV形式还有显著的优势。例如，为了推断压力方程，我们不使用传统的分裂方法[24]中的额外泊松压力方程，VP-NSFnet无需为压力提供边界或初始条件数据；压力是隐状态，通过不可压缩性约束获得。类似地，在VV-NSFnet中，涡量边界条件以约束的形式直接嵌入到损失函数中。

我们模拟了几个层流，包括二维稳态Kovasznay流、二维非稳态圆柱尾流和三维非稳态Beltrami流，使用这两种NSFnets。我们使用[25]的工作，系统研究了损失函数中各组成部分的动态权重，以加速训练并提高准确性。我们还报告了使用PINNs直接模拟湍流的初步结果。为此，我们考虑了 $Re_\tau \sim 1,000$ 的湍流通道流，主要使用VP-NSFnet，因为可用数据库基于VP形式。我们通过考虑通道中不同位置、不同大小的子域以及不同时间间隔进行NSFnet模拟。此外，我们研究了损失函数中权重对VP-NSFnet准确性的影响。

本文组织如下。我们首先在第2节介绍NSFnets，并在第3节介绍层流的问题设置和NSFnet模拟结果。然后在第4节介绍湍流通道流的VP-NSFnet结果。我们在第5节总结我们的发现。

{

#### 通俗解释版引言

在过去几年里，很多研究人员尝试把 **神经网络（NNs）** 和 **Navier-Stokes方程**（流体运动的基本方程）结合起来，来帮助模拟流体，尤其是湍流。

传统做法里，人们常用机器学习来改进**湍流模型**（比如RANS、LES）：

- 有的研究用深度神经网络提高 **雷诺应力** 的预测精度。
- 有的用随机森林来修正传统模型和DNS（高精度模拟）之间的误差。
- 还有人结合神经网络和参数化方法，做出了更精细的湍流应力模型。
- 此外，CNN、RNN 等也被用于预测圆柱尾流、降阶模型（ROM）、以及分析实验室中的湍流测量数据。

这些工作大多数还是基于已有的数值方法（比如网格离散），机器学习只是在 **模型闭合** 或 **降阶预测** 上起辅助作用。

------

#### 我们的不同思路

我们走了一条不一样的路：直接把神经网络当作 **Navier-Stokes方程的“求解器”**。

- 这种方法叫 **PINNs（物理信息神经网络）**。
- 它利用神经网络的“通用逼近”能力 + **自动微分（AD）**，把偏微分方程的约束直接写进损失函数里。
- 优点：
  - 不需要生成网格（省去了复杂的数值离散化）。
  - 避免数值模拟常见的“数值扩散”“色散误差”。
  - 即使输入数据有噪声，AD 也能稳健地处理。

Raissi 等人最早提出 PINNs 来解各种 PDE，并应用在流体问题上（比如涡激振动、逆问题“隐藏流体力学”框架）。不过之前的工作基本只研究了低雷诺数的层流。

于是我们提出了一个关键问题：

- **PINNs能否直接模拟湍流？**
- **哪种Navier-Stokes方程形式更合适？速度–压力（VP）还是涡量–速度（VV）？**

------

#### 我们的工作

在本文里，我们系统研究了这两个问题。

- 我们提出了 **NSFnets**：
  - **VP-NSFnet**：输入时空坐标 → 输出速度 + 压力。
  - **VV-NSFnet**：输入时空坐标 → 输出速度 + 涡量。
  - 压力或涡量并不需要单独的数据，它们通过方程约束或边界条件自然得到。
- 技术手段：
  - 用自动微分（AD）来处理方程中的微分算子。
  - 在损失函数里动态调整数据项和物理项的权重（提升训练速度和准确度）。
- 我们的实验：
  1. **层流案例**：Kovasznay流、圆柱尾流、三维Beltrami流。
  2. **湍流案例**：直接用 PINNs 模拟 $Re_\tau \sim 1000$ 的通道湍流（初步结果）。

在湍流实验中，我们主要使用 VP-NSFnet，因为 DNS 数据库本身是基于速度–压力形式的。我们尝试了不同子域大小、不同时间段，并研究了损失函数权重对结果的影响。

}

---

## 2. 求解方法

我们介绍了不可压缩三维非稳态Navier-Stokes方程的两种形式：速度-压力（VP）形式和涡量-速度（VV）形式，以及对应的物理信息神经网络（PINNs），如图1所示。

### 2.1 速度-压力（VP）形式

不可压缩Navier-Stokes方程的VP形式为：

$$
\begin{aligned}
\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} &= -\nabla p + \frac{1}{Re} \nabla^2 \mathbf{u} \quad \text{在 } \Omega, \tag{1a} \\
\nabla \cdot \mathbf{u} &= 0 \quad \text{在 } \Omega, \tag{1b} \\
\mathbf{u} &= \mathbf{u}_\Gamma \quad \text{在 } \Gamma_D, \tag{1c} \\
\frac{\partial \mathbf{u}}{\partial \mathbf{n}} &= 0 \quad \text{在 } \Gamma_N, \tag{1d}
\end{aligned}
$$

其中，$t$ 是无量纲时间，$\mathbf{u}(x, t) = [u, v, w]^T$ 是无量纲速度向量，$p$ 是无量纲压力，$Re = U_{ref}D_{ref}/\nu$ 是雷诺数，由特征长度 $D_{ref}$、参考速度 $U_{ref}$ 和运动粘度 $\nu$ 定义。求解方程（1）需要初始和边界条件。这里，$\Gamma_D$ 和 $\Gamma_N$ 分别表示Dirichlet和Neumann边界。在本研究中，我们不使用传统的计算流体力学（CFD）方法，而是探索使用神经网络（NNs）求解Navier-Stokes方程的可能性。换句话说，Navier-Stokes方程的解由深度神经网络逼近，网络以空间和时间坐标作为输入，预测相应的速度和压力场，即 $(t, x, y, z) \mapsto (u, v, w, p)$。

PINNs求解方程（1）的示意图如图1a所示，包含一个全连接网络和残差网络。这里，非线性激活函数 $\sigma$ 为双曲正切函数 $\tanh$。对于VP形式，残差包括动量方程和不可压缩性约束的误差。为了计算Navier-Stokes方程的残差 $e_{VP1}$ 到 $e_{VP4}$，使用自动微分（AD）计算偏微分算子，这可以在深度学习框架中直接实现，例如在TensorFlow中使用“tf.gradients()”。

训练VP-NSFnet以获得方程（1）解的损失函数定义如下：

$$
\begin{aligned}
L &= L_e + \alpha L_b + \beta L_i, \tag{2a} \\
L_e &= \frac{1}{N_e} \sum_{i=1}^4 \sum_{n=1}^{N_e} |e^n_{VP i}|^2, \tag{2b} \\
L_b &= \frac{1}{N_b} \sum_{n=1}^{N_b} |\mathbf{u}^n - \mathbf{u}^n_b|^2, \tag{2c} \\
L_i &= \frac{1}{N_i} \sum_{n=1}^{N_i} |\mathbf{u}^n - \mathbf{u}^n_i|^2, \tag{2d}
\end{aligned}
$$

其中，$L_e$、$L_b$ 和 $L_i$ 分别表示对应于Navier-Stokes方程残差、边界条件和初始条件的损失函数部分；$N_b$、$N_i$ 和 $N_e$ 表示不同部分的训练数据数量；$\mathbf{u}^n_b = [u^n_b, v^n_b, w^n_b]^T$ 和 $\mathbf{u}^n_i = [u^n_i, v^n_i, w^n_i]^T$ 分别是边界和初始时间第 $n$ 个数据点的给定速度；$e^n_{VP i}$ 表示第 $i$ 个方程在第 $n$ 个数据点的残差。权重系数 $\alpha$ 和 $\beta$ 用于平衡损失函数的不同项，加速训练过程中的收敛。我们将初始和边界条件视为监督数据驱动部分，将Navier-Stokes方程的残差视为无监督的物理信息部分。需要注意的是，VP-NSFnet无需为压力提供边界或初始条件数据，压力是隐状态，通过不可压缩性约束获得，而无需像传统CFD方法[19]那样拆分Navier-Stokes方程。使用自适应优化算法Adam[26]最小化损失函数（2）。神经网络参数使用Xavier方案[27]随机初始化。当NSFnet训练收敛时，即总损失函数达到某个很小的值时，得到解。

### 2.2 涡量-速度（VV）形式

我们还提出了Navier-Stokes方程VV形式的NSFnets，作为模拟不可压缩流动的替代方法；VP和VV形式的等价性已在[28,29]中证明。Navier-Stokes方程的VV形式的旋转形式为：

$$
\begin{aligned}
\frac{\partial \boldsymbol{\omega}}{\partial t} + \nabla \times (\boldsymbol{\omega} \times \mathbf{u}) &= -\frac{1}{Re} \nabla \times \nabla \times \boldsymbol{\omega} \quad \text{在 } \Omega, \tag{3a} \\
\nabla^2 \mathbf{u} &= -\nabla \times \boldsymbol{\omega} \quad \text{在 } \Omega, \tag{3b} \\
\boldsymbol{\omega} &= \nabla \times \mathbf{u} \quad \text{在 } \Gamma, \tag{3c} \\
\mathbf{u} &= \mathbf{u}_\Gamma \quad \text{在 } \Gamma_D, \tag{3d} \\
\frac{\partial \mathbf{u}}{\partial \mathbf{n}} &= 0 \quad \text{在 } \Gamma_N, \tag{3e} \\
\nabla \cdot \mathbf{u} &= 0 \quad \text{在 } \Gamma \text{ 的一个点}, \tag{3f} \\
\boldsymbol{\omega} &= \nabla \times \mathbf{u} \quad \text{在 } t=0 \text{ 时在 } \Omega, \tag{3g}
\end{aligned}
$$

其中，$\boldsymbol{\omega} = [\omega_x, \omega_y, \omega_z]^T$ 是具有三个分量的涡量。边界条件由方程（3c）到（3f）定义，初始条件由方程（3g）约束。同样，我们假设方程（3）的解由神经网络逼近，其功能可以写为 $(t, x, y, z) \mapsto (u, v, w, \omega_x, \omega_y, \omega_z)$。求解方程（3）的VV-NSFnet架构如图1b所示，其中 $e_{VV1}$ 到 $e_{VV6}$ 表示Navier-Stokes方程（3a）和（3b）的VV形式的残差。VV-NSFnet的相应损失函数定义如下：

$$
\begin{aligned}
L &= L_e + \alpha L_b + \beta L_i, \tag{4a} \\
L_e &= \frac{1}{N_e} \sum_{i=1}^6 \sum_{n=1}^{N_e} |e^n_{VV i}|^2, \tag{4b} \\
L_b &= \frac{1}{N_b} \sum_{n=1}^{N_b} \left( |\mathbf{u}^n - \mathbf{u}^n_b|^2 + |\boldsymbol{\omega}^n - \nabla \times \mathbf{u}^n_b|^2 + |\nabla \cdot \mathbf{u}^n_b|^2 \right), \tag{4c} \\
L_i &= \frac{1}{N_i} \sum_{n=1}^{N_i} \left( |\mathbf{u}^n - \mathbf{u}^n_i|^2 + |\boldsymbol{\omega}^n - \nabla \times \mathbf{u}^n_i|^2 \right), \tag{4d}
\end{aligned}
$$

其中，$\boldsymbol{\omega}^n = [\omega^n_x, \omega^n_y, \omega^n_z]^T$ 表示NSFnet在第 $n$ 个数据点的涡量；$e^n_{VV i}$ 表示第 $i$ 个方程在第 $n$ 个数据点的残差。注意，损失函数中仅提供速度的边界和初始值。对于涡量项，边界和初始条件以约束形式嵌入到损失函数（4c）和（4d）中。神经网络参数同样使用Adam优化器学习。

需要注意的是，损失函数（2）和（4）中的权重系数在训练过程中起着非常重要的作用。然而，为NSFnets选择合适的权重通常非常繁琐。一方面，最优的 $\alpha$ 和 $\beta$ 值因问题而异，我们无法为不同流动固定它们。另一方面，任意调整权重需要试错过程，相当繁琐且耗时。为了解决这个问题，我们应用了[25]的动态权重策略来选择NSFnets模拟中的 $\alpha$ 和 $\beta$。动态权重的思想是利用网络训练期间反向传播的梯度统计自适应更新系数。对于一般的梯度下降算法，NSFnets参数的迭代公式可以表示为：

$$
\theta^{(k+1)} = \theta^{(k)} - \eta \nabla_\theta L_e - \eta \alpha \nabla_\theta L_b - \eta \beta \nabla_\theta L_i, \tag{5}
$$

其中，$\theta$ 表示神经网络的参数，即所有全连接层的权重，$k$ 是迭代步骤，$\eta$ 是学习率。为了平衡方程（5）中不同项的贡献，Wang等人[25]提出在网络训练期间使用动态权重策略。在每次训练步骤，例如第 $(k+1)$ 次迭代，$\alpha$ 和 $\beta$ 的估计值可以通过以下公式计算：

$$
\hat{\alpha}^{(k+1)} = \frac{\max_\theta \{|\nabla_\theta L_e|\}}{|\nabla_\theta \alpha^{(k)} L_b|}, \quad \hat{\beta}^{(k+1)} = \frac{\max_\theta \{|\nabla_\theta L_e|\}}{|\nabla_\theta \beta^{(k)} L_i|}, \tag{6}
$$

其中，$\max_\theta \{|\nabla_\theta L_e|\}$ 是 $|\nabla_\theta L_e|$ 的最大值；$|\nabla_\theta \alpha^{(k)} L_b|$ 和 $|\nabla_\theta \beta^{(k)} L_i|$ 分别表示 $|\nabla_\theta \alpha^{(k)} L_b|$ 和 $|\nabla_\theta \beta^{(k)} L_i|$ 的均值。作为替代，我们还提出以下方式估计 $\alpha$ 和 $\beta$：

$$
\hat{\alpha}^{(k+1)} = \frac{|\nabla_\theta L_e|}{|\nabla_\theta L_b|}, \quad \hat{\beta}^{(k+1)} = \frac{|\nabla_\theta L_e|}{|\nabla_\theta L_i|}. \tag{7}
$$

神经网络参数的梯度可以通过深度学习框架中的AD轻松计算。因此，下一迭代的权重系数使用移动平均形式更新：

$$
\alpha^{(k+1)} = (1 - \lambda) \alpha^{(k)} + \lambda \hat{\alpha}^{(k+1)}, \quad \beta^{(k+1)} = (1 - \lambda) \beta^{(k)} + \lambda \hat{\beta}^{(k+1)}, \tag{8}
$$

其中 $\lambda = 0.1$。动态权重策略将在后续的大多数NSFnet模拟中应用。

我们介绍了对应于VP形式和VV形式的两种不同NSFnets。我们使用不同大小的NSFnets进行了多项数值实验。然而，通过优化架构大小、学习率甚至优化器，可能进一步提高每种情况的准确性，这超出了当前工作的范围。

{

## 2. 求解方法（通俗解释）

在这里，我们用两种不同的方式来写不可压缩的三维非稳态 Navier–Stokes 方程：

1. **速度–压力（VP）形式**
2. **涡量–速度（VV）形式**

基于这两种形式，我们设计了对应的 **PINNs（物理信息神经网络）** 来近似求解。

------

### 2.1 速度–压力（VP）形式

Navier–Stokes 方程的 VP 形式为：
$$
\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u}\cdot\nabla)\mathbf{u} = -\nabla p + \frac{1}{Re}\nabla^2 \mathbf{u}, \quad \mathbf{x}\in\Omega
$$
其中：

- $\mathbf{u}(x,t)=[u,v,w]^T$ 是速度向量
- $p$ 是压力
- $Re=U_{ref}D_{ref}/\nu$ 是雷诺数

这里的思路是：用神经网络直接逼近解，即
$$
(t,x,y,z)\;\mapsto\;(u,v,w,p)
$$

------

#### VP-PINN 的损失函数

损失函数由三部分组成：
$$
L = L_e + \alpha L_b + \beta L_i
$$
其中：

- $L_e$：方程残差
- $L_b$：边界条件误差
- $L_i$：初始条件误差

具体写法：
$$
L_e = \frac{1}{N_e}\sum_{i=1}^4 \sum_{n=1}^{N_e} |e^n_{VP,i}|^2
$$
这里，$\alpha,\beta$ 是权重，用来平衡三部分的重要性。
 特别之处在于：**不需要给压力提供边界/初始条件数据**，压力通过不可压缩性约束自然得到。

------

### 2.2 涡量–速度（VV）形式

VV 形式的方程如下：
$$
\frac{\partial \boldsymbol{\omega}}{\partial t} + \nabla \times (\boldsymbol{\omega}\times \mathbf{u}) = -\frac{1}{Re}\nabla \times \nabla \times \boldsymbol{\omega}
$$
其中 $\boldsymbol{\omega} = [\omega_x,\omega_y,\omega_z]^T$ 是涡量。
 神经网络的映射关系是：
$$
(t,x,y,z)\;\mapsto\;(u,v,w,\omega_x,\omega_y,\omega_z)
$$

------

#### VV-PINN 的损失函数

$$
L = L_e + \alpha L_b + \beta L_i
$$

其中：
$$
L_e = \frac{1}{N_e}\sum_{i=1}^6 \sum_{n=1}^{N_e} |e^n_{VV,i}|^2
$$
区别在于：只需要提供速度的边界和初始值，涡量条件由损失函数自动嵌入。

------

### 动态权重策略

一个难点是：如何选择 $\alpha,\beta$？
 不同问题下它们的最优值不同，手动调很费时。
 因此引入 **动态权重** 方法。

神经网络参数更新公式：
$$
\theta^{(k+1)} = \theta^{(k)} - \eta \nabla_\theta L_e - \eta \alpha \nabla_\theta L_b - \eta \beta \nabla_\theta L_i
$$
其中 $\theta$ 是网络参数，$\eta$ 是学习率。

动态权重的估计方式例如：
$$
\hat{\alpha}^{(k+1)} = \frac{\max_\theta |\nabla_\theta L_e|}{|\nabla_\theta \alpha^{(k)} L_b|}, 
\quad 
\hat{\beta}^{(k+1)} = \frac{\max_\theta |\nabla_\theta L_e|}{|\nabla_\theta \beta^{(k)} L_i|}
$$
或者更简单的比例：
$$
\hat{\alpha}^{(k+1)} = \frac{|\nabla_\theta L_e|}{|\nabla_\theta L_b|}, 
\quad 
\hat{\beta}^{(k+1)} = \frac{|\nabla_\theta L_e|}{|\nabla_\theta L_i|}
$$
最后用移动平均更新：
$$
\alpha^{(k+1)} = (1-\lambda)\alpha^{(k)} + \lambda \hat{\alpha}^{(k+1)}, 
\quad 
\beta^{(k+1)} = (1-\lambda)\beta^{(k)} + \lambda \hat{\beta}^{(k+1)}
$$
其中 $\lambda=0.1$。

------

### 总结

- VP 形式：输出速度 + 压力，压力由不可压缩条件自动推得。
- VV 形式：输出速度 + 涡量，涡量边界条件嵌入损失函数。
- 动态权重：训练过程中自适应调整 $\alpha,\beta$，避免繁琐的手动调参，提高精度和效率。

}

---

## 3. 层流模拟

在本节中，我们应用提出的NSFnets模拟不同的不可压缩Navier-Stokes流动，包括二维稳态Kovasznay流、二维非稳态圆柱尾流和三维非稳态Beltrami流。我们展示了VV和VP-NSFnets之间的比较，并研究了动态权重对解准确性的影响。其他增强方法包括使用自适应激活函数加速训练[30,31]，但在当前工作中我们未追求这一点。为了评估NSFnet模拟的性能，我们定义了每个时间步的相对 $L_2$ 误差为：

$$
\epsilon_V = \frac{\|\hat{V} - V\|_2}{\|V\|_2}, \tag{9}
$$

其中，$V$ 表示速度分量 $(u, v, w)$ 或压力 $p$，帽号表示NSFnets推断的值。参考速度和压力由解析解或高保真DNS结果给出。需要注意的是，为了评估NSFnet解的准确性，我们对NSFnet模拟结果进行移位，使参考DNS结果和NSFnet结果的压力均值相同。

### 3.1 Kovasznay流

我们使用Kovasznay流作为第一个测试案例，展示NSFnets的性能。这种二维稳态Navier-Stokes流具有以下解析解：

$$
\begin{aligned}
u(x, y) &= 1 - e^{\lambda x} \cos(2\pi y), \\
v(x, y) &= \frac{\lambda}{2\pi} e^{\lambda x} \sin(2\pi y), \\
p(x, y) &= \frac{1}{2} (1 - e^{2\lambda x}),
\end{aligned} \tag{10}
$$

其中，

$$
\lambda = \frac{1}{2\nu} - \sqrt{\frac{1}{4\nu^2} + 4\pi^2}, \quad \nu = \frac{1}{Re} = \frac{1}{40}.
$$

我们考虑的计算域为 $[-0.5, 1.0] \times [-0.5, 1.5]$。每个边界上固定空间坐标有101个点，因此边界条件的训练数据有 $N_b = 400$。为了计算NSFnets的方程损失，域内随机选择2601个点。此稳态流没有初始条件。所有NSFnets采用两步训练评估：首先使用Adam优化器进行5000、5000、50000和50000次迭代，学习率分别为 $1 \times 10^{-3}$、$1 \times 10^{-4}$、$1 \times 10^{-5}$ 和 $1 \times 10^{-6}$，然后应用有限内存Broyden-Fletcher-Goldfarb-Shanno算法（L-BFGS-B）微调结果。L-BFGS-B的训练过程根据增量容差自动终止。

对于Kovasznay流，我们首先研究神经网络架构的影响。我们通过改变隐藏层数和每层神经元数量，采用不同大小的网络。边界约束的权重系数 $\alpha$ 在训练这些NSFnets时选择为100。结果总结在表1中，每个数字是十次独立模拟中的最佳值。如表所示，两种NSFnets形式均能获得高精度的解，相对误差在 $10^{-5}$ 到 $10^{-3}$ 之间。我们还观察到，随着网络规模的增加，NSFnets的性能有所提高。对于小型网络，VP形式优于VV形式，而使用大型网络时，VV-NSFnet提供更准确的解。

我们还研究了边界约束的权重系数 $\alpha$。除了 $\alpha = 100$，我们还应用了 $\alpha = 1$ 并实现了动态权重（即方程（6）和（7））进行比较。在这项评估中，我们采用具有4个隐藏层和每层50个神经元的小型神经网络。训练过程中学习率如上所述逐渐降低。这一策略与[25]中使用的动态权重一致。训练过程中动态权重的变化如图2所示。这里，动态权重策略的 $\alpha$ 初始化为1。我们发现，系数 $\alpha$ 会振荡，并因学习率的变化而变化。VV-NSFnet的动态权重策略类似。动态权重的结果优于固定系数值（即 $\alpha = 1$ 和 $\alpha = 100$）的情况。$\alpha$ 的最终极限值在10的量级。VP-NSFnet在 $\alpha = 1$、$\alpha = 100$ 和动态权重策略下的损失函数如图3(a)、(b)和(c)所示，VV-NSFnet的损失函数如图3(d)、(e)和(f)所示。从训练损失的曲线来看，Adam优化器对VP-NSFnet表现稳健，而对VV-NSFnet的表现不一致。应用两步优化可以获得更一致的结果。不同权重的NSFnet模拟的相对 $L_2$ 误差如表2所示。对于固定权重（$\alpha = 1$ 和 $\alpha = 100$），VP-NSFnet在模拟Kovasznay流时优于VV-NSFnet。$\alpha = 100$ 的神经网络性能略优于 $\alpha = 1$ 的情况。在网络训练期间应用动态权重，我们可以获得比前两种情况更准确的解。

为了展示动态权重的有效性，我们分析了损失函数关于NSFnets参数的梯度。第10000次迭代后反向传播梯度（$\nabla_\theta L_e$ 和 $\nabla_\theta (\alpha L_b)$）的直方图如图4所示。我们的目标是平衡 $\nabla_\theta L_e$ 和 $\nabla_\theta (\alpha L_b)$ 的分布，从而使这两项对参数更新（即方程（5））的贡献相等。如图4(a)和4(e)所示，当边界约束没有权重系数（即 $\alpha = 1$）时，两个不同项的梯度不平衡。应用动态权重时，$\nabla_\theta L_e$ 和 $\nabla_\theta (\alpha L_b)$ 的直方图更加一致。对于VP-NSFnet，第二种动态权重形式（即方程（7））比第一种（方程（6）给出的权重）表现更好。然而，对于VV-NSFnet，这种情况相反，如图4(g)和4(h)所示。

**表1**：Kovasznay流：不同大小NSFnets的速度和压力解的相对 $L_2$ 误差（$\alpha = 100$，NN大小为隐藏层数 × 每层神经元数）。

| NN大小   | VP-NSFnet    |              |              | VV-NSFnet    |              |
| -------- | ------------ | ------------ | ------------ | ------------ | ------------ |
|          | $\epsilon_u$ | $\epsilon_v$ | $\epsilon_p$ | $\epsilon_u$ | $\epsilon_v$ |
| 4 × 50   | 0.076%       | 0.412%       | 0.516%       | 0.131%       | 0.368%       |
| 7 × 50   | 0.038%       | 0.255%       | 0.114%       | 0.111%       | 0.520%       |
| 7 × 100  | 0.016%       | 0.183%       | 0.062%       | 0.078%       | 0.397%       |
| 10 × 100 | 0.020%       | 0.115%       | 0.044%       | 0.040%       | 0.233%       |
| 10 × 200 | 0.012%       | 0.103%       | 0.042%       | 0.022%       | 0.121%       |
| 10 × 250 | 0.011%       | 0.101%       | 0.031%       | 0.011%       | 0.081%       |
| 10 × 300 | 0.008%       | 0.072%       | 0.041%       | 0.004%       | 0.037%       |

**表2**：Kovasznay流：不同权重的速度和压力解的相对 $L_2$ 误差。NN大小为 4 × 50。

| 权重            | VP-NSFnet    |              |              | VV-NSFnet    |              |
| --------------- | ------------ | ------------ | ------------ | ------------ | ------------ |
|                 | $\epsilon_u$ | $\epsilon_v$ | $\epsilon_p$ | $\epsilon_u$ | $\epsilon_v$ |
| $\alpha = 1$    | 0.084%       | 0.425%       | 0.309%       | 0.211%       | 1.071%       |
| $\alpha = 100$  | 0.076%       | 0.412%       | 0.516%       | 0.131%       | 0.368%       |
| 动态，方程（6） | 0.072%       | 0.352%       | 0.212%       | 0.056%       | 0.436%       |
| 动态，方程（7） | 0.026%       | 0.199%       | 0.141%       | 0.067%       | 0.446%       |

### 3.2 二维圆柱尾流

这里我们使用NSFnets模拟 $Re = 100$ 时圆柱后二维涡脱落。圆柱放置在 $(x, y) = (0, 0)$，直径 $D = 1$。来自[15]的高保真DNS数据用作参考，并为NSFnet训练提供边界和初始数据。我们考虑的域为 $[1, 8] \times [-2, 2]$，时间区间为 $[0, 7]$（大约一个脱落周期），时间步长 $\Delta t = 0.1$。对于训练数据，我们在 $x$ 方向边界放置100个点，在 $y$ 方向边界放置50个点以强制执行边界条件，并使用域内140,000个时空散点计算残差。NSFnets包含10个隐藏层，每层100个神经元。除了默认模型 $\alpha = \beta = 1$ 和 $\alpha = \beta = 100$，我们再次实现了NSFnets的动态权重策略。训练过程与Kovasznay流使用的相同。

在 $t = 4.0$ 时的涡量等高线快照如图5所示，显示了NSFnet推断与DNS结果的定性一致。图6展示了VP-和VV-NSFnets的动态权重。在这种情况下，权重均初始化为1。权重的变化对应于学习率的变化。我们观察到 $\alpha$ 和 $\beta$ 均在10的量级，初始条件的权重大于边界条件的权重。训练期间加权损失函数的分离项如图7所示。我们采用两步训练确保所有NSFnets的收敛。NSFnet模拟的相对 $L_2$ 误差随时间变化如图8所示。我们看到，VV-NSFnet比VP-NSFnet表现更好，应用动态权重可以提高两种形式的模拟准确性。

### 3.3 三维Beltrami流

由Ethier和Steinman[32]开发的非稳态三维Beltrami流的解析解为：

$$
\begin{aligned}
u(x, y, z, t) &= -a \left[ e^{ax} \sin(ay + dz) + e^{az} \cos(ax + dy) \right] e^{-d^2 t}, \\
v(x, y, z, t) &= -a \left[ e^{ay} \sin(az + dx) + e^{ax} \cos(ay + dz) \right] e^{-d^2 t}, \\
w(x, y, z, t) &= -a \left[ e^{az} \sin(ax + dy) + e^{ay} \cos(az + dx) \right] e^{-d^2 t}, \\
p(x, y, z, t) &= -\frac{1}{2} a^2 \left[ e^{2ax} + e^{2ay} + e^{2az} + 2 \sin(ax + dy) \cos(az + dx) e^{a(y+z)} \right. \\
&\quad \left. + 2 \sin(ay + dz) \cos(ax + dy) e^{a(z+x)} + 2 \sin(az + dx) \cos(ay + dz) e^{a(x+y)} \right] e^{-2d^2 t},
\end{aligned} \tag{11}
$$

其中，$a = d = 1$。在NSFnet模拟中，计算域定义为 $[-1, 1] \times [-1, 1] \times [-1, 1]$，时间区间为 $[0, 1]$；时间步长为0.1。对于NSFnet训练数据，每个面上使用 $31 \times 31$ 个点用于边界条件，时空域内使用10,000个点用于方程。损失函数的权重系数固定为 $\alpha = \beta = 100$。采用两步优化（Adam和L-BFGS-B）训练神经网络，默认架构为10层，每层100个神经元。在 $t = 1$ 和 $z = 0$ 切面上的速度场快照如图9所示。两种NSFnets在不同时间步的模拟结果误差如表3所示，其中给出了三个速度分量的相对 $L_2$ 误差。如表所示，两种NSFnets均能获得Beltrami流的Navier-Stokes方程的准确解，但VV-NSFnet优于VP-NSFnet。

**表3**：Beltrami流：VP-NSFnet和VV-NSFnet的相对 $L_2$ 误差。

| t    | VP-NSFnet    |              |              |              | VV-NSFnet    |              |              |
| ---- | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
|      | $\epsilon_u$ | $\epsilon_v$ | $\epsilon_w$ | $\epsilon_p$ | $\epsilon_u$ | $\epsilon_v$ | $\epsilon_w$ |
| 0    | 0.067%       | 0.059%       | 0.061%       | 0.700%       | 0.069%       | 0.067%       | 0.066%       |
| 0.25 | 0.158%       | 0.132%       | 0.140%       | 0.778%       | 0.109%       | 0.094%       | 0.108%       |
| 0.50 | 0.221%       | 0.189%       | 0.233%       | 1.292%       | 0.118%       | 0.119%       | 0.132%       |
| 0.75 | 0.287%       | 0.217%       | 0.406%       | 2.149%       | 0.156%       | 0.154%       | 0.187%       |
| 1.00 | 0.426%       | 0.366%       | 0.587%       | 4.766%       | 0.255%       | 0.284%       | 0.263%       |

---

{

### 3. 层流模拟（通俗解释）

在这一节里，我们用提出的 **NSFnets** 来模拟几种典型的不可压缩流动：

1. **二维稳态 Kovasznay 流**
2. **二维非稳态圆柱尾流**
3. **三维非稳态 Beltrami 流**

我们比较了 **VP-NSFnet** 和 **VV-NSFnet** 的性能，并研究了 **动态权重策略** 是否能提高解的准确性。

为了评价模拟效果，我们采用了每个时间步的 **相对 $L_2$ 误差**：
$$
\epsilon_V = \frac{\|\hat{V} - V\|_2}{\|V\|_2},
$$
其中：

- $V$ 是真实解的速度 $(u,v,w)$ 或压力 $p$；
- $\hat{V}$ 是 NSFnet 推断结果。

另外，在对比时，我们会把 NSFnet 解和 DNS 解的 **压力均值对齐**，这样误差评价更合理。

------

#### 3.1 Kovasznay 流

Kovasznay 流是一个二维稳态的 Navier–Stokes 方程解析解，常用来测试方法精度。其解析解为：
$$
u(x,y) = 1 - e^{\lambda x}\cos(2\pi y),
$$
其中
$$
\lambda = \frac{1}{2\nu} - \sqrt{\frac{1}{4\nu^2} + 4\pi^2}, 
\quad \nu = \frac{1}{Re} = \tfrac{1}{40}.
$$

- 计算域：$[-0.5, 1.0]\times[-0.5, 1.5]$
- 边界点数：$N_b = 400$
- 方程残差点：2601 个

训练方式：

- Adam 优化（学习率从 $10^{-3}$ 到 $10^{-6}$，逐步递减），迭代 $5k\sim 50k$ 次
- 再用 L-BFGS-B 微调

#### 结果观察

- **网络越大，解越准**；小网络时 VP-NSFnet 更好，大网络时 VV-NSFnet 更优。
- 边界约束权重 $\alpha$ 的选择很关键：
  - $\alpha=1$ 或 $\alpha=100$ 时，结果还行；
  - 用 **动态权重** 时最优。

------

**表1**：不同规模网络的 $L_2$ 误差（$\alpha=100$）

| NN大小   | VP-$\epsilon_u$ | VP-$\epsilon_v$ | VP-$\epsilon_p$ | VV-$\epsilon_u$ | VV-$\epsilon_v$ |
| -------- | --------------- | --------------- | --------------- | --------------- | --------------- |
| 4 × 50   | 0.076%          | 0.412%          | 0.516%          | 0.131%          | 0.368%          |
| 7 × 50   | 0.038%          | 0.255%          | 0.114%          | 0.111%          | 0.520%          |
| 7 × 100  | 0.016%          | 0.183%          | 0.062%          | 0.078%          | 0.397%          |
| 10 × 100 | 0.020%          | 0.115%          | 0.044%          | 0.040%          | 0.233%          |
| 10 × 200 | 0.012%          | 0.103%          | 0.042%          | 0.022%          | 0.121%          |
| 10 × 250 | 0.011%          | 0.101%          | 0.031%          | 0.011%          | 0.081%          |
| 10 × 300 | 0.008%          | 0.072%          | 0.041%          | 0.004%          | 0.037%          |

**表2**：不同 $\alpha$ 权重下的误差（NN大小 4×50）

| 权重         | VP-$\epsilon_u$ | VP-$\epsilon_v$ | VP-$\epsilon_p$ | VV-$\epsilon_u$ | VV-$\epsilon_v$ |
| ------------ | --------------- | --------------- | --------------- | --------------- | --------------- |
| $\alpha=1$   | 0.084%          | 0.425%          | 0.309%          | 0.211%          | 1.071%          |
| $\alpha=100$ | 0.076%          | 0.412%          | 0.516%          | 0.131%          | 0.368%          |
| 动态(式6)    | 0.072%          | 0.352%          | 0.212%          | 0.056%          | 0.436%          |
| 动态(式7)    | 0.026%          | 0.199%          | 0.141%          | 0.067%          | 0.446%          |

结论：**动态权重最优**，能有效平衡梯度，让训练更稳定。

------

#### 3.2 二维圆柱尾流

这里模拟 $Re=100$ 的圆柱尾流涡街。

- 圆柱直径 $D=1$，放在原点
- 计算域：$[1,8]\times[-2,2]$
- 时间：$[0,7]$（约一个脱落周期），步长 $\Delta t=0.1$

训练数据：

- 边界点：$100\times 50$
- 内部残差点：140,000 个
- 网络规模：10层，每层100神经元

结果：

- 涡量等高线（在 $t=4$）与 DNS 对齐
- 动态权重下 $\alpha,\beta$ 最终都在 10 左右
- 初始条件的权重 > 边界条件的权重
- VV-NSFnet 比 VP-NSFnet 准确
- 动态权重进一步提升效果

------

#### 3.3 三维 Beltrami 流

Beltrami 流是一个三维非稳态解析解，公式为：
$$
u(x,y,z,t) = -a\Big[ e^{ax}\sin(ay+dz) + e^{az}\cos(ax+dy) \Big] e^{-d^2t}
$$
其中 $a=d=1$。

- 计算域：$[-1,1]^3$
- 时间：$[0,1]$，步长 0.1
- 数据点：每个面 $31\times 31$，内部1万个
- 权重：$\alpha=\beta=100$
- 网络：10层 × 100 神经元

#### 结果（表3）

| t    | VP-$\epsilon_u$ | VP-$\epsilon_v$ | VP-$\epsilon_w$ | VP-$\epsilon_p$ | VV-$\epsilon_u$ | VV-$\epsilon_v$ | VV-$\epsilon_w$ |
| ---- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
| 0    | 0.067%          | 0.059%          | 0.061%          | 0.700%          | 0.069%          | 0.067%          | 0.066%          |
| 0.25 | 0.158%          | 0.132%          | 0.140%          | 0.778%          | 0.109%          | 0.094%          | 0.108%          |
| 0.50 | 0.221%          | 0.189%          | 0.233%          | 1.292%          | 0.118%          | 0.119%          | 0.132%          |
| 0.75 | 0.287%          | 0.217%          | 0.406%          | 2.149%          | 0.156%          | 0.154%          | 0.187%          |
| 1.00 | 0.426%          | 0.366%          | 0.587%          | 4.766%          | 0.255%          | 0.284%          | 0.263%          |

结论：两种 NSFnet 都能得到高精度解，但 **VV-NSFnet 始终比 VP-NSFnet 稍优**。

------

✅ 总体结论：

- 小网络：VP 形式更稳健
- 大网络 & 复杂问题：VV 形式更优
- 动态权重能大幅提升精度和稳定性

}

## 4. 湍流通道流模拟

### 4.1 问题设置

我们使用VP-NSFnets系统地模拟 $Re_\tau = 9.9935 \times 10^2$ 的湍流通道流。我们使用http://turbulence.pha.jhu.edu的湍流通道流数据库[20,21,22]作为参考DNS解。数据库提供参考数据以及VP-NSFnet的一些初始或边界条件。数据库中通道流的DNS域为 $[0, 8\pi] \times [-1, 1] \times [0, 3\pi]$；平均压力梯度为 $dP/dx = 0.0025$。DNS的非量纲时间步长为0.0013，而在线数据库的时间步长为0.0065（DNS的五倍）。因此，NSFnets评估残差也使用0.0065的时间步长。我们通过考虑通道中不同位置、不同大小的子域进行NSFnet模拟。在第一个例子中，我们放置一个壁单位约200的盒子，覆盖较长时间段。然后，我们测试覆盖半个通道高度的较大域的NSFnet模拟。最后，我们检查超参数对NSFnet模拟准确性的影响。本研究中我们使用小批量训练NSFnets。输入数据分为三部分，分别对应初始条件、边界条件和方程残差。因此，我们指定一个训练周期的总迭代次数 $n_{it}$，每部分的数据平均分为 $n_{it}$ 个小批量。整个小批量的数据包括每个小批量的数据。

### 4.2 长时间区间模拟结果

我们首先研究VP-NSFnet是否能维持湍流，因此我们进行了覆盖较长时间区间的模拟。在这个测试中，考虑的子域为 $[12.47, 12.66] \times [-0.90, -0.70] \times [4.61, 4.82]$（壁单位为 $190 \times 200 \times 210$）。我们进行了两个不同模拟，覆盖非量纲时间域 $[0, 0.52]$（81个时间步，壁单位25.97）和 $[0, 0.832]$（129个时间步，壁单位41.55）。这里，我们定义模拟区域的局部对流时间单位为 $T^+_c = L^+_x / U(y)_{min} = 12.0$。（$L^+_x$ 是域在流向的尺寸。）因此，25.97和41.55分别覆盖了超过两个对流时间单位，即 $2.2T^+_c$ 和 $3.5T^+_c$。我们使用域内20,000个点、每个时间步采样的6,644个边界点以及初始时间步的33,524个点来计算损失函数。我们设置一个训练周期的总迭代次数 $n_{it} = 150$。VP-NSFnet有10个隐藏层，每层300个神经元。Adam的初始学习率在训练过程中从 $10^{-3}$（1000个训练周期）衰减到 $10^{-4}$（4000个训练周期）、$10^{-5}$（1000个训练周期）和 $10^{-6}$（500个训练周期）。方程（2a）中的权重为 $\alpha = 100$，$\beta = 100$。参考DNS和VP-NSFnet在 $t^+ = 24.67$ 时的瞬时流场比较如图10所示。损失函数的收敛如图11所示。VP-NSFnet解的准确性比较如图12所示。所有速度分量的模拟误差均小于10%，但压力的相对 $L_2$ 误差可达15%或19%。总体而言，获得了良好的VP-NSFnet模拟准确性。这些结果表明，VP-NSFnet可以长时间维持湍流。

### 4.3 大域模拟结果

在这个测试中，我们考虑覆盖半个通道高度的较大域。VP-NSFnet模拟域为 $[12.47, 12.66] \times [-1, -0.0031] \times [4.61, 4.82]$（壁单位约为 $190 \times 997 \times 210$）；非量纲时间域为 $[0, 0.104]$（17个时间步，壁单位5.19）。我们放置域内100,000个点、每个时间步采样的26,048个边界点和初始时间步的147,968个点来确定损失函数。一个训练周期的总迭代次数 $n_{it}$ 为150。VP-NSFnet有10个隐藏层，每层300个神经元。Adam的初始学习率从 $10^{-3}$（250个训练周期）衰减到 $10^{-4}$（4250个训练周期）、$10^{-5}$（500个训练周期）和 $10^{-6}$（500个训练周期）。方程（2a）中的权重为 $\alpha = 100$，$\beta = 100$。参考DNS和VP-NSFnet在最后模拟时间步的瞬时流场比较如图13所示。VP-NSFnet解的收敛和准确性如图14所示。所有速度分量的模拟误差均小于10%，但压力的相对 $L_2$ 误差可达17%。在如此大的域中，不同尺度涡流的复杂相互作用发生，该域覆盖了包括壁面定律、粘性子层、缓冲层、对数律区和外层的整个范围[33]。然而，VP-NSFnet仍能获得非常准确的解。结果还表明，壁法向和展向速度的相对 $L_2$ 误差远高于流向速度，即几乎高一个量级。这是由于流向速度的幅度比其他两个速度分量大近一个量级，如图14所示。适当的归一化和各向异性权重的仔细调整可能导致更平衡的准确性。

### 4.4 权重的研究

在上述湍流通道流的数值实验中，所有权重均通过手动调整以获得满意的结果。在本节中，我们研究权重，特别是动态权重，对VP-NSFnet模拟准确性的影响。方程（6）的形式应用于VP-NSFnet的损失函数。然而，与方程（6）不同，本节考虑了归一化因子 $\gamma$。因此，湍流模拟的动态权重可以表示为：

$$
\hat{\alpha}^{(k+1)} = \frac{\max_\theta \{|\nabla_\theta L_e|\}}{\gamma |\nabla_\theta \alpha^{(k)} L_b|}. \tag{12}
$$

我们考虑的VP-NSFnet模拟域为 $[12.53, 12.59] \times [-1, -0.9762] \times [4.69, 4.75]$（壁单位约为 $60 \times 24 \times 60$）；非量纲时间域为 $[0, 0.104]$（17个时间步，壁单位5.19）。对于这个小域，损失函数中不使用初始速度值，即 $\beta = 0$，但我们可以学习它们。域内有2000个点，边界上每个时间步采样1100个点来确定损失函数。一个训练周期的总迭代次数 $n_{it}$ 为10。VP-NSFnet有5个隐藏层，每层200个神经元。在所有例子中，Adam的初始学习率从 $10^{-3}$（5000个训练周期）衰减到 $10^{-4}$（5000个训练周期）、$10^{-5}$（25000个训练周期）和 $5 \times 10^{-6}$（25000个训练周期）。我们设置了五种不同的策略来为边界数据选择不同的权重。在前两种策略中，我们使用固定权重 $\alpha = 1$ 和 $\alpha = 100$。然后，我们使用方程（12）给出的动态权重，归一化因子分别为 $\gamma = 1$、$\gamma = 5$ 和 $\gamma = 10$。参考DNS和VP-NSFnet在 $t^+ = 5.19$ 和 $x^+ = -12.27$ 的瞬时流场（$z-y$ 平面）比较如图15所示。训练过程中动态权重的演变如图16所示。不同超参数的VP-NSFnet模拟收敛如图17所示。不同超参数的VP-NSFnet模拟准确性如图18所示。从图15可以看出，参考解与固定权重 $\alpha = 1$ 的VP-NSFnet解存在较大差异，因此其准确性较低。因此，固定权重 $\alpha = 1$ 的VP-NSFnet模拟准确性未在图18中显示。总体而言，应用动态权重 $\gamma = 5$ 时获得最佳湍流模拟结果。动态权重的变化可以提高VP-NSFnet的性能。

{

### 湍流通道流模拟（通俗解释）

这一节研究了 **VP-NSFnet** 在高雷诺数湍流通道流 ($Re_\tau \approx 1000$) 下的表现。参考数据来自 **JHU 湍流数据库**（DNS 高精度模拟结果）。

DNS 的计算域为：
$$
[0,8\pi] \times [-1,1] \times [0,3\pi],
$$
其中流向是 $x$，法向是 $y$，展向是 $z$。平均压力梯度为
$$
\frac{dP}{dx} = 0.0025.
$$
DNS 的时间步长为 $0.0013$，数据库提供的为 $0.0065$（五倍大）。所以 NSFnet 训练也用 $0.0065$ 的步长。

------

#### 4.1 问题设置

- 训练时，数据分为三类：初始条件、边界条件、方程残差。
- 使用小批量训练（mini-batch）。
- 网络输入：$(t,x,y,z)$
- 输出：$(u,v,w,p)$
- 实验设置：在不同子域和不同时间区间下测试 NSFnet 是否能稳定维持湍流。

------

#### 4.2 长时间区间模拟结果

首先测试 **小域 + 长时间**。

- 子域范围：
  $$
  [12.47,12.66] \times [-0.90,-0.70] \times [4.61,4.82]
  $$
  （壁单位约 $190\times 200 \times 210$）

- 时间区间：

  - $[0,0.52]$（81个步长，对应 $25.97$ 壁单位）
  - $[0,0.832]$（129个步长，对应 $41.55$ 壁单位）

- 对流时间单位定义：
  $$
  T_c^+ = \frac{L_x^+}{U(y)_{min}} = 12.0,
  $$
  所以这两个区间分别覆盖 $2.2T_c^+$ 和 $3.5T_c^+$。

- 数据点：

  - 域内 20,000 个点
  - 每步边界点 6644
  - 初始点 33,524

- 网络：10层 × 300神经元

- 学习率逐步下降：$10^{-3} \to 10^{-6}$

- 权重：$\alpha=\beta=100$

#### 结果

- 速度的相对误差 < 10%
- 压力误差可达 15–19%
- 总体准确性不错，**湍流可以维持较长时间**。

------

#### 4.3 大域模拟结果

接着测试 **大域 + 短时间**。

- 子域范围：
  $$
  [12.47,12.66] \times [-1,-0.0031] \times [4.61,4.82]
  $$
  （壁单位约 $190\times 997 \times 210$）

- 时间区间：$[0,0.104]$（17个时间步，壁单位 5.19）

- 数据点：

  - 域内 100,000
  - 每步边界点 26,048
  - 初始点 147,968

- 网络：10层 × 300神经元

- 学习率：$10^{-3}\to 10^{-6}$

- 权重：$\alpha=\beta=100$

#### 结果

- 速度误差 < 10%
- 压力误差最高 17%
- 这种大域包含了通道的 **全层结构**：粘性子层、缓冲层、对数律区、外层。
- VP-NSFnet 仍然能给出很准确的结果。
- **壁法向速度 $v$ 和展向速度 $w$ 的误差远大于流向速度 $u$**。这是因为 $u$ 的量级大约比 $v,w$ 高一个量级。
- 提示：可以通过 **归一化** 或 **各向异性权重** 来平衡不同分量的误差。

------

#### 4.4 权重的研究

最后，研究了 **固定权重 vs 动态权重** 的影响。

动态权重的改进公式（加了归一化因子 $\gamma$）为：
$$
\hat{\alpha}^{(k+1)} = \frac{\max_\theta |\nabla_\theta L_e|}{\gamma \, |\nabla_\theta \alpha^{(k)} L_b|}.
$$
实验设置：

- 子域：
  $$
  [12.53,12.59] \times [-1,-0.9762] \times [4.69,4.75]
  $$
  （壁单位约 $60\times 24\times 60$）

- 时间区间：$[0,0.104]$（17个时间步，壁单位 5.19）

- 无初始条件（$\beta=0$），但网络可以学习

- 数据点：域内2000，边界每步1100

- 网络：5层 × 200神经元

- 学习率：$10^{-3}\to 5\times 10^{-6}$

- 权重策略：

  1. 固定 $\alpha=1$
  2. 固定 $\alpha=100$
  3. 动态权重，$\gamma=1,5,10$

#### 结果

- 固定 $\alpha=1$ → 差异很大，精度很差。
- 固定 $\alpha=100$ → 还行。
- **动态权重（$\gamma=5$）最好**，能显著提升湍流模拟效果。

------

#### 总结

1. **小域 + 长时间**：VP-NSFnet 能维持湍流，速度精度好，但压力稍差。
2. **大域 + 短时间**：VP-NSFnet 仍然准确，但不同速度分量误差差异较大。
3. **权重选择**：动态权重比固定权重更有效，特别是 $\gamma=5$ 最优。

}

---

## 5. 总结

在本研究中，我们探索了PINNs直接模拟从层流到湍流通道流的不可压缩流动的有效性。我们基于Navier-Stokes方程的两种不同形式制定了NSFnets：速度-压力（VP）形式和涡量-速度（VV）形式。空间和时间坐标是PINNs的输入，瞬时速度和压力场是VP-NSFnet的输出；类似地，瞬时速度和涡量场是VV-NSFnet的输出。我们使用自动微分表示Navier-Stokes方程中的所有微分算子；然后，方程可以通过神经网络表示。我们将初始和边界条件视为监督数据驱动部分，将Navier-Stokes方程的残差视为PINNs损失函数中的无监督物理信息部分。我们注意到，VP-NSFnet无需为压力提供边界或初始条件数据，压力是隐状态，通过不可压缩性约束间接获得，而无需拆分方程。NSFnets的收敛通过总损失函数以及各个损失函数进行监控。我们模拟了几个层流，包括二维稳态Kovasznay流、二维圆柱尾流和三维Beltrami流，使用两种形式的NSFnets。我们还研究了损失函数中各组成部分权重的影响。我们发现，对于层流情况，VV-NSFnet比VP-NSFnet实现更高的准确性，动态权重的变化可以提高两种NSFnets的性能。

此外，我们探索了使用NSFnets模拟 $Re_\tau \sim 1,000$ 的湍流通道流的可能性。我们通过考虑通道中不同位置、不同大小的子域以及不同时间间隔进行NSFnet模拟。已建立的DNS数据库为NSFnet模拟提供了适当的初始和边界条件。在损失函数收敛后，DNS结果与VP-NSFnet模拟结果之间获得了良好的一致性。长时间段模拟表明，NSFnets可以维持湍流，误差保持在合理水平。我们还研究了动态权重与固定权重的使用，并展示了动态权重如何进一步提高VP-NSFnet的准确性。与VP形式不同，尝试训练VV-NSFnet未能提供满意的控制方程损失函数收敛。对于 $\alpha = 50,000$，我们获得了合理的准确性，边界条件的损失函数收敛到较小的值，但控制方程的残差仍然很大。这可能与数据来源于基于VP形式的DNS数据库有关，因此推导的边界条件可能不那么准确，而是与VV形式的控制方程不一致。我们计划在未来工作中重新审视这个问题。

使用NSFnets建模湍流的当前研究是评估PINN性能的首次尝试，虽然初步结果令人鼓舞，但更广泛的问题是，PINNs能否提供足够的准确性，以在非随机边界条件下维持湍流，而是使用光谱类型湍流模拟中整个域的简单零Dirichlet和周期性边界条件。为了迅速解决这个问题，PINNs的效率必须显著提高，包括开发多节点GPU代码，以显著加速训练过程。使用自适应激活函数可以进一步增强这种加速，如[30,31]所示。此外，需要进一步研究为三个不同速度分量推导适当的归一化程序，以便获得均匀的准确性，因为在当前研究中，流向分量的推断准确性比横向速度分量高一个量级，而流向速度比横向速度高一个量级。

---

## 致谢

金晓伟和李辉感谢国家自然科学基金（NSFC，资助号U1711265）的资助。蔡胜泽和G.E. Karniadakis感谢DARPA-AIRA资助（HR00111990025）和DOE资助（DE-AC05-76RL01830）的支持。

---

## 参考文献

[1] J. Ling, A. Kurzawski, J. Templeton, Reynolds averaged turbulence modelling using deep neural networks with embedded invariance, *Journal of Fluid Mechanics* 807 (2016) 155–166.  
[2] J.-X. Wang, J.-L. Wu, H. Xiao, Physics-informed machine learning approach for reconstructing Reynolds stress modeling discrepancies based on DNS data, *Physical Review Fluids* 2 (2017) 034603.  
[3] C. Jiang, J. Mi, S. Laima, H. Li, A novel algebraic stress model with machine-learning-assisted parameterization, *Energies* 13 (2020) 258.  
[4] Z. Zhou, G. He, S. Wang, G. Jin, Subgrid-scale model for large-eddy simulation of isotropic turbulent flows using an artificial neural network, *Computers & Fluids* 195 (2019) 104319.  
[5] X. Jin, P. Cheng, W.-L. Chen, H. Li, Prediction model of velocity field around circular cylinder over various Reynolds numbers by fusion convolutional neural networks based on pressure on the cylinder, *Physics of Fluids* 30 (2018) 047105.  
[6] P. Wu, J. Sun, X. Chang, W. Zhang, R. Arcucci, Y. Guo, C. C. Pain, Data-driven reduced order model with temporal convolutional neural network, *Computer Methods in Applied Mechanics and Engineering* 360 (2020) 112766.  
[7] X. Jin, S. Laima, W.-L. Chen, H. Li, Time-resolved reconstruction of flow field around a circular cylinder by recurrent neural networks based on non-time-resolved particle image velocimetry measurements, *Experiments in Fluids* (2020) accepted.  
[8] Z. Hosseini, R. J. Martinuzzi, B. R. Noack, Sensor-based estimation of the velocity in the wake of a low-aspect-ratio pyramid, *Experiments in Fluids* 56 (2015) 13.  
[9] S. Discetti, M. Raiola, A. Ianiro, Estimation of time-resolved turbulent fields through correlation of non-time-resolved field measurements and time-resolved point measurements, *Experimental Thermal and Fluid Science* 93 (2018) 119–130.  
[10] S. Cai, S. Zhou, C. Xu, Q. Gao, Dense motion estimation of particle images via a convolutional neural network, *Experiments in Fluids* 60 (2019) 73.  
[11] K. Duraisamy, G. Iaccarino, H. Xiao, Turbulence modeling in the age of data, *Annual Review of Fluid Mechanics* 51 (2019) 357–377.  
[12] S. L. Brunton, B. R. Noack, P. Koumoutsakos, Machine learning for fluid mechanics, *Annual Review of Fluid Mechanics* 52 (2020) 477–508.  
[13] M. Raissi, P. Perdikaris, G. E. Karniadakis, Physics informed deep learning (part i): Data-driven solutions of nonlinear partial differential equations, *arXiv preprint arXiv:1711.10561* (2017).  
[14] M. Raissi, P. Perdikaris, G. E. Karniadakis, Physics informed deep learning (part ii): Data-driven discovery of nonlinear partial differential equations, *arXiv preprint arXiv:1711.10561* (2017).  
[15] M. Raissi, P. Perdikaris, G. E. Karniadakis, Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations, *Journal of Computational Physics* 378 (2019) 686–707.  
[16] M. Raissi, Z. Wang, M. S. Triantafyllou, G. E. Karniadakis, Deep learning of vortex-induced vibrations, *Journal of Fluid Mechanics* 861 (2019) 119–137.  
[17] M. Raissi, A. Yazdani, G. E. Karniadakis, Hidden fluid mechanics: Learning velocity and pressure fields from flow visualizations, *Science* 367 (2020) 1026–1030.  
[18] J. Kim, P. Moin, R. Moser, Turbulence statistics in fully developed channel flow at low Reynolds number, *Journal of Fluid Mechanics* 177 (1987) 133–166.  
[19] G. E. Karniadakis, S. Sherwin, *Spectral/hp Element Methods for Computational Fluid Dynamics*, Oxford University Press, 2013.  
[20] E. Perlman, R. Burns, Y. Li, C. Meneveau, Data exploration of turbulence simulations using a database cluster, in: *Proceedings of the 2007 ACM/IEEE conference on Supercomputing*, ACM, 2007, p. 23.  
[21] Y. Li, E. Perlman, M. Wan, Y. Yang, C. Meneveau, R. Burns, S. Chen, A. Szalay, G. Eyink, A public turbulence database cluster and applications to study Lagrangian evolution of velocity increments in turbulence, *Journal of Turbulence* 9 (2008) N31.  
[22] J. Graham, K. Kanov, X. Yang, M. Lee, N. Malaya, C. Lalescu, R. Burns, G. Eyink, A. Szalay, R. Moser, et al., A web services accessible database of turbulent channel flow and its use for testing a new integral wall model for les, *Journal of Turbulence* 17 (2016) 181–215.  
[23] A. G. Baydin, B. A. Pearlmutter, A. A. Radul, J. M. Siskind, Automatic differentiation in machine learning: a survey, *Journal of Machine Learning Research* 18 (2018).  
[24] G. E. Karniadakis, M. Israeli, S. A. Orszag, High-order splitting methods for the incompressible Navier-Stokes equations, *Journal of Computational Physics* 97 (1991) 414–443.  
[25] S. Wang, Y. Teng, P. Perdikaris, Understanding and mitigating gradient pathologies in physics-informed neural networks, *arXiv preprint arXiv:2001.04536* (2020).  
[26] D. P. Kingma, J. Ba, Adam: A method for stochastic optimization, *arXiv preprint arXiv:1412.6980* (2014).  
[27] X. Glorot, Y. Bengio, Understanding the difficulty of training deep feedforward neural networks, in: *Proceedings of the thirteenth international conference on artificial intelligence and statistics*, 2010, pp. 249–256.  
[28] J. Trujillo, G. E. Karniadakis, A penalty method for the vorticity-velocity formulation, *Journal of Computational Physics* 149 (1999) 32–58.  
[29] H. L. Meitz, H. F. Fasel, A compact-difference scheme for the Navier–Stokes equations in vorticity-velocity formulation, *Journal of Computational Physics* 157 (2000) 371–403.  
[30] A. D. Jagtap, K. Kawaguchi, G. E. Karniadakis, Adaptive activation functions accelerate convergence in deep and physics-informed neural networks, *Journal of Computational Physics* 404 (2020) 109136.  
[31] A. D. Jagtap, K. Kawaguchi, G. E. Karniadakis, Locally adaptive activation functions with slope recovery term for deep and physics-informed neural networks, *arXiv preprint arXiv:1909.12228* (2019).  
[32] C. R. Ethier, D. Steinman, Exact fully 3D Navier-Stokes solutions for benchmarking, *International Journal for Numerical Methods in Fluids* 19 (1994) 369–375.  
[33] S. B. Pope, *Turbulent Flows*, Cambridge University Press, Cambridge, (2000) 276.

---

**图表说明**

**图1**：NSFnets示意图：(a) 速度-压力（VP）形式；(b) 涡量-速度（VV）形式。神经网络的左部分是无信息网络，右部分使用自动微分实现VP和VV形式。我们仅展示了右部分的算子，因为由VP和VV微分算子诱导的神经网络过于复杂，即使使用TensorFlow中的“TensorBoard”等专门方法也无法可视化。

**图2**：Kovasznay流：动态权重：(a) VP-NSFnet，方程（6）；(b) VV-NSFnet，方程（6）；(c) VP-NSFnet，方程（7）；(d) VV-NSFnet，方程（7）。NN大小为 4 × 50。

**图3**：Kovasznay流：损失函数（物理损失 $L_e$ 和边界损失 $L_b$）：(a) VP-NSFnet，固定权重 $\alpha = 1$；(b) VP-NSFnet，固定权重 $\alpha = 100$；(c) VP-NSFnet，动态权重；(d) VV-NSFnet，固定权重 $\alpha = 1$；(e) VV-NSFnet，固定权重 $\alpha = 100$；(f) VV-NSFnet，动态权重。“DW1”表示方程（6）给出的动态权重，“DW2”表示方程（7）给出的动态权重。垂直虚线绿色线之前使用Adam优化器，之后使用L-BFGS-B优化器。NN大小为 4 × 50。

**图4**：Kovasznay流：训练NSFnets期间第10000次迭代后梯度（$\nabla_\theta L_e$ 和 $\nabla_\theta (\alpha L_b)$）的直方图：(a) VP-NSFnet，固定权重 $\alpha = 1$；(b) VP-NSFnet，固定权重 $\alpha = 100$；(c) VP-NSFnet，动态权重（方程（6））；(d) VP-NSFnet，动态权重（方程（7））；(e) VV-NSFnet，固定权重 $\alpha = 1$；(f) VV-NSFnet，固定权重 $\alpha = 100$；(g) VV-NSFnet，动态权重（方程（6））；(h) VV-NSFnet，动态权重（方程（7））。

**图5**：圆柱流动：涡量等高线在 $t = 4.0$ 时的同一等高水平：(a) 参考DNS解，来自[15]；(b) VP-NSFnet，固定权重 $\alpha = \beta = 1$；(c) VP-NSFnet，固定权重 $\alpha = \beta = 100$；(d) VP-NSFnet，动态权重；(e) VV-NSFnet，固定权重 $\alpha = \beta = 1$；(f) VV-NSFnet，固定权重 $\alpha = \beta = 100$；(g) VV-NSFnet，动态权重。此处动态权重由方程（6）给出。

**图6**：圆柱流动：动态权重由方程（6）给出：(a) VP-NSFnet；(b) VV-NSFnet。

**图7**：圆柱流动：损失函数（物理损失 $L_e$ 和边界损失 $L_b$）：(a) VP-NSFnet，固定权重 $\alpha = \beta = 1$；(b) VP-NSFnet，固定权重 $\alpha = \beta = 100$；(c) VP-NSFnet，动态权重，方程（6）；(d) VV-NSFnet，固定权重 $\alpha = \beta = 1$；(e) VV-NSFnet，固定权重 $\alpha = \beta = 100$；(f) VV-NSFnet，动态权重，方程（6）。虚线绿色线之前使用Adam优化器，之后使用L-BFGS-B优化器。NN大小为 10 × 100。

**图8**：圆柱流动：NSFnets模拟的相对 $L_2$ 误差：(a) 流向速度；(b) 横向速度；(c) 压力。此处动态权重由方程（6）给出。

**图9**：Beltrami流：在 $t = 1$ 和 $z = 0$ 平面上的速度场：(a) 解析解；(b) VP-NSFnet结果；(c) VV-NSFnet结果。

**图10**：长时间区间：参考DNS与VP-NSFnet在 $t^+ = 24.67$ 时的瞬时 $z-y$ 平面流场比较：(a) 参考解；(b) VP-NSFnet，模拟覆盖 $2.2T^+_c$；(c) VP-NSFnet，模拟覆盖 $3.5T^+_c$。

**图11**：长时间区间：VP-NSFnet模拟的收敛：(a) 总损失函数和边界损失函数的收敛；(b) 控制方程残差的收敛。

**图12**：长时间区间：VP-NSFnet模拟的准确性：实线和虚线分别表示覆盖 $2.2T^+_c$ 和 $3.5T^+_c$ 的模拟。

**图13**：大域：参考DNS与VP-NSFnet在 $t^+ = 5.19$ 覆盖半个通道高度的瞬时流场比较：(a) 参考解，$x-y$ 平面，$z^+ = 0$；(b) VP-NSFnet，$x-y$ 平面，$z^+ = 0$；(c) 参考解，$z-y$ 平面，$x^+ = 0$；(d) VP-NSFnet，$z-y$ 平面，$x^+ = 0$。

**图14**：大域：覆盖半个通道高度的VP-NSFnet的收敛和准确性：(a) 损失函数的收敛；(b) 相对 $L_2$ 误差。

**图15**：权重的影响：参考DNS与VP-NSFnet在 $t^+ = 5.19$ 和 $x^+ = -12.27$ 的瞬时流场（$z-y$ 平面）比较：(a) 参考解；(b) VP-NSFnet，固定权重 $\alpha = 1$；(c) VP-NSFnet，固定权重 $\alpha = 100$；(d) VP-NSFnet，动态权重，归一化因子 $\gamma = 1$；(e) VP-NSFnet，动态权重，归一化因子 $\gamma = 5$；(f) VP-NSFnet，动态权重，归一化因子 $\gamma = 10$。此处所有动态权重由方程（12）给出。

**图16**：权重的影响：不同归一化因子的动态权重与训练周期的关系。所有动态权重由方程（12）给出。

**图17**：权重的影响：VP-NSFnet模拟的收敛：(a) VP-NSFnet，固定权重 $\alpha = 1$；(b) VP-NSFnet，固定权重 $\alpha = 100$；(c) VP-NSFnet，动态权重，归一化因子 $\gamma = 1$；(d) VP-NSFnet，动态权重，归一化因子 $\gamma = 5$；(e) VP-NSFnet，动态权重，归一化因子 $\gamma = 10$。所有动态权重由方程（12）给出。

**图18**：权重的影响：不同权重的VP-NSFnet的准确性：(a) 到 (d) $u$、$v$、$w$ 和 $p$ 的相对 $L_2$ 误差。