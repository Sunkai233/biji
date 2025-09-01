# PhysicsSolver: 基于Transformer增强的物理信息神经网络用于偏微分方程的正向和预测问题

**作者：**

- 朱振毅 (zyzhu@math.cuhk.edu.hk)
- 黄宇晨 (ychuang@math.cuhk.edu.hk)
- 刘柳 (lliu@math.cuhk.edu.hk)

香港中文大学数学系

## 摘要

时变偏微分方程是描述各种物理现象随时间演化的一类重要方程。科学计算中的一个开放问题是预测解在给定时间域之外的行为。大多数传统数值方法应用于给定的时空区域，只能准确近似给定区域内的解。为了解决这个问题，许多基于深度学习的方法，包括数据驱动和无数据方法，已被开发用来解决这些问题。然而，大多数数据驱动方法需要大量数据，消耗大量计算资源，且未能利用偏微分方程(PDE)中嵌入的所有必要信息。此外，无数据方法（如物理信息神经网络(PINNs)）在实际应用中可能不太理想，因为传统的PINNs主要依赖多层感知器(MLP)和卷积神经网络(CNN)，往往忽略了真实物理系统中固有的关键时间依赖性。我们提出了一种名为PhysicsSolver的方法，融合了两种方法的优势：可以在不使用数据的情况下学习物理系统内在特性的无数据方法，以及擅长进行预测的数据驱动方法。大量数值实验证明了我们所提出方法的效率和鲁棒性。我们在 https://github.com/PhysicsSolver 提供代码。

（

为此，我们提出了一个新的方法，叫做 **PhysicsSolver**。它的思路是：

- 既能像 PINNs 一样，不依赖大量数据，而是利用物理规律；
- 又能像数据驱动方法一样，擅长做预测。

）

## 1 引言

求解偏微分方程对许多实际应用都至关重要，包括天气预报、工业设计和材料分析。求解PDE的数值方法，如有限元法[2]和伪谱法[6]，在科学和工程中得到了广泛研究，但它们往往面临高计算成本，特别是在处理复杂或高维PDE时。大多数情况下，这些传统数值方法主要设计用于获得给定区域内的解。因此，它们无法高精度地预测给定时间域之外的解。为了预测给定区域之外的解，它们通常依赖于外推方法[20]，如线性、多项式和圆锥外推等。

近年来，无数据机器学习技术，特别是PINNs，在通过提供能够补充传统方法的更快解决方案来转变科学计算方面显示出了前景[17]。然而，传统神经网络通常在固定的有限维空间内工作，这限制了它们在不同离散化中泛化解的能力。这一限制突出了开发网格不变神经网络的必要性。此外，这些方法通常是无数据的，这在建模某些复杂系统时可能使预测不如数据驱动方法准确。

此外，基于算子学习的数据驱动深度学习模型已经成为解决PDE的有效工具。这些模型利用其强大的非线性映射能力，通过利用提供的训练数据来学习PDE相关任务中输入和输出之间的关系。因此，它们在推理阶段能比传统数值方法提供快得多的解。最受欢迎的数据驱动模型之一是傅里叶神经算子(FNO)方法[12]，它可以基于低分辨率输入有效预测高分辨率解。然而，训练这些模型有效所需的大数据集仍然是一个重大挑战。此外，与直接集成此类信息的PINNs不同，这些方法没有充分利用PDE提供的物理信息。因此，它们可能无法捕获数据集背后的一些内在细节。

为了解决上述三种方法的不足，我们提出了一个基于transformer和PINNs混合的模型，称为PhysicsSolver。具体来说，我们首先进行数据工程，引入插值伪序列网格生成器模块，将点状时空输入转换为时间序列。然后，我们应用使用Halton序列的蒙特卡洛采样，这使我们能够采样相对稀疏但更具代表性的数据点，为后续训练提供更好的数据集。为了学习内在物理模式及其随时间的演化，我们提出了基于transformers的PhysicsSolver模块，结合了物理注意力模块。该模块可以编码输入数据集和网格点，然后通过解码器生成输出。在训练过程中，我们开发了一个鲁棒的算法，利用方程信息和稀疏训练数据同时优化模型。最后，我们在几个一维和高维PDE上进行了大量实验，PhysicsSolver实现了一致的最佳性能，具有显著的相对优势。

本工作的主要贡献总结如下：

1. 首先，我们制定了PDE中的一般预测问题，然后研究了不同模型预测未来物理状态的能力，这通常是数值分析中的一个具有挑战性的问题。
2. 超越先前的方法，我们提出了一个名为PhysicsSolver的模型，它融合了PINNs和transformer的优势，使我们的模型能够更好地学习物理状态的演化。我们发现，我们提出的PhysicsSolver比外推法和其他基于深度学习的方法能更准确地捕获内在物理现象的趋势，并能更好地预测给定时间域外的解。
3. 我们开发了一些数据工程模块，如插值伪序列网格生成器(IPSGG)和Halton序列生成(HSG)，以定制PhysicsSolver的学习过程。
4. PhysicsSolver在PDE正向和预测问题的所有基准问题上都实现了鲁棒和优越的性能。

本文的其余部分安排如下。第2节首先介绍时变PDE中正向问题、逆问题和预测问题的问题表述。第3节回顾了解决PDE的各种方法的相关工作。第4节提供了我们提出的PhysicsSolver方法的方法论和学习方案的详细信息。第5节展示了一系列数值实验来验证我们方法的有效性，其中调查了正向和预测问题。最后，结论和未来工作将在最后一节讨论。

（

我们还设计了一些 **数据工程模块** 来增强学习效果：

- **插值伪序列网格生成器 (IPSGG)**：把零散的时空点转成连续的时间序列输入；
- **Halton 序列采样 (HSG)**：用蒙特卡洛方式挑选更具代表性的数据点，让训练更高效。

最终，PhysicsSolver 用“物理注意力模块”把物理规律和数据模式结合起来，学习到更深层次的动态特征。在实验中，它在多个一维和高维 PDE 问题上都表现出了很强的预测能力，尤其在预测“超出已知时间范围的解”时，比传统数值方法和其他深度学习模型都更好。

）

## 2 问题表述

设$\mathcal{T}$为给定的时间空间，$\mathcal{D}$为$\mathbb{R}^N$中的有界开集(空间域)，边界$\gamma$为$(N-1)$维流形(边界空间)，$\Omega_{\mathbf{v}}$为速度空间。考虑以下具有空间输入$\mathbf{x}$、速度输入$\mathbf{v}$和时间输入$t$的PDE，符合以下抽象形式：

$$ \begin{cases} \mathcal{L}[u(t,\mathbf{x},\mathbf{v},\boldsymbol{\beta})] = f(t,\mathbf{x},\mathbf{v}), & \forall (t,\mathbf{x},\mathbf{v}) \in \mathcal{T} \times \mathcal{D} \times \Omega_{\mathbf{v}}, \ \mathcal{B}[u(t,\mathbf{x},\mathbf{v},\boldsymbol{\beta})] = g(t,\mathbf{x},\mathbf{v}), & \forall (t,\mathbf{x},\mathbf{v}) \in \mathcal{T} \times \gamma \times \Omega_{\mathbf{v}}, \ \mathcal{I}[u(0,\mathbf{x},\mathbf{v},\boldsymbol{\beta})] = h(\mathbf{x},\mathbf{v}), & \forall (\mathbf{x},\mathbf{v}) \in \mathcal{D} \times \Omega_{\mathbf{v}}. \end{cases} \tag{1} $$

其中$u$是PDE的解，$\boldsymbol{\beta} \in \mathcal{H}^d$是PDE的物理参数，$\mathcal{H}$是参数空间，$d$是PDE系统中物理参数的维度。$\mathcal{L}$是一个(可能非线性)微分算子，规范系统的行为，$\mathcal{B}$描述边界条件，$\mathcal{I}$通常描述初始条件。具体地，$(t,\mathbf{x},\mathbf{v}) \in \mathcal{T} \times \mathcal{D} \times \Omega_{\mathbf{v}}$是残差点，$(t,\mathbf{x},\mathbf{v}) \in \mathcal{T} \times \gamma \times \Omega_{\mathbf{v}}$是边界点，$(0,\mathbf{x},\mathbf{v}) \in \mathcal{T} \times \gamma \times \Omega_{\mathbf{v}}$是初始点。$f,g,h$分别是源项、(可能混合的)边界值和初始值。

PDE数值分析中主要有两个问题：正向问题和逆问题。预测问题也是一个重要问题，但尚未得到广泛调查或制定。接下来我们简要介绍它们。

**正向问题**：在正向问题中，我们希望使用传统数值格式（如有限差分法）或使用深度神经网络表示的近似解$u_{NN}$在$n$个给定网格${t_i,\mathbf{x}_i,\mathbf{v}*i}*{i=0}^n$上近似真实解$u$。

**逆问题**：在逆问题[1]中，我们需要推断PDE的未知物理参数$\boldsymbol{\beta}$。与正向问题不同，我们需要另外使用合成数据集${u(t_j,\mathbf{x}*j,\mathbf{v}\*j)}\*{j=0}^{n*{data}}$，其中$n_{data}$是给定PDE可用数据集的量。

**预测问题**：在预测问题中，我们希望使用学习的模型来预测未知时间区域中的解。设我们学习的模型为$\mathcal{F}(t,\mathbf{x},\mathbf{v},\theta)$，那么预测解是$\mathcal{F}(\tilde{t},\mathbf{x},\mathbf{v},\theta)$，其中$\tilde{t}$是不同于网格中时间戳$t$的未知时间戳。

（

我们要研究的对象是一个偏微分方程（PDE）。它的解 $u$ 依赖于：

- **时间 $t$**
- **空间位置 $\mathbf{x}$**
- **速度 $\mathbf{v}$**
- **一些物理参数 $\boldsymbol{\beta}$（比如扩散系数、粘性系数、导热系数等）**

整个方程可以抽象写成下面的形式：
$$
\begin{cases} 
\mathcal{L}[u(t,\mathbf{x},\mathbf{v},\boldsymbol{\beta})] = f(t,\mathbf{x},\mathbf{v}), & \text{在区域内部}, \\
\mathcal{B}[u(t,\mathbf{x},\mathbf{v},\boldsymbol{\beta})] = g(t,\mathbf{x},\mathbf{v}), & \text{在边界上}, \\
\mathcal{I}[u(0,\mathbf{x},\mathbf{v},\boldsymbol{\beta})] = h(\mathbf{x},\mathbf{v}), & \text{在初始时刻}.
\end{cases} \tag{1}
$$
其中：

- $\mathcal{L}$ 是控制系统行为的算子（比如“热传导方程”里的扩散算子，或者“流体方程”里的对流和压力算子）。
- $\mathcal{B}$ 是边界条件，规定边界上解的取值。
- $\mathcal{I}$ 是初始条件，规定 $t=0$ 时解的状态。
- $f, g, h$ 分别是外部的“源项”（比如热源）、边界值和初始值。

简单理解：

- **系统内部**：方程 $\mathcal{L}[u] = f$ 描述内部演化规律。
- **系统边界**：边界条件 $\mathcal{B}[u] = g$ 限制了解在外部边界的行为。
- **系统初始状态**：初始条件 $\mathcal{I}[u] = h$ 决定了系统一开始的情况。

在数值分析和机器学习里，PDE 会对应三类常见问题：

1. **正向问题（Forward Problem）**
    已知方程、参数、初始条件和边界条件，要求解 $u$ 的演化过程。
   - 就像“给定一个物理模型和初始环境，算出系统会怎么变化”。
   - 比如用有限差分法一步步算，或者用神经网络近似解。
2. **逆问题（Inverse Problem）**
    已知部分解的观测数据，要求反推出方程里的 **未知参数 $\boldsymbol{\beta}$**。
   - 就像“我们观测到一些温度随时间的变化，反过来推算出材料的导热系数是多少”。
   - 这种问题需要额外的数据集来辅助。
3. **预测问题（Prediction Problem）**
    已知某个时间段内的解，希望预测 **未来时刻** 的解。
   - 就像“我们知道过去 3 天的天气分布，想预测第 4 天的情况”。
   - 用神经网络表示时，如果模型是 $\mathcal{F}(t,\mathbf{x},\mathbf{v},\theta)$，那我们希望能预测在一个新时间点 $\tilde{t}$ 的解 $\mathcal{F}(\tilde{t},\mathbf{x},\mathbf{v},\theta)$。

）

## 3 相关工作

**传统数值方法**：作为一个基本的科学问题，获得偏微分方程的解析解通常很困难。因此，在实践中，PDE通常被离散化为网格，并使用数值方法求解，如有限差分法[21]、有限元法[8]、谱方法[19]等。然而，这些数值方法对于复杂结构通常需要几小时甚至几天[22]，在预测问题上表现不佳。

**无数据深度学习方法**：最受欢迎的无数据深度学习方法之一是PINNs。这种方法将PDE的约束（包括方程本身以及初始和边界条件）制定为深度学习模型中的目标函数[17, 28]。在训练过程中，这些模型的输出逐渐与PDE约束对齐，使它们能够有效地近似PDE的解。然而，这种方法主要依赖卷积神经网络，忽略了实际物理系统中固有的关键时间依赖性，使得预测给定训练网格外的解变得困难。此外，直接使用PINNs在某些复杂场景中将无法学习PDE的解。例如，[10, 14]提出了渐近保持PINNs来解决多尺度方程造成的困难。

**基于算子学习的方法**：基于算子学习的方法构成了数据驱动深度学习方法中的一个重要类别，涉及训练神经算子来近似由偏微分方程控制的任务中的输入-输出关系。这种方法可以应用于许多物理场景，如基于过去观察预测未来流体行为或估计固体材料中的内应力[15]。该领域一些最知名的模型包括傅里叶神经算子[12]及其变体，如[13]和[16]中描述的模型。已有一些工作[27]专注于预测PDE中的动力学。然而，它们通常需要大量数据，且大多忽略了利用PDE内部的内在物理机制。

**基于Transformer的模型**：Transformer模型[23]因其捕获长期依赖性的能力而受到相当关注，在自然语言处理任务中取得了重大进展[11]。此外，Transformers已被适用于各种其他领域，包括计算机视觉、语音识别和时间序列分析[5, 7, 25]。然而，在解决PDE的transformers应用方面的研究很少。最近的研究者已经使用transformer来学习给定PDE的解[4, 26, 29]。然而，PINNs和transformer的结合尚未很好地融合，PDE中的预测任务也没有得到充分研究。

## 4 方法论

为了解决正向和预测任务，特别是预测任务，我们设计了名为PhysicsSolver的创新模型。在本节中，我们将主要介绍所提出方法的两个部分：模型设计和学习方案。在模型设计部分，我们将介绍几个模块，包括插值伪序列网格生成器(IPSGG)、Halton序列生成(HSG)和物理注意力(P-Attention)。在学习方案部分，我们将提供损失函数和训练算法等完整细节。

### 4.1 模型结构

![image-20250901112237164](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20250901112237164.png)

我们在图1中总结了我们提出的PhysicsSolver用于正向和预测问题的框架。在经过IPSGG和HSG模块处理后，生成的网格点被输入到物理注意力模块中。在编码-解码过程中，我们可以获得近似解。一个估计解$\hat{u}*{physics}$将被放入物理系统进行训练，而另一个基于Halton序列的估计解$\hat{u}*{data}$将被放入数据系统进行后续训练。我们在训练过程中不使用给定PDE的任何真实数据，除了由HSG模块生成的稀疏网格点上的解。

(

这张图展示了我们提出的 **PhysicsSolver** 的整体框架，它的目标是同时解决 **正向问题** 和 **预测问题**。整体可以分为三个部分：

------

##### ① 输入与编码

- 输入包括 **时间 $t$、空间位置 $x$、速度 $v$**（图左下角的彩色小方块）。
- 这些输入经过 **伪序列网格生成器 (IPSGG)** 和 **Halton 序列采样 (HSG)** 处理之后，生成一组稀疏但有代表性的网格点。
- 输入点先经过一个小的多层感知机（MLP）嵌入，再送入 **Transformer 的编码器（Encoder）和解码器（Decoder）**。
- **P-Attention** 表示“物理注意力模块”，它能捕捉输入点之间的物理相关性和随时间的变化规律。

------

##### ② 物理系统与数据系统

PhysicsSolver 同时训练两套系统：

1. **物理系统 (Physics System)**
   - 用于学习 PDE 的约束。
   - 它强制预测解 $\hat{u}_{physics}$ 满足 PDE 的基本条件，包括：
     - 方程约束 $\mathcal{L}[u]=f$
     - 边界条件 $\mathcal{B}[u]=g$
     - 初始条件 $\mathcal{I}[u]=h$
2. **数据系统 (Data System)**
   - 用于利用采样得到的稀疏数据点（来自 Halton 序列）。
   - 预测解 $\hat{u}_{data}$ 会和这些采样点对齐，起到补充训练的作用。

------

##### ③ 损失函数与优化

训练时，PhysicsSolver 会同时考虑几类损失：

- **$loss_{Re}$**：残差损失，约束 PDE 方程本身。
- **$loss_{BC}$**：边界条件损失。
- **$loss_{IC}$**：初始条件损失。
- **$loss_{Data}$**：数据损失，用于对齐稀疏采样数据。

这些损失加权求和（权重 $w_1, w_2, w_3, w_4$），形成总损失 $loss_{total}$，训练的目标就是最小化这个总损失。

------

##### ④ 模型的关键点

- 在训练过程中，PhysicsSolver **不依赖任何真实的大规模 PDE 数据**，只利用 PDE 本身的方程信息 + Halton 采样的稀疏点。
- **物理系统**让模型遵守物理规律，**数据系统**让模型具备预测能力。两者结合后，模型既节省数据，又能更准确地预测超出已知区域的解。

------

👉 通俗比喻：
 可以把 PhysicsSolver 想象成一个“物理学徒”：

- 一方面，师傅（PDE 方程）不断纠正它，让它不要违背基本物理规律；
- 另一方面，它也要根据自己手头上有限的实验数据（Halton 序列采样点）不断修正自己的预测。
- 最后，它既学会了“规矩”（物理约束），又能举一反三去预测未来（预测任务）。

)

### 4.2 数据生成工程

在本节中，我们介绍两个名为插值伪序列网格生成器和Halton序列生成的数据生成模块，旨在为PhysicsSolver准备训练序列。

#### 4.2.1 插值伪序列网格生成器

基于Transformer的模型旨在捕获序列数据中的依赖关系，而传统的PINNs依赖固定在特定网格上的非序列数据作为神经网络输入。因此，为了将PINNs与基于Transformer的模型集成，我们必须将点状时空输入转换为时间序列。同时，生成的序列点应该独立于给定的网格点。所以我们设计了插值伪序列网格生成器，它可以以以下方式为给定的第$p$个$(t_p,\mathbf{x}_p,\mathbf{v}_p)$生成序列网格：

$$ (t_p,\mathbf{x}_p,\mathbf{v}_p) \xrightarrow{IPSGG} {(t_p,\mathbf{x}_p,\mathbf{v}_p), (t_p+\gamma\Delta t,\mathbf{x}_p,\mathbf{v}_p), \ldots, (t_p+(k-1)\gamma\Delta t,\mathbf{x}_p,\mathbf{v}_p)}, \tag{2} $$

其中$k \in \mathbb{Z}$，$(k-1)\gamma \notin \mathbb{Z}$，$\mathbb{Z}$表示整数集。参数$k$和$\Delta t$是超参数，决定给定网格应向前移动的步数和每步的持续时间，而$\gamma$是控制参数，确保生成的序列点与用于正向和预测问题的网格保持独立。

#### 4.2.2 Halton序列生成器

为了将真实数据的额外信息纳入我们的PhysicsSolver中，我们设计了Halton序列生成器，它可以帮助从给定区域采样相对稀疏但更具代表性的数据点。与无数据模型相比，Halton序列的使用提高了模型精度。Halton序列方法是一种拟蒙特卡洛方法，专门用于解决复杂系统中的一些问题，如高维系统，因为它具有多维低差异特性[3, 24]。我们基于以下原因使用Halton序列。与非采样方法相比，采样方法可以显著减少计算资源。因此，样本大小$n_{data}$通常远小于物理系统中使用的网格大小$n_{physics}$。此外，拟蒙特卡洛方法可以基于随机采样增强蒙特卡洛近似的精度[9, 18]。Halton序列的生成如下：假设上面定义的正向问题和预测问题的网格空间是$\mathcal{X} \in \mathbb{R}^d$，其中$d$是网格点的维度。我们使用互质数作为基础，从网格空间$\mathcal{X}' \in \mathbb{R}^d$构造Halton序列。在实践中，$\mathcal{X}'$通常是$\mathcal{X}$的补集。

(

##### 4.2 数据生成工程（通俗解释）

PhysicsSolver 在训练时，需要把原始输入数据（时间、空间、速度点）变成适合 **Transformer** 学习的“序列”，还需要挑选一些稀疏但有代表性的点来辅助学习。为此，我们设计了两个数据生成模块：

------

##### 4.2.1 插值伪序列网格生成器 (IPSGG)

- **为什么需要它？**

  - Transformer 模型天生适合处理“序列数据”（像句子、时间序列）。
  - 但传统的 PINNs 输入通常是“点状的”，固定在某些网格点上，不是序列。
  - 所以，如果直接把 PINNs 和 Transformer 结合，二者就对不上。

- **解决办法：**
   我们把一个点 $(t_p, \mathbf{x}_p, \mathbf{v}_p)$ 转换成一段“伪时间序列”：
  $$
  (t_p, \mathbf{x}_p, \mathbf{v}_p) 
  \;\;\xrightarrow{IPSGG}\;\;
  (t_p, \mathbf{x}_p, \mathbf{v}_p), \; (t_p+\gamma \Delta t, \mathbf{x}_p, \mathbf{v}_p), \ldots
  $$

  - 其中 $\Delta t$ 是步长，$k$ 是生成的序列长度。
  - $\gamma$ 是一个特殊的控制参数，用来确保新生成的序列点不和原来训练网格里的点“重叠”。

- **直观理解：**
   就好比你只有某个时刻的温度数据，我们让模型自己延伸出一个“小时间片段”，这样它就能学会捕捉随时间演化的规律。

------

##### 4.2.2 Halton 序列生成器 (HSG)

- **为什么需要它？**
  - 完全“无数据”的模型往往预测不够准。
  - 如果我们能补充一些“额外的数据点”，模型会学得更稳。
  - 但直接在高维网格上采样点，代价太大（点数多、计算量爆炸）。
- **解决办法：**
   使用 **Halton 序列** 来采样。
  - Halton 序列是一种 **拟蒙特卡洛方法**，特点是“分布均匀、覆盖全面”。
  - 特别适合高维场景（比如同时有时间、空间、速度多个维度）。
  - 用它采样得到的点数少，但代表性强，可以大幅减少计算量。
- **直观理解：**
   想象你要在草坪上插探测棒测温度：
  - 随机撒点（普通蒙特卡洛）：可能有的地方挤在一起，有的地方漏掉。
  - 规则网格：点太多，成本太高。
  - Halton 序列：少量点，但分布得比较均匀，能代表整体情况。
- **效果：**
  - 数据点数量 $n_{data}$ 通常远小于 PDE 物理网格点数量 $n_{physics}$。
  - 用更少的数据，就能让模型学到更多有用信息。

------

👉 总结来看：

- **IPSGG**：把孤立点扩展成伪时间序列，让 Transformer 能理解“时间演化”。
- **HSG**：用少量但均匀分布的代表性数据点，帮助模型更准确、更高效地学习。

)

### 4.3 物理信息神经网络

标准的物理信息神经网络(PINNs)[17]将底层PDE的残差以及初始和边界条件或其他相关物理属性集成到损失函数中。这些网络利用自动微分在计算域中特定点评估的损失函数中嵌入物理定律以及数据驱动项。

我们首先在我们的问题背景下回顾标准PINNs方法。对于正向问题，我们的目标是近似指定PDE(1)的解。考虑一个具有$L$层的神经网络，输入层接受$(t,\mathbf{x},\mathbf{v})$，最后一层的输出表示为$u_{NN}(t,\mathbf{x},\mathbf{v};m,w,b)$或更简洁地表示为$u_\theta^{NN}$，其中$\theta$表示神经网络参数。第$l$层和第$(l+1)$层之间的关系($l=1,2,\cdots,L-1$)由下式给出：

$$ n_j^{(l+1)} = \sum_{i=1}^{m_l} w_{ji}^{(l+1)}\sigma_l(n_i^l) + b_j^{(l+1)}, \tag{3} $$

其中$m=(m_0,m_1,m_2,\ldots,m_{L-1})$，$w={w_{ji}^{(k)}}*{i,j,k=1}^{m*{k-1},m_k,L}$和$b={b_j^{(k)}}_{j=1,k=1}^{m_k,L}$。更具体地：

1. $n_i^l$：第$l$层中的第$i$个神经元
2. $\sigma_l$：第$l$层中的激活函数
3. $w_{ji}^{(l+1)}$：第$l$层中第$i$个神经元和第$(l+1)$层中第$j$个神经元之间的权重
4. $b_j^{(l+1)}$：第$(l+1)$层中第$j$个神经元的偏置
5. $m_l$：第$l$层中的神经元数量

设$\hat{u}$为神经网络近似，在传统PINNs方法中，我们使用以下损失函数来获得$\hat{u}$上的约束。

$$ \mathcal{L}*{PINNs} := w_r \int*{\mathcal{T}} \int_{\mathcal{D}} \int_{\Omega_{\mathbf{v}}} |\mathcal{L}[\hat{u}(t,\mathbf{x},\mathbf{v})] - f(t,\mathbf{x},\mathbf{v})|^2 d\mathbf{v} d\mathbf{x} dt \tag{4} $$ $$

- w_b \int_{\mathcal{T}} \int_{\gamma} \int_{\Omega_{\mathbf{v}}} |\mathcal{B}[\hat{u}(t,\mathbf{x},\mathbf{v})] - g(t,\mathbf{x},\mathbf{v})|^2 d\mathbf{v} d\mathbf{x} dt $$ $$
- w_i \int_{\mathcal{D}} \int_{\Omega_{\mathbf{v}}} |\mathcal{I}[\hat{u}(0,\mathbf{x},\mathbf{v})] - h(0,\mathbf{x},\mathbf{v})|^2 d\mathbf{v} d\mathbf{x}. $$

其中$\mathcal{T}$是时间空间，$\mathcal{D}$是空间域，$\gamma$是边界空间，$\Omega_{\mathbf{v}}$是速度空间。$w_r, w_b, w_i$分别是残差权重、边界权重和初始值权重。方程(4)的经验损失$\mathcal{L}_{PINNs_empirical}$可以在附录的方程(23)中找到。在实践中，神经网络通过最小化经验损失函数进行训练。

(和上面说的一样)

### 4.4 物理注意力

PhysicsSolver的基本结构是编码器-解码器架构，类似于Transformer[23]。在PhysicsSolver中，编码器由一个称为物理注意力(P-Attention)的自注意力层和一个前馈层组成，其中P-Attention主要用于处理包含物理输入信息和数据输入信息的混合嵌入。解码器保持与编码器类似的结构。此外，编码器和解码器都使用层归一化方案。编码器用于处理由MLP处理的物理输入和数据输入的混合嵌入。具体地，物理和数据输入首先通过一个MLP，生成两个不同的嵌入。然后将这些嵌入组合成混合表示，由编码器进一步处理。

直观上，P-Attention的自注意力机制使模型能够学习所有时空信息的物理依赖性。同时，它可以有效地学习从输入数据表示到输出解的映射，这显著有助于预测任务。这种能力使模型能够捕获比传统PINNs更多的信息，传统PINNs主要专注于近似当前状态的解。它也超越了某些基于transformer的模型，如PINNsformer，后者也没有有效利用数据。

假设我们已经从输入中获得了混合嵌入$\mathcal{Y} \in \mathbb{R}^{l \times d}$。这里$l$是选定网格的数量，$d = 2N + 1$是每个网格的输入维度。$\mathcal{Y}$首先通过MLP层转换为$\tilde{\mathcal{Y}} \in \mathbb{R}^{l \times d'}$，其中$d' \geq d$。然后通过编码器转换为$\mathcal{Y}_{En} \in \mathbb{R}^{l \times d'}$，其中：

$$ \mathcal{Y}_{En} = \text{Ln}(\text{Feed}(\text{Attn}(\text{Ln}(\tilde{\mathcal{Y}})))), \tag{5} $$

Ln是层归一化，Feed是前馈层，Attn是注意力算子。特别地，当我们在编码和解码阶段执行注意力操作时，我们定义类似于[4]的可训练投影矩阵。潜在表示$Q, K, V$定义如下：

$$ Q := \tilde{\mathcal{Y}}W^Q, \quad K := \tilde{\mathcal{Y}}W^K, \quad V := \tilde{\mathcal{Y}}W^V, \tag{6} $$

其中$W^Q, W^K, W^V \in \mathbb{R}^{d' \times d'}$是可训练投影矩阵。注意力操作定义为：

$$ \text{Attn}(\tilde{\mathcal{Y}}) := \text{Softmax}\left(\frac{QK^T}{|QK^T|_{l_2}}\right)V. \tag{7} $$

在解码过程中，编码嵌入$\mathcal{Y}_{En}$被传输到解码器中，我们得到解码嵌入如下：

$$ \mathcal{Y}*{De} = \text{Ln}(\text{Feed}(\text{Ln}(\text{Attn}(\mathcal{Y}*{En})))), \tag{8} $$

其中$\mathcal{Y}*{De} \in \mathbb{R}^{l \times d'}$。通过输出MLP层处理后，$\mathcal{Y}*{De}$转换为$\mathcal{Y}*{final}$。$\mathcal{Y}*{final}$可以被视为输入时间序列的最终表示，并作为PDE系统的近似解$\hat{u}$。该表示随后可以用作物理系统和数据系统的近似解$\hat{u}*{physics}$和$\hat{u}*{data}$。

（

PhysicsSolver 中的 **P-Attention** 改造了**Attention**机制，使其更适合 PDE 场景：

1. **输入不同**

   - 普通 Attention 输入是自然语言序列或时间序列。
   - P-Attention 输入的是 **物理输入（$t,x,v$）+ 数据输入** 的混合嵌入。

2. **归一化方式不同**

   - 标准 Attention 的分母是 $\sqrt{d_k}$。
   - P-Attention 用的是 $||QK^T||_{l_2}$（L2 范数归一化）。
   - 这样可以更稳定地处理 **高维连续时空数据**，避免数值过大或过小。

   

）

### 4.5 学习方案

在本节中，我们通过调整传统PINNs损失(4)来展示PhysicsSolver的学习方案。对于物理系统，我们考虑残差损失$\mathcal{L}*{physics_res}$、边界条件损失$\mathcal{L}*{physics_bc}$和初始条件损失$\mathcal{L}_{physics_ic}$如下：

$ \mathcal{L}*{physics_res} = \frac{1}{kN_r}\sum*{p=1}^{N_r}\sum_{j=0}^{k-1}|\mathcal{L}[\hat{u}_{physics}(t_p+j\gamma\Delta t,\mathbf{x}_p,\mathbf{v}_p)] - f(t_p+j\gamma\Delta t,\mathbf{x}_p,\mathbf{v}_p))|^2, \tag{9} $

$ \mathcal{L}*{physics_bc} = \frac{1}{kN_b}\sum*{p=1}^{N_b}\sum_{j=0}^{k-1}|\mathcal{B}[\hat{u}_{physics}(t_p+j\gamma\Delta t,\mathbf{x}_p,\mathbf{v}_p)] - g(t_p+j\gamma\Delta t,\mathbf{x}_p,\mathbf{v}_p))|^2, $

$ \mathcal{L}*{physics_ic} = \frac{1}{N_i}\sum*{p=1}^{N_i}|\mathcal{I}[\hat{u}_{physics}(0,\mathbf{x}_p,\mathbf{v}_p)] - h(0,\mathbf{x}_p,\mathbf{v}_p)|^2. $

其中$N_r, N_b$和$N_i$分别是残差点、边界条件点和初始条件点的数量。物理系统的总损失如下：

$ \mathcal{L}*{physics} = w_r\mathcal{L}*{physics_res} + w_b\mathcal{L}*{physics_bc} + w_i\mathcal{L}*{physics_ic}. \tag{10} $

我们还引入数据系统的解失配损失：

$ \mathcal{L}*{data} = \sum*{d=1}^{n_{data}}(u_{data}(t_d,\mathbf{x}_d,\mathbf{v}_d) - u(t_d,\mathbf{x}_d,\mathbf{v}_d))^2, \tag{11} $

其中网格点${(t_d,\mathbf{x}*d,\mathbf{v}\*d)}\*{d=1}^{n*{data}}$是由HSG模块生成的数据点，$u(t_d,\mathbf{x}_d,\mathbf{v}_d)$是真实解。最后，PhysicsSolver用于正向和预测问题的完整损失定义如下：

$ \mathcal{L}*{PhysicsSolver} = \lambda_1\mathcal{L}*{physics} + \lambda_2\mathcal{L}_{data}. \tag{12} $

这里$\lambda_1$和$\lambda_2$是惩罚参数。

（

**物理损失** 就像“老师要求你写的解答必须符合公式推导”（物理规则）。

**数据损失** 就像“你还要把答案和实验数据对齐”。

**总损失** 就是这两类要求的加权平均。

想要得高分，模型必须同时满足 **物理规律** 和 **数据一致性**。

）

### 4.6 算法

我们在算法1中展示了训练PhysicsSolver的算法。定义$C$为我们使用的网格的特征维度（除时间维度外）。我们有：

$ C = X \times V, \quad C_b = X_b \times V_b, \quad C_i = X_i \times V_i. \tag{13} $

这里$T, T_b$是内部域和边界的时间点数；$X, X_b, X_i$是内部域、边界和初始域的空间点数；$V, V_b, V_i$分别是内部域、边界和初始域使用的速度点数。

**算法1** PhysicsSolver的训练过程

1. **初始化**：初始化超参数：学习率$\eta$，IPSGG中的参数$k$、控制参数$\gamma$。初始化PhysicsSolver模块的网络权重参数$\theta_{PhysicsSolver}$，这是MLP模块和编码器-解码器模块的综合权重
2. **数据生成**：通过IPSGG模块生成$n_1$个伪序列网格$\mathbf{X}*{physics} = {(t_p,\mathbf{x}\*p,\mathbf{v}\*p)}\*{p=1}^{n_1}$，见方程(2)。我们有$\mathbf{X}\*{physics} = \mathbf{X}*{1:N_r} \cup \mathbf{X}*{1:N_b} \cup \mathbf{X}*{1:N_i}$。这里$\mathbf{X}*{1:N_r} \in \mathbb{R}^{T \times C}$，$\mathbf{X}*{1:N_b} \in \mathbb{R}^{T_b \times C_b}$和$\mathbf{X}*{1:N_i} \in \mathbb{R}^{C_i}$分别是内部、边界和初始网格点。通过HSG模块生成$n_2$个网格点$\mathbf{X}*{data} = {(t_d,\mathbf{x}_d,\mathbf{v}*d)}*{d=1}^{n_2}$
3. **输入**：内部网格点$\mathbf{X}*{1:N_r} \in \mathbb{R}^{T \times C}$，边界网格点$\mathbf{X}*{1:N_b} \in \mathbb{R}^{T_b \times C_b}$，初始网格点$\mathbf{X}*{1:N_i} \in \mathbb{R}^{C_i}$，和数据网格点$\mathbf{X}*{data}$
4. **for** epoch = 1 to max_epoch **do**
5. ​    将$\mathbf{X}*{1:N_r}, \mathbf{X}*{1:N_b}, \mathbf{X}*{1:N_i}$和$\mathbf{X}*{data}$输入MLP层，然后是PhysicsSolver中的编码器。然后获得表示$\mathbf{E}*{1:N_r}, \mathbf{E}*{1:N_b}, \mathbf{E}*{1:N_i}$和$\mathbf{E}*{data}$，见方程(5)
6. ​    解码$\mathbf{E}*{1:N_r}, \mathbf{E}*{1:N_b}, \mathbf{E}*{1:N_i}$和$\mathbf{E}*{data}$以获得表示$\mathbf{D}*{1:N_r}, \mathbf{D}*{1:N_b}, \mathbf{D}*{1:N_i}$和$\mathbf{D}*{data}$，见方程(8)。通过PhysicsSolver中的MLP层处理，我们可以获得物理系统和数据系统的近似解$\hat{u}*{physics}$和$\hat{u}*{data}$
7. ​    获得物理系统的训练损失$\mathcal{L}_{physics}$，见方程(10)
8. ​    获得数据系统的训练损失$\mathcal{L}_{data}$，见方程(11)
9. ​    获得PhysicsSolver的总损失$\mathcal{L}*{PhysicsSolver} = \lambda_1\mathcal{L}*{physics} + \lambda_2\mathcal{L}_{data}$，见方程(12)
10. ​    **梯度更新**：$\theta \leftarrow \theta - \eta\nabla_\theta\mathcal{L}*{PhysicsSolver}(\theta)$，其中$\theta = \theta*{PhysicsSolver}$
11. **end for**

（

这一节描述的是 **PhysicsSolver 的训练过程**，其实就是一套改造过的 **深度学习训练循环**。整体框架和普通神经网络类似：
 **初始化 → 生成数据 → 前向传播 → 计算损失 → 反向传播更新参数**。

------

##### 1. 初始化

- 设置超参数，比如：
  - 学习率 $\eta$（决定梯度下降步子的大小）。
  - $k, \gamma$（IPSGG 模块的参数，控制伪序列的长度和间隔）。
- 初始化 PhysicsSolver 模型的参数 $\theta_{PhysicsSolver}$（包括 MLP、编码器、解码器的权重）。

👉 就像准备考试前先定学习计划（参数）和清空笔记本（初始化网络）。

------

##### 2. 数据生成

- **用 IPSGG 模块**：在时空域生成 $n_1$ 个伪序列网格点，包含：
  - 内部点（残差点）$\mathbf{X}_{1:N_r}$
  - 边界点 $\mathbf{X}_{1:N_b}$
  - 初始点 $\mathbf{X}_{1:N_i}$
- **用 HSG 模块**：额外生成 $n_2$ 个稀疏但有代表性的数据点 $\mathbf{X}_{data}$。

👉 就像先准备“理论练习题”（物理点）+ “实验数据点”（真实观测）。

------

##### 3. 输入

把所有点（内部/边界/初始/数据）都作为 PhysicsSolver 的输入。

------

##### 4. 训练循环（epoch 循环）

对每一轮训练（epoch），做以下步骤：

#### (a) 编码

- 所有输入点先通过 MLP 进行嵌入。
- 再送入 **P-Attention 编码器**，得到表示 $\mathbf{E}*{1:N_r}, \mathbf{E}*{1:N_b}, \mathbf{E}*{1:N_i}, \mathbf{E}*{data}$。

#### (b) 解码

- 编码后的表示送入解码器，再得到对应的 $\mathbf{D}$ 表示。
- 最后通过输出层 MLP 得到近似解：
  - **物理系统解** $\hat{u}_{physics}$
  - **数据系统解** $\hat{u}_{data}$

#### (c) 计算损失

- **物理系统损失** $\mathcal{L}_{physics}$（残差+边界+初始，见公式 (10)）。
- **数据系统损失** $\mathcal{L}_{data}$（预测值 vs 真实数据点，见公式 (11)）。
- **总损失** $\mathcal{L}*{PhysicsSolver} = \lambda_1 \mathcal{L}*{physics} + \lambda_2 \mathcal{L}_{data}$。

👉 就像老师同时打“公式分”和“实验分”，两者加权得总分。

#### (d) 梯度更新

- 用反向传播计算 $\nabla_\theta \mathcal{L}$。

- 更新参数：
  $$
  \theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}_{PhysicsSolver}
  $$

- 让网络逐渐学会既满足物理规律，又拟合稀疏数据。

------

### 5. 循环结束

经过若干轮训练（max_epoch），模型收敛，PhysicsSolver 就可以用来解决 **正向问题**（已知条件下求解）和 **预测问题**（外推未来解）

）

## 5 数值实验

为了展示我们的模型相对于传统数值方法和其他深度学习模型的优越性，我们在四个PDE系统上测试了不同模型的性能，一些实验案例来自[17]和[29]。对于每个案例，我们对目标PDE进行两种数值实验：正向问题和预测问题。对于正向问题，我们将PhysicsSolver与PINNs和PINNsformer等深度学习方法进行测试。对于预测问题，我们将PhysicsSolver与PINNs和PINNsformer等深度学习方法以及外推法等传统数值方法进行测试。在预测问题中，我们考虑单步预测，这意味着我们只使用训练模型预测一个未知时间步。

**网络架构**：对于PhysicsSolver方法，我们使用具有2个头的Transformer骨干，隐藏大小设置为512，嵌入大小设置为32。我们设置控制参数$\gamma = 1.1$，超参数$k = 5$。在不同的实验设置中，步长$\Delta t$在$1 \times 10^{-2}, 1 \times 10^{-3}$和$1 \times 10^{-4}$之间变化，HSG生成的数据点数$n_{data}$设置为2、4、5或10。对于PINNsFormer方法，我们使用[29]中的类似设置。对于PINNs方法，我们通过前馈神经网络(FNN)近似解，该网络有一个输入层、一个输出层和4个隐藏层，每层有512个神经元，除非另有说明。双曲正切函数(Tanh)被选为我们的激活函数。对于预测问题中使用的外推法，我们考虑以下有限差分公式来估计第$t+1$个时间步的解：

$ \frac{\hat{u}(t+1,\mathbf{x}) - u(t,\mathbf{x})}{\Delta t} = \frac{u(t,\mathbf{x}) - u(t-1,\mathbf{x})}{\Delta t}. \tag{14} $

所以我们有估计解$\hat{u}(t+1,\mathbf{x})$如下：

$ \hat{u}(t+1,\mathbf{x}) = 2u(t,\mathbf{x}) - u(t-1,\mathbf{x}). \tag{15} $

**训练设置**：一般来说，我们使用固定随机种子训练所有模型，使用L-BFGS优化器与Strong Wolfe线性搜索进行500次迭代，并在以下数值实验中的大多数实验中使用全批次，除非另有说明。所有超参数都通过试错选择。

**损失设计**：对于我们的大多数实验，我们考虑空间和时间域分别为$[0, 2\pi]$和$[0, 1]$。我们以以下方式为$u(t,x)$选择配点${(t_i,x_i)}$。对于空间点$x_i$，我们在$[0, 2\pi]$中选择均匀间隔的内部点。对于时间点$t_i$，我们在范围$[0, 1]$中选择20个均匀间隔的内部点。我们对配点使用张量积网格。为简单起见，方程(10)中的权重参数设置为$(w_1, w_2, w_3) = (1, 1, 1)$。方程(12)中的惩罚参数设置为$(\lambda_1, \lambda_2) = (1, 1)$。

**测试设置**：在评估两个主要任务：正向问题和预测问题时，我们采用了相关工作[10, 29]中常用的指标，包括解近似$u_{NN}(t,x,\beta)$和参考解$u_{ref}(t,x_j,\beta)$之间的相对$\ell_2$或$\ell_\infty$误差，相对$\ell_2$误差定义为：

$ \mathcal{E}(t) := \frac{\sum_j|u_{NN}(t,x_j,\beta) - u_{ref}(t,x_j,\beta)|^2}{\sum_j|u_{ref}(t,x_j,\beta)|^2}. \tag{16} $

**可重现性**：所有模型都在PyTorch中实现，我们在配备Intel(R) Xeon(R) Gold 6230和两个A40 48GB GPU的服务器上运行实验。

### 5.1 线性对流方程

一维对流方程是一个时间相关方程，广泛用于建模输运现象。考虑1D情况下具有初值条件和周期边界条件的以下一般形式：

$ \begin{cases} \frac{\partial u}{\partial t} + \frac{\partial f(u)}{\partial x} = 0, & (t,x) \in [0,1] \times [0,2\pi], \ u(x,0) = \sin(x), \ u(0,t) = u(2\pi,t). \end{cases} \tag{17} $

其中$f(u)$是通量，$u(t,x)$是守恒量。这里我们设置$\frac{df}{du} = 50$为常数。参考解通过使用不同的有限方法获得。对于物理系统，我们从上述空间和时间域中采样101个时间网格和101个空间网格，类似于[29]。对于数据系统，我们然后使用HSG模块从网格空间为数据系统生成额外的5个时间戳（每个时间戳选择101个空间点）。我们在正向问题和预测问题中使用前100个时间步结合来自数据系统的额外5个时间戳进行训练，并在预测问题中使用最后的时间步进行测试。

**正向问题**：在正向问题中，我们调查不同方法对解推理的性能。图2显示了不同方法之间预测解的比较。我们可以发现PINNs随着$t$的增加无法捕获解，而PINNsFormer和PhysicsSolver可以很好地捕获全局解。更多细节可以在表1中找到，表明PhysicsSolver在正向问题中的表现相对优于PINNsFormer。

**单步预测问题**：在单步预测问题中，我们旨在预测下一个时间步的解。图3比较了不同方法在最后和倒数第二个时间点获得的解。可以得出结论，当解随时间变化相对显著时，外推法无法准确预测未来解。更多细节可以在表2中找到。我们可以观察到PhysicsSolver优于其他三种方法，这得益于其通过数据系统学习更好映射的能力，这增强了其预测能力。

| 方法 | PINNs                  | PINNsFormer            | PhysicsSolver                   |
| ---- | ---------------------- | ---------------------- | ------------------------------- |
| 误差 | $8.189 \times 10^{-1}$ | $2.240 \times 10^{-2}$ | $\mathbf{1.558 \times 10^{-2}}$ |

*表1：相对$l_2$误差。*

| 方法 | 外推法                 | PINNs                 | PINNsFormer            | PhysicsSolver                   |
| ---- | ---------------------- | --------------------- | ---------------------- | ------------------------------- |
| 误差 | $2.462 \times 10^{-1}$ | $9.96 \times 10^{-1}$ | $3.634 \times 10^{-2}$ | $\mathbf{3.246 \times 10^{-2}}$ |

*表2：相对$l_2$误差。*

### 5.2 反应方程

一维反应方程是一个演化PDE，常用于建模化学反应过程。考虑具有初值条件和周期边界条件的以下1D方程：

$ \begin{cases} \frac{\partial u}{\partial t} - \rho u(1-u) = 0, & (t,x) \in [0,1] \times [0,2\pi], \ u(x,0) = e^{-\frac{(x-\pi)^2}{2(\pi/4)^2}}, \ u(0,t) = u(2\pi,t). \end{cases} \tag{18} $

其中$\rho$是反应系数，这里我们设置$\rho = 5$，类似于[29]。

参考解通过以下形式获得：

$ u_{reference} = \frac{h(x)e^{\rho t}}{h(x)e^{\rho t} + 1 - h(x)}. \tag{19} $

其中$h(x)$是方程(1)中定义的初始条件函数。对于物理系统，我们从上述空间和时间域中采样101个时间网格和101个空间网格，类似于[29]。对于数据系统，我们应用HSG模块从网格空间生成额外的10个时间戳（每个时间戳选择101个空间点）。我们在正向问题和预测问题中使用前100个时间步结合来自数据系统的额外10个时间戳进行训练，并在预测问题中使用最后的时间步进行测试。

**正向问题**：在正向问题中，我们调查不同方法对解推理的性能。图4显示了不同方法之间预测解的比较。我们可以发现PINNs随着$t$的增加无法捕获解，而PINNsFormer和PhysicsSolver可以很好地捕获全局解。更多细节可以在表3中找到，表明PhysicsSolver在正向问题中的表现远优于PINNsFormer，这得益于其通过数据系统学习更好映射的能力，这可以提高正向问题的精度。

**单步预测问题**：在单步预测问题中，我们旨在预测下一个时间步的解。图5比较了不同方法在最后和倒数第二个时间点获得的解。可以得出结论，PINNs方法无法准确预测未来解。然而，当解随时间变化相对较小时，外推法效果很好。更多细节可以在表4中找到。我们可以观察到PhysicsSolver优于其他三种方法。

| 方法 | PINNs                  | PINNsFormer            | PhysicsSolver                   |
| ---- | ---------------------- | ---------------------- | ------------------------------- |
| 误差 | $9.803 \times 10^{-1}$ | $4.563 \times 10^{-2}$ | $\mathbf{5.550 \times 10^{-3}}$ |

*表3：相对$l_2$误差。*

| 方法 | 外推法                 | PINNs                  | PINNsFormer            | PhysicsSolver                   |
| ---- | ---------------------- | ---------------------- | ---------------------- | ------------------------------- |
| 误差 | $6.915 \times 10^{-3}$ | $9.726 \times 10^{-1}$ | $8.590 \times 10^{-2}$ | $\mathbf{9.460 \times 10^{-4}}$ |

*表4：相对$l_2$误差。*

### 5.3 热扩散PDE

一维热方程是典型的抛物型偏微分方程，用于建模热量等量在给定区域内的扩散。考虑具有周期边界条件的1D情况的以下一般形式：

$ \begin{cases} \frac{\partial u}{\partial t} = \alpha\frac{\partial^2 u}{\partial x^2}, & (t,x) \in [0,0.2] \times [0,1], \ u(x,0) = \sin(\pi x), \ u(0,t) = u(1,t) = 0. \end{cases} \tag{20} $

其中$\alpha$是称为介质热扩散率的正系数，这里为简单起见我们设置$\alpha = 1$。

参考解通过以下形式获得：

$ u_{reference} = e^{-\pi^2 t}\sin(\pi x). \tag{21} $

对于物理系统，我们从上述空间和时间域中采样5个时间网格（或时间戳）和101个空间网格。对于数据系统，我们使用HSG模块从网格空间为数据系统生成额外的2个时间戳（每个时间戳选择101个空间点）。我们在正向问题和预测问题中使用前4个时间步结合来自数据系统的额外2个时间戳进行训练，并在预测问题中使用最后的时间步进行测试。

**正向问题**：在正向问题中，我们调查不同方法对解推理的性能。图6显示了不同方法之间预测解的比较。我们可以发现三种方法都能在$t$增加时捕获解，而PINNsFormer和PhysicsSolver可以更好地捕获全局解。更多细节可以在表5中找到，表明PhysicsSolver在正向问题中的表现远优于PINNsFormer，这得益于其通过数据系统学习更好映射的能力，这提高了正向问题的精度。

**单步预测问题**：同样，我们旨在预测下一个时间步的解。图7比较了不同方法在最后和倒数第二个时间点获得的解。可以得出结论，PINNs无法准确预测未来解，因为它无法学习不同解之间的时间相关关系。当解随时间变化相对显著时，外推法无法准确预测未来解。此外，PINNsformer无法很好地捕获解，因为模型忽略了可用的数据信息。更多细节可以在表6中找到。我们可以观察到PhysicsSolver优于其他三种方法。

| 方法 | PINNs                  | PINNsFormer            | PhysicsSolver                   |
| ---- | ---------------------- | ---------------------- | ------------------------------- |
| 误差 | $1.000 \times 10^{-1}$ | $2.702 \times 10^{-2}$ | $\mathbf{4.622 \times 10^{-3}}$ |

*表5：相对$l_2$误差。*

| 方法 | 外推法                 | PINNs                  | PINNsFormer            | PhysicsSolver                   |
| ---- | ---------------------- | ---------------------- | ---------------------- | ------------------------------- |
| 误差 | $6.380 \times 10^{-1}$ | $3.241 \times 10^{-1}$ | $6.560 \times 10^{-2}$ | $\mathbf{1.070 \times 10^{-3}}$ |

*表6：相对$l_2$误差。*

### 5.4 二维Navier-Stokes PDE

Navier–Stokes方程在数学上表示牛顿流体的质量和动量守恒定律，它们将压力、速度、温度和密度联系起来。Navier–Stokes方程，无论是其完整形式还是简化形式，在飞机和汽车设计、血流研究、发电站开发、污染分析以及广泛的其他应用中都起着关键作用。当与麦克斯韦方程耦合时，它们也可用于建模和分析磁流体动力学。考虑以下抛物型PDE系统，它建模二维空间中不可压缩流体流动的行为。

$ \begin{cases} u_x + v_y = 0, \ u_t + \lambda_1[(u,v) \cdot \nabla u] = -p_x + \lambda_2\Delta u, \ v_t + \lambda_1[(u,v) \cdot \nabla v] = -p_y + \lambda_2\Delta v. \end{cases} \tag{22} $

其中$u(t,x,y)$是速度场$\mathbf{u}$的第一个分量，$v(t,x,y)$是速度场$\mathbf{u}$的第二个分量，即$\mathbf{u} = (u,v)$。$p(t,x,y)$是压力。$\nabla u = (\frac{\partial u}{\partial x}, \frac{\partial u}{\partial y})$，$\Delta u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}$，$\nabla v = (\frac{\partial v}{\partial x}, \frac{\partial v}{\partial y})$，$\Delta v = \frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2}$。$\lambda = (\lambda_1, \lambda_2)$是未知参数。这里，我们设置$\lambda_1 = 1$和$\lambda_2 = 0.01$。该系统没有解析解，我们使用[17]提供的仿真解作为参考解。在实践中，我们设置空间域$x \in [1,7]$，$y \in [-2,2]$，时间域$t \in [0,1]$。

对于物理系统，我们分别从上述空间和时间域中采样10个时间网格（或时间戳）和250个空间网格。对于数据系统，我们使用HSG模块从网格空间为数据系统生成额外的4个时间戳（每个时间戳选择250个空间点）。我们在正向和预测问题中使用前9个时间步结合来自数据系统的额外2个时间戳进行训练，并在预测问题中使用最后的时间步进行测试。

**正向问题**：在正向问题中，我们调查不同方法对压力$p$解推理的性能。图8显示了不同方法之间预测解的比较。我们可以发现PINNs方法和PINNsFormer方法都无法在不同时间戳很好地捕获解，而PhysicsSolver可以更好地捕获全局解。更多细节可以在表7中找到，表明PhysicsSolver在正向问题中的表现远优于PINNs和PINNsFormer，这得益于其通过数据系统学习更好映射的能力。

也可以得出结论，当解在不同时间戳间变化很小时，数据系统损失的纳入对正向问题更有效。

| 方法 | PINNs               | PINNsFormer         | PhysicsSolver                   |
| ---- | ------------------- | ------------------- | ------------------------------- |
| 误差 | $5.824 \times 10^0$ | $2.139 \times 10^0$ | $\mathbf{9.224 \times 10^{-3}}$ |

*表7：相对$l_2$误差。*

**单步预测问题**：在单步预测问题中，我们也旨在预测下一个时间步的解。图9比较了不同方法在最后和倒数第二个时间点获得的解。可以得出结论，当时间长度相对较大时，PINNs无法准确预测未来解，因为它无法很好地学习不同解之间的时间相关关系。同时，PINNsFormer无法很好地捕获解，因为它忽略了可用的数据信息。然而，外推法可以准确预测未来解，因为解随时间变化相对较小。更多细节可以在表8中找到。我们可以观察到PhysicsSolver优于其他两种方法，并与外推法具有相似的精度。

| 方法 | 外推法                          | PINNs               | PINNsFormer         | PhysicsSolver                   |
| ---- | ------------------------------- | ------------------- | ------------------- | ------------------------------- |
| 误差 | $\mathbf{7.437 \times 10^{-3}}$ | $7.796 \times 10^0$ | $1.912 \times 10^0$ | $\mathbf{1.384 \times 10^{-2}}$ |

*表8：相对$l_2$误差。*

## 6 结论与未来工作

在这项工作中，我们提出了一个名为PhysicsSolver的创新基于transformer的模型，它有效地解决了PDE系统中的正向和预测问题。与以前的方法不同，我们的模型可以通过物理系统和数据系统的集成同时学习内在物理信息并准确预测未来状态。我们进行了大量的数值实验，结果显示了我们提出方法的优越性。对于未来的工作，我们也将考虑PhysicsSolver的不同变体在解决更复杂问题（如多尺度动力学系统）中的应用。

## 致谢

刘柳感谢中国科技部国家重点研发计划(2021YFA1001200)、香港研究资助局早期职业计划(24301021)和一般研究基金(14303022 & 14301423)在2021-2023年期间的支持。

## 附录

### 经验损失

PINNs的经验损失为：

$ \mathcal{L}*{PINNs_empirical} = w_r\sum*{i=1}^{N_r}|\mathcal{L}[\hat{u}(t_i,\mathbf{x}_i,\mathbf{v}_i)] - f(t_i,\mathbf{x}_i,\mathbf{v}_i)|^2 \tag{23} $ $

- w_b\sum_{i=1}^{N_b}|\mathcal{B}[\hat{u}(t_i,\mathbf{x}_i,\mathbf{v}_i)] - g(t_i,\mathbf{x}_i,\mathbf{v}_i)|^2 $ $
- w_i\sum_{i=1}^{N_i}|\mathcal{I}[\hat{u}(t_i,\mathbf{x}_i,\mathbf{v}_i)] - h(t,\mathbf{x}_i,\mathbf{v}_i)|^2. $

其中$N_r, N_b$和$N_i$是用于残差损失、边界条件损失和初始条件损失计算的网格数量，$w_r, w_b$和$w_i$是相应的权重。

## 参考文献

[1] R. C. Aster, B. Borchers, and C. H. Thurber, Parameter estimation and inverse problems, Elsevier, 2018.

[2] K.-J. Bathe, Finite element procedures, Klaus-Jurgen Bathe, 2006.

[3] R. E. Caflisch, Monte carlo and quasi-monte carlo methods, Acta numerica, 7 (1998), pp. 1–49.

[4] S. Cao, Choose a transformer: Fourier or galerkin, Advances in neural information processing systems, 34 (2021), pp. 24924–24940.

[5] L. Dong, S. Xu, and B. Xu, Speech-transformer: a no-recurrence sequence-to-sequence model for speech recognition, in 2018 IEEE international conference on acoustics, speech and signal processing (ICASSP), IEEE, 2018, pp. 5884–5888.

[6] B. Fornberg, A practical guide to pseudospectral methods, no. 1, Cambridge university press, 1998.

[7] K. Han, Y. Wang, H. Chen, X. Chen, J. Guo, Z. Liu, Y. Tang, A. Xiao, C. Xu, Y. Xu, et al., A survey on vision transformer, IEEE transactions on pattern analysis and machine intelligence, 45 (2022), pp. 87–110.

[8] K. H. Huebner, D. L. Dewhirst, D. E. Smith, and T. G. Byrom, The finite element method for engineers, John Wiley & Sons, 2001.

[9] W. Jank, Quasi-monte carlo sampling to improve the efficiency of monte carlo em, Computational statistics & data analysis, 48 (2005), pp. 685–701.

[10] S. Jin, Z. Ma, and K. Wu, Asymptotic-preserving neural networks for multiscale kinetic equations, arXiv preprint arXiv:2306.15381, (2023).

[11] K. S. Kalyan, A. Rajasekharan, and S. Sangeetha, Ammus: A survey of transformer-based pretrained models in natural language processing, arXiv preprint arXiv:2108.05542, (2021).

[12] Z. Li, N. Kovachki, K. Azizzadenesheli, B. Liu, K. Bhattacharya, A. Stuart, and A. Anandkumar, Fourier neural operator for parametric partial differential equations, arXiv preprint arXiv:2010.08895, (2020).

[13] Z. Li, N. Kovachki, C. Choy, B. Li, J. Kossaifi, S. Otta, M. A. Nabian, M. Stadler, C. Hundt, K. Azizzadenesheli, et al., Geometry-informed neural operator for large-scale 3d pdes, Advances in Neural Information Processing Systems, 36 (2024).

[14] L. Liu, Y. Wang, X. Zhu, and Z. Zhu, Asymptotic-preserving neural networks for the semiconductor boltzmann equation and its application on inverse problems, arXiv preprint arXiv:2407.16169, (2024).

[15] L. Lu, P. Jin, G. Pang, Z. Zhang, and G. E. Karniadakis, Learning nonlinear operators via deeponet based on the universal approximation theorem of operators, Nature machine intelligence, 3 (2021), pp. 218–229.

[16] M. A. Rahman, Z. E. Ross, and K. Azizzadenesheli, U-no: U-shaped neural operators, arXiv preprint arXiv:2204.11127, (2022).

[17] M. Raissi, P. Perdikaris, and G. E. Karniadakis, Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations, Journal of Computational physics, 378 (2019), pp. 686–707.

[18] A. Shapiro, Monte carlo sampling methods, Handbooks in operations research and management science, 10 (2003), pp. 353–425.

[19] J. Shen, T. Tang, and L.-L. Wang, Spectral methods: algorithms, analysis and applications, vol. 41, Springer Science & Business Media, 2011.

[20] A. Sidi, Practical extrapolation methods: Theory and applications, vol. 10, Cambridge university press, 2003.

[21] G. A. Sod, A survey of several finite difference methods for systems of nonlinear hyperbolic conservation laws, Journal of computational physics, 27 (1978), pp. 1–31.

[22] N. Umetani and B. Bickel, Learning three-dimensional flow for interactive aerodynamic design, ACM Transactions on Graphics (TOG), 37 (2018), pp. 1–10.

[23] A. Vaswani, Attention is all you need, Advances in Neural Information Processing Systems, (2017).

[24] X. Wang and F. J. Hickernell, Randomized halton sequences, Mathematical and Computer Modelling, 32 (2000), pp. 887–899.

[25] Q. Wen, T. Zhou, C. Zhang, W. Chen, Z. Ma, J. Yan, and L. Sun, Transformers in time series: A survey, arXiv preprint arXiv:2202.07125, (2022).

[26] H. Wu, H. Luo, H. Wang, J. Wang, and M. Long, Transolver: A fast transformer solver for pdes on general geometries, arXiv preprint arXiv:2402.02366, (2024).

[27] Y. Yin, M. Kirchmeyer, J.-Y. Franceschi, A. Rakotomamonjy, and P. Gallinari, Continuous pde dynamics forecasting with implicit neural representations, arXiv preprint arXiv:2209.14855, (2022).

[28] B. Yu et al., The deep ritz method: a deep learning-based numerical algorithm for solving variational problems, Communications in Mathematics and Statistics, 6 (2018), pp. 1–12.

[29] Z. Zhao, X. Ding, and B. A. Prakash, Pinnsformer: A transformer-based framework for physics-informed neural networks, arXiv preprint arXiv:2307.11833, (2023).