```markdown

```

# PINNsFormer：一种基于Transformer的物理知情神经网络框架
赵致远  
佐治亚理工学院  
亚特兰大，GA 30332  
leozhao1997@gatech.edu  

丁雪颖  
卡内基梅隆大学  
匹兹堡，PA 15213  
xding2@andrew.cmu.edu  

B·阿迪蒂亚·普拉卡什  
佐治亚理工学院  
亚特兰大，GA 30332  
badityap@cc.gatech.edu  


## 摘要
物理知情神经网络（Physics-Informed Neural Networks, PINNs）已成为一种极具潜力的深度学习框架，用于逼近偏微分方程（PDEs）的数值解。然而，传统PINNs依赖多层感知机（MLP），忽略了实际物理系统中固有的关键时间依赖性，导致无法全局传播初始条件约束，也难以在多种场景下准确捕捉真实解。本文提出一种新颖的基于Transformer的框架——PINNsFormer，旨在解决这一局限。PINNsFormer利用多头注意力机制捕捉时间依赖性，从而准确逼近PDE解。该框架将逐点输入转换为伪序列，并将PINNs的逐点损失替换为序列损失。此外，PINNsFormer还引入了一种新的激活函数Wavelet，该函数通过深度神经网络预测傅里叶分解。实验结果表明，在包括PINNs失效场景和高维PDE在内的多种场景下，PINNsFormer均展现出更优的泛化能力和精度。同时，PINNsFormer能够灵活集成现有PINNs学习方案，进一步提升性能。

（

##### PINNsFormer 的新思路

PINNsFormer 是一个结合 **Transformer** 的新框架，核心改进有：

1. **引入注意力机制，捕捉时间依赖**
   - Transformer 的多头注意力能很好地建模“序列之间的关系”。
   - PINNsFormer 把网格点的输入转化成“伪序列”，让模型学到随时间的演化规律。
2. **逐点损失 → 序列损失**
   - 传统 PINNs 是对每个点单独计算误差。
   - PINNsFormer 改成对整个序列计算误差，更能体现时间上的整体一致性。
3. **新激活函数 Wavelet**
   - 传统激活函数（ReLU, Tanh 等）对物理规律的表达能力有限。
   - Wavelet 激活函数让网络能够更好地表示 **傅里叶分解**（相当于把函数拆成一堆正弦波），这样对 PDE 特别友好。

）


## 1 引言
偏微分方程（PDEs）的数值求解在科学与工程领域已得到广泛研究。传统方法（如有限元法（Bathe, 2007）、伪谱法（Fornberg, 1998））在为高维PDE构建网格时面临高昂的计算成本。随着科学机器学习的发展，物理知情神经网络（PINNs）（Lagaris et al., 1998; Raissi et al., 2019）成为一种极具潜力的新方法。传统PINNs及其多数变体采用多层感知机（MLP）作为端到端框架进行逐点预测，在多种场景下取得了显著成功。

然而，近年来的研究表明，当PDE解具有高频或多尺度特征时（Raissi, 2018; Fuks & Tchelepi, 2020; Krishnapriyan et al., 2021; Wang et al., 2022a），即使对应的解析解较为简单，PINNs也会失效。在这类场景中，PINNs往往给出过度平滑或粗略的逼近结果，与真实解存在偏差。

现有缓解PINNs失效的方法大致可分为两类策略：  
1. **数据插值策略**（Raissi et al., 2017; Zhu et al., 2019; Chen et al., 2021）：利用仿真或真实场景中观测到的数据进行正则化，但该类方法面临获取真实数据的挑战；  
2. **改进训练方案策略**（Mao et al., 2020; Krishnapriyan et al., 2021; Wang et al., 2021; 2022a）：这类方法在实际应用中可能存在计算成本过高的问题。例如，Krishnapriyan等人（2021）提出的Seq2Seq方法需要依次训练多个神经网络，而其他部分方法则因误差累积面临收敛问题；神经正切核（Neural Tangent Kernel, NTK）（Wang et al., 2022a）需要构建维度为$K \in \mathbb{R}^{D \times P}$的核矩阵（其中$D$为样本量，$P$为模型参数），当样本量或模型参数增加时，会面临可扩展性问题。

尽管大多数改进PINNs泛化能力、解决其失效问题的研究都聚焦于上述方面，但传统PINNs（主要依赖MLP架构）往往会忽略真实物理系统中重要的时间依赖性。例如，有限元法通过全局解的逐次传播，隐式地融入了时间依赖性——这种传播基于“$t+\Delta t$时刻的状态依赖于$t$时刻的状态”这一原理。与之相反，PINNs作为一种逐点（point-to-point）框架，并未显式建模PDE中的时间依赖性。忽略时间依赖性会导致PINNs难以全局传播初始条件约束，进而引发失效问题：在初始条件附近的逼近结果仍能保持准确，但随后会逐渐退化为过度平滑或粗略的逼近。

为解决PINNs中时间依赖性被忽略的问题，一个自然的思路是采用基于Transformer的模型——这类模型通过多头自注意力和编码器-解码器注意力（Vaswani et al., 2017），在捕捉序列数据的长期依赖性方面表现突出，其变体已在多个领域取得显著成功。然而，将原本为序列数据设计的Transformer适配到PINNs的逐点框架中，面临着非 trivial 的挑战，这些挑战涉及框架内的数据表示和正则化损失两个层面。

### 主要贡献
本文提出PINNsFormer——一种基于Transformer架构的新颖序列到序列（sequence-to-sequence）PDE求解器。据我们所知，PINNsFormer是PINNs领域内首个明确聚焦并学习PDE中时间依赖性的框架。主要贡献总结如下：  
1. **新框架**：提出一种新颖且直观的基于Transformer的框架PINNsFormer。该框架通过生成的伪序列为PINNs赋予捕捉时间依赖性的能力，从而提升其泛化能力和逼近精度，有效求解PDE；  
2. **新激活函数**：引入一种新的非线性激活函数Wavelet。Wavelet旨在对任意目标信号的傅里叶变换进行预测，使其成为无限宽度神经网络的通用逼近器。Wavelet还有望对不同模型架构下的多种深度学习任务产生积极作用；  
3. **大量实验**：在多种场景下对PINNsFormer进行全面评估。验证了其在处理PINNs失效场景、求解高维PDE时，在优化和逼近精度方面的优势；同时证明了PINNsFormer集成不同PINNs学习方案的灵活性和有效性。


## 2 相关工作
### 物理知情神经网络（PINNs）
物理知情神经网络（PINNs）已成为解决科学与工程问题的重要方法。Raissi等人（2019）提出将物理定律融入神经网络训练以求解PDE的框架，该工作已在流体动力学、固体力学、量子力学等多个领域得到应用（Carleo et al., 2019; Yang et al., 2020）。研究人员还探索了PINNs的不同学习方案（Mao et al., 2020; Wang et al., 2021; 2022a），在收敛性、泛化性和可解释性方面取得了显著提升。

### PINNs的失效模式
尽管PINNs潜力巨大，但近年来的研究表明其存在固有失效模式，尤其在处理具有高频或多尺度特征的PDE时（Fuks & Tchelepi, 2020; Raissi, 2018; McClenny & Braga-Neto, 2020; Krishnapriyan et al., 2021; Zhao et al., 2022; Wang et al., 2022a）。研究人员从多个角度应对这一挑战，包括设计不同模型架构、改进学习方案或采用数据插值（Han et al., 2018; Lou et al., 2021; Wang et al., 2021; 2022a; b）。深入理解PINNs的局限性及潜在失效模式，是其应用于复杂物理问题的基础。

### 基于Transformer的模型
Transformer模型（Vaswani et al., 2017）因其捕捉长期依赖性的能力受到广泛关注，并在自然语言处理任务中取得重大突破（Devlin et al., 2018; Radford et al., 2018）。Transformer还被扩展到计算机视觉、语音识别、时间序列分析等其他领域（Liu et al., 2021; Dosovitskiy et al., 2020; Gulati et al., 2020; Zhou et al., 2021）。研究人员还开发了提升Transformer效率的技术，如稀疏注意力和模型压缩（Child et al., 2019; Sanh et al., 2019）。


## 3 方法
### 预备知识
设$\Omega$为$\mathbb{R}^d$中的开集，边界为$\partial\Omega \in \mathbb{R}^{d-1}$。包含空间输入$x$和时间输入$t$的PDE通常可抽象为以下形式：  
$$
\begin{cases}
\mathcal{D}[u(x, t)] = f(x, t), \quad \forall x, t \in \Omega \\
\mathcal{B}[u(x, t)] = g(x, t), \quad \forall x, t \in \partial\Omega
\end{cases} \tag{1}
$$
其中，$u$为PDE的解，$\mathcal{D}$为正则化系统行为的微分算子，$\mathcal{B}$为边界条件或初始条件的通用描述。具体而言，$\{x, t\} \in \Omega$为残差点，$\{x, t\} \in \partial\Omega$为边界/初始点。设$\hat{u}$为神经网络的逼近结果，则PINNs框架通过以下约束对$\hat{u}$进行经验正则化：  
$$
\mathcal{L}_{\text{PINNs}} = \lambda_r \sum_{i=1}^{N_r} \|\mathcal{D}[\hat{u}(x, t)] - f(x, t)\|^2 + \lambda_b \sum_{i=1}^{N_b} \|\mathcal{B}[\hat{u}(x, t)] - g(x, t)\|^2 \tag{2}
$$
其中，$N_b$、$N_r$分别表示残差点和边界/初始点的数量，$\lambda_r$、$\lambda_b$为平衡各损失项权重的正则化参数。神经网络$\hat{u}$以向量化的$\{x, t\}$为输入，输出逼近解。训练目标是通过机器学习方法优化神经网络$\hat{u}$，使式(2)中的损失最小化。

### 方法概述
PINNs聚焦于逐点预测，但对真实物理系统中时间依赖性的探索仍显不足。传统PINNs方法仅利用单对空间信息$x$和时间信息$t$逼近数值解$u(x, t)$，未考虑前后时间步间的时间依赖性。然而，这种简化仅适用于椭圆型PDE（未知函数与其导数的关系不显式包含时间）；对于双曲型和抛物型PDE，其包含时间导数，意味着某一时刻的状态会影响前后时刻的状态。因此，考虑时间依赖性对于利用PINNs有效求解这类PDE至关重要。

本节介绍一种基于Transformer的PINNs新框架——PINNsFormer。与逐点预测不同，PINNsFormer将PINNs的能力扩展到序列预测：既能准确逼近特定时间步的解，又能学习和正则化输入状态间的时间依赖性。该框架包含四个核心组件：伪序列生成器（Pseudo Sequence Generator）、时空混合器（Spatio-Temporal Mixer）、带多头注意力的编码器-解码器（Encoder-Decoder with multi-head attention）和输出层（Output Layer）。此外，本文还引入一种新的激活函数Wavelet，该函数采用实傅里叶变换（Real Fourier Transform）技术预测PDE的解。框架结构如图1所示，下文将详细解释各组件及学习方案。


### 3.1 伪序列生成器
Transformer及基于Transformer的模型旨在捕捉序列数据的长期依赖性，而传统PINNs以非序列数据作为神经网络输入。因此，要将PINNs与Transformer结合，需先将逐点时空输入转换为时间序列。对于给定的空间输入$x \in \mathbb{R}^{d-1}$和时间输入$t \in \mathbb{R}$，伪序列生成器执行以下操作：  
$$
[x, t] \xrightarrow{\text{生成器}} \left\{ [x, t], [x, t+\Delta t], \dots, [x, t+(k-1)\Delta t] \right\} \tag{3}
$$
其中，$[\cdot]$表示拼接操作，即$[x, t] \in \mathbb{R}^d$为向量化形式，生成器输出形状为$\mathbb{R}^{k \times d}$的伪序列。伪序列生成器通过将单个时空输入扩展为多个等距离散时间步，外推得到时间序列。

$k$和$\Delta t$为超参数：$k$直观决定伪序列需“前瞻”的步数，$\Delta t$决定每步的“跨度”。在实际应用中，$k$和$\Delta t$不宜设置过大——较大的$k$会导致沉重的计算和内存开销，而较大的$\Delta t$可能破坏相邻离散时间步的时间依赖关系。

（

假设我们有一个输入：

- 空间位置 $x$ （比如某个点的坐标）；
- 时间 $t$ （比如 1 秒时刻）。

**伪序列生成器** 会把它扩展成一段“时间片段”：
$$
[x, t] \;\;\rightarrow\;\; \{ [x, t], [x, t+\Delta t], [x, t+2\Delta t], \dots, [x, t+(k-1)\Delta t] \}
$$
简单来说：

- 从同一个空间点出发，
- 沿着时间轴，往前“多看几个时刻”，
- 把它们拼成一个小序列。

这样，原来的一点点信息就变成了一段“短视频片段”。

）


### 3.2 模型架构
除伪序列生成器外，PINNsFormer的架构还包含三个组件：时空混合器、带多头注意力的编码器-解码器和输出层。其中，输出层为简单的全连接MLP，下文重点介绍前两个组件。值得注意的是，PINNsFormer仅依赖线性层和非线性激活函数，避免了卷积或循环等复杂操作，确保了实际应用中的计算效率。

#### 时空混合器
多数PDE包含低维空间或时间信息，直接将低维数据输入编码器可能无法捕捉各特征维度间的复杂关系。因此，需将原始序列数据嵌入到更高维空间，使每个向量能编码更多信息。

与传统Transformer将原始数据嵌入高维空间（向量间距离反映语义相似性）（Vaswani et al., 2017; Devlin et al., 2018）不同，PINNsFormer通过全连接MLP构建线性投影，将时空输入映射到高维空间。嵌入后的数据通过融合所有原始时空特征，增强了信息表达能力，这种线性投影被称为**时空混合器**。

#### 编码器-解码器架构
PINNsFormer采用与Transformer类似的编码器-解码器架构：  
- **编码器**：由多个相同层堆叠而成，每层包含一个编码器自注意力层（encoder self-attention layer）和一个前馈层（feedforward layer）；  
- **解码器**：与标准Transformer略有不同，其每层仅包含一个编码器-解码器注意力层（encoder-decoder self-attention layer）和一个前馈层。  

在解码器层面，PINNsFormer使用与编码器相同的时空嵌入，因此解码器无需为相同的输入嵌入重新学习依赖性。编码器-解码器架构如图2所示。

（图2：PINNsFormer编码器-解码器层架构。解码器未配备自注意力。）

从直观上看，编码器自注意力使模型能够学习所有时空信息的依赖关系；解码器的编码器-解码器注意力则允许模型在解码过程中选择性聚焦输入序列中的特定依赖关系，从而比传统PINNs捕捉更多信息。由于PINNs的核心是逼近当前状态的解（与语言任务或时间序列预测中的“下一状态预测”不同），因此编码器和解码器使用相同的嵌入。

（

##### 1. 时空混合器（Spatio-Temporal Mixer）

- 在 PDE 任务里，原始输入通常是 **低维的**（比如 2D/3D 的坐标 + 时间）。
- 直接把这些低维输入丢给 Transformer，模型可能学不到它们之间复杂的关系。
- 于是 PINNsFormer 先用一个 **线性投影**（全连接层 + 非线性激活）把它们映射到 **高维空间**。
- 这个高维向量融合了所有时空特征，可以更丰富地表达物理信息。

打个比方：

- 原始输入就像一张“身份证号” → 信息有限。
- 时空混合器相当于把它翻译成一份“完整档案”，包含更多维度的信息（姓名、性别、出生地、学历……）。

------

##### 2. 编码器-解码器架构

PINNsFormer 延续了 Transformer 的经典框架，但做了一些针对 PDE 的改动：

- **编码器（Encoder）**：
  - 有多层，每层包含：
    1. **自注意力**：学习输入序列内部（所有时空点之间）的依赖关系；
    2. **前馈层**：进一步处理和变换特征。
- **解码器（Decoder）**：
  - 和标准 Transformer 不同，它没有“自注意力”，只有：
    1. **编码器-解码器注意力**：让解码器在生成输出时，可以有选择地关注输入序列中的特定部分；
    2. **前馈层**。
  - 解码器和编码器用的是 **同一套时空嵌入**，所以不需要重复学习。

）


### 3.3 Wavelet激活函数
Transformer通常采用LayerNorm和ReLU非线性激活函数（Vaswani et al., 2017; Gehring et al., 2017; Devlin et al., 2018），但这些激活函数在PINNs求解中可能并不适用。例如，在PINNs中使用ReLU激活会导致性能不佳——PINNs的有效性高度依赖导数的准确计算，而ReLU的导数存在不连续性（Haghighat et al., 2021; de Wolff et al., 2021）。近年来的研究针对特定场景采用正弦（Sin）激活，以模拟PDE解的周期性（Li et al., 2020; Jagtap et al., 2020; Song et al., 2022），但这种方法需要对解的行为有较强的先验知识，适用范围有限。

为解决这一问题，本文提出一种新颖且简洁的激活函数**Wavelet**，定义如下：  
$$
\text{Wavelet}(x) = \omega_1 \sin(x) + \omega_2 \cos(x) \tag{4}
$$
其中，$\omega_1$和$\omega_2$为可学习的注册参数。Wavelet激活函数的设计灵感源于实傅里叶变换：周期信号可分解为多个频率正弦函数的积分，而所有信号（无论周期与否）均可分解为不同频率正弦和余弦函数的积分。显然，Wavelet具备逼近任意函数的能力（足够的逼近能力），由此可得到以下命题：

**命题1** 设$N$为具有无限宽度的双隐藏层神经网络，且配备Wavelet激活函数，则$N$是任意实值目标函数$f$的通用逼近器。

**证明概要** 证明过程基于实傅里叶变换（傅里叶积分变换）：对于任意输入$x$及其对应的实值目标$f(x)$，其傅里叶积分为：  
$$
f(x) = \int_{-\infty}^{\infty} F_c(\omega) \cos(\omega x) d\omega + \int_{-\infty}^{\infty} F_s(\omega) \sin(\omega x) d\omega
$$
其中，$F_c$和$F_s$分别为正弦项和余弦项的系数。其次，通过黎曼和逼近，积分可由无限和近似表示：  
$$
f(x) \approx \sum_{n=1}^{N} \left[ F_c(\omega_n) \cos(\omega_n x) + F_s(\omega_n) \sin(\omega_n x) \right] \equiv W_2(\text{Wavelet}(W_1 x))
$$
其中，$W_1$和$W_2$分别为$N$的第一隐藏层和第二隐藏层的权重。由于$W_1$和$W_2$具有无限宽度，可将分段求和划分为无限小的区间，使逼近结果能任意接近真实积分。因此，$N$是任意给定$f$的通用逼近器。

在实际应用中，多数PDE解仅包含有限个主要频率，因此使用有限参数的神经网络也能对真实解进行有效逼近。尽管Wavelet激活函数在本文中主要用于PINNsFormer以改进PINNs，但它也可能在其他深度学习任务中具有应用潜力。与ReLU、$\sigma(\cdot)$、Tanh等激活函数（均能使无限宽度双隐藏层神经网络成为通用逼近器）（Cybenko, 1989; Hornik, 1991; Glorot et al., 2011）类似，我们预计Wavelet在本文研究范围之外的其他应用中也能展现有效性。

（

### 新方案：Wavelet 激活函数

作者提出了一种新的激活函数：
$$
\text{Wavelet}(x) = \omega_1 \sin(x) + \omega_2 \cos(x)
$$

- 它同时结合了 **正弦** 和 **余弦** 两部分。
- 系数 $\omega_1, \omega_2$ 是可学习的参数，网络会自己调整它们。

灵感来自 **傅里叶变换**：

- 任意信号都可以拆解成一堆不同频率的正弦和余弦叠加。
- 所以只要学会足够多的 sin 和 cos，就能逼近任何函数。

------

### 为什么强大？

作者提出一个命题：

- 如果一个神经网络足够大，并且用 Wavelet 作为激活函数，**它就能逼近任意函数**。
- 证明方法：傅里叶积分展开 + 黎曼和近似。换句话说，Wavelet 就是个万能“拼图块”，能拼出任意复杂曲线。

在实际应用中：

- 多数 PDE 解只涉及有限几个主要频率。
- 所以即便网络不是无限大，Wavelet 激活函数也能很好地逼近真实解。

）


### 3.4 学习方案
传统PINNs聚焦于逐点预测，而将PINNs适配到伪序列输入的研究尚未开展。在PINNsFormer中，序列中的每个生成点（即$[x_i, t_i+j\Delta t]$）均映射到对应的逼近结果（即对任意$j \in \mathbb{N}, j<k$，$\hat{u}(x_i, t_i+j\Delta t)$）。这种方法允许独立计算任意有效阶数$n$下，关于$x$或$t$的$n$阶导数。例如，对于给定的输入伪序列$\{[x_i, t_i], [x_i, t_i+\Delta t], \dots, [x_i, t_i+(k-1)\Delta t]\}$及其对应的逼近结果$\{\hat{u}(x_i, t_i), \hat{u}(x_i, t_i+\Delta t), \dots, \hat{u}(x_i, t_i+(k-1)\Delta t)\}$，可分别计算关于$x$和$t$的一阶导数：  
$$
\begin{cases}
\frac{\partial \{\hat{u}(x_i, t_i+j\Delta t)\}_{j=0}^{k-1}}{\partial \{\hat{t}_i+j\Delta t\}_{j=0}^{k-1}} = \left\{ \frac{\partial \hat{u}(x_i, t_i)}{\partial t_i}, \frac{\partial \hat{u}(x_i, t_i+\Delta t)}{\partial (t_i+\Delta t)}, \dots, \frac{\partial \hat{u}(x_i, t_i+(k-1)\Delta t)}{\partial (t_i+(k-1)\Delta t)} \right\} \\
\frac{\partial \{\hat{u}(x_i, t_i+j\Delta t)\}_{j=0}^{k-1}}{\partial x_i} = \left\{ \frac{\partial \hat{u}(x_i, t_i)}{\partial x_i}, \frac{\partial \hat{u}(x_i, t_i+\Delta t)}{\partial x_i}, \dots, \frac{\partial \hat{u}(x_i, t_i+(k-1)\Delta t)}{\partial x_i} \right\}
\end{cases} \tag{5}
$$

这种计算“序列逼近结果关于序列输入的导数”的方案，可轻松扩展到更高阶导数，且适用于残差点、边界点和初始点。但与式(2)中PINNs的通用优化目标（融合初始条件和边界条件目标）不同，PINNsFormer通过学习方案区分二者，并对初始条件和边界条件应用不同的正则化策略：  
- 对于残差点和边界点：所有序列输出均可通过PINNs损失进行正则化。这是因为所有生成的伪时间步均与原始输入处于同一域中（例如，若$[x_i, t_i]$采样自边界，则对任意$j \in \mathbb{N}^+$，$[x_i, t_i+j\Delta t]$也位于边界上）；  
- 对于初始点：仅正则化$t=0$的条件（对应序列输出的第一个元素）。这是因为伪序列中仅第一个元素精确匹配$t=0$的初始条件，其他生成时间步（$t=j\Delta t, j \in \mathbb{N}^+$）均超出初始条件范围。

基于上述考虑，将PINNs损失适配为序列版本，具体如下：  
$$
\begin{cases}
\mathcal{L}_{\text{res}} = \frac{1}{kN_{\text{res}}} \sum_{i=1}^{N_{\text{res}}} \sum_{j=0}^{k-1} \|\mathcal{D}[\hat{u}(x_i, t_i+j\Delta t)] - f(x_i, t_i+j\Delta t)\|^2 \\
\mathcal{L}_{\text{bc}} = \frac{1}{kN_{\text{bc}}} \sum_{i=1}^{N_{\text{bc}}} \sum_{j=0}^{k-1} \|\mathcal{B}[\hat{u}(x_i, t_i+j\Delta t)] - g(x_i, t_i+j\Delta t)\|^2 \\
\mathcal{L}_{\text{ic}} = \frac{1}{N_{\text{ic}}} \sum_{i=1}^{N_{\text{bc}}} \|\mathcal{I}[\hat{u}(x_i, 0)] - h(x_i, 0)\|^2 \\
\mathcal{L}_{\text{PINNsFormer}} = \lambda_{\text{res}}\mathcal{L}_{\text{res}} + \lambda_{\text{ic}}\mathcal{L}_{\text{ic}} + \lambda_{\text{bc}}\mathcal{L}_{\text{bc}}
\end{cases} \tag{6}
$$
其中，$N_{\text{res}} = N_r$（与式(2)中的残差点数量一致），$N_{\text{bc}}$、$N_{\text{ic}}$分别表示边界点和初始点的数量（满足$N_{\text{bc}} + N_{\text{ic}} = N_b$）；$\lambda_{\text{res}}$、$\lambda_{\text{bc}}$、$\lambda_{\text{ic}}$为正则化权重，用于平衡PINNsFormer中各损失项的重要性（与PINNs损失类似）。

训练阶段，PINNsFormer将所有残差点、边界点和初始点输入模型，得到对应的序列逼近结果；随后使用基于梯度的优化算法（如L-BFGS或Adam）优化式(6)中的$\mathcal{L}_{\text{PINNsFormer}}$，更新模型参数直至收敛。测试阶段，PINNsFormer将任意$\{x, t\}$对输入模型，得到序列逼近结果，其中序列逼近结果的第一个元素即为$\hat{u}(x, t)$的期望输出值。

（

### 怎么做？

1. **伪序列输入**：
   - 输入：${[x, t], [x, t+\Delta t], \dots, [x, t+(k-1)\Delta t]}$
   - 输出：${\hat{u}(x, t), \hat{u}(x, t+\Delta t), \dots, \hat{u}(x, t+(k-1)\Delta t)}$
2. **导数计算更灵活**：
   - 传统 PINNs 只能在某个点上求导。
   - PINNsFormer 可以在整个序列上对 $x$ 和 $t$ 求导（一阶、二阶都行），更容易适配 PDE 的方程形式。
3. **不同类型点的区别对待**：
   - **残差点 / 边界点**：
     - 这些点的伪序列里，每一个时间步都还是有效的（比如边界点始终在边界上）。
     - 所以序列里的所有输出都要满足 PDE 的残差约束或边界条件。
   - **初始点**：
     - 初始条件只在 $t=0$ 那一刻成立。
     - 所以伪序列里，只有第一个输出需要满足初始条件，其余的点已经超出了初始条件范围，不用强行约束。

）


### 3.5 损失景观分析
（图3：PINNs（左）和PINNsFormer（右）的损失景观可视化（对数尺度）。PINNsFormer的损失景观比传统PINNs平滑得多。）

尽管为基于Transformer的模型建立理论收敛性或泛化界具有挑战性，但评估优化轨迹的一种替代方法是**损失景观可视化**——这种方法已用于Transformer和PINNs的分析（Krishnapriyan et al., 2021; Yao et al., 2020; Park & Kim, 2022）。损失景观通过沿Hessian前两个主特征向量方向扰动训练后的模型构建，这种方法比随机参数扰动更具信息量。通常，损失景观越平滑、局部极小值越少，模型越容易收敛到全局极小值。本文对PINNs和PINNsFormer的损失景观进行了可视化（如图5所示）。

可视化结果清晰表明，PINNs的损失景观比PINNsFormer更复杂。具体而言，我们估计了两种损失景观的Lipschitz常数：$\mathcal{L}_{\text{PINNs}} = 776.16$，远大于$\mathcal{L}_{\text{PINNsFormer}} = 32.79$。此外，PINNs的损失景观在最优解附近存在多个尖锐“锥状”区域，表明在收敛点（零扰动）附近存在多个局部极小值。传统PINNs粗糙的损失景观和大量局部极小值表明，优化式(6)中的PINNsFormer目标函数能更轻松地收敛到全局极小值，这意味着PINNsFormer在避免PINNs失效模式方面具有优势。下一节的实验结果将进一步验证这一分析。


## 4 实验
### 4.1 实验设置
#### 目标
实验评估旨在验证PINNsFormer的三个关键优势：  
1. 与PINNs及变体架构相比，PINNsFormer能提升泛化能力并缓解失效模式；  
2. PINNsFormer能灵活集成多种学习方案，从而实现更优性能；  
3. 在求解高维PDE（PINNs及其变体面临挑战的场景）时，PINNsFormer收敛更快、泛化能力更强。

#### 实验配置
实验评估基于四类PDE：对流方程（convection）、1D反应方程（1D-reaction）、1D波动方程（1D-wave）和纳维-斯托克斯方程（Navier–Stokes）——这些PDE的设置均遵循前期研究（Raissi et al., 2019; Krishnapriyan et al., 2021; Wang et al., 2022a），以确保公平比较。选取的基线模型包括PINNs、QRes（Bu & Karpatne, 2021）和First-Layer Sine（FLS）（Wong et al., 2022）。

- 对于对流方程、1D反应方程和1D波动方程：均匀采样$N_{\text{ic}} = N_{\text{bc}} = 101$个初始点和边界点，残差域采用$101 \times 101$网格（共$N_{\text{res}} = 10201$个残差点）。训练PINNsFormer时，减少配点数量：$N_{\text{ic}} = N_{\text{bc}} = 51$个初始点和边界点，残差域采用$51 \times 51$网格——减少训练样本量有两个目的：提升训练效率，以及验证PINNsFormer在有限训练数据下的泛化能力。测试阶段，残差域采用$101 \times 101$网格。  
- 对于纳维-斯托克斯方程：从3D网格的残差域中采样2500个点用于训练；评估指标为预测最终时间步$t=20.0$的压力值。

#### 评估方式
为突出PINNsFormer的优势源于其捕捉时间依赖性的能力（而非单纯依赖模型过参数化），所有基线模型和PINNsFormer的参数数量保持大致相近。所有模型均使用带Strong Wolfe线性搜索的L-BFGS优化器训练1000轮。为简化实验，式(6)优化目标中设置$\lambda_{\text{res}} = \lambda_{\text{ic}} = \lambda_{\text{bc}} = 1$。详细超参数见附录A；激活函数的消融实验和$\{k, \Delta t\}$的超参数敏感性分析见附录C。

评估指标采用相关研究中常用的相对平均绝对误差（rMAE，即相对$\ell_1$误差）和相对均方根误差（rRMSE，即相对$\ell_2$误差）（Krishnapriyan et al., 2021; Raissi et al., 2019; McClenny & Braga-Neto, 2020），指标的详细公式见附录A。

#### 可复现性
所有模型均基于PyTorch（Paszke et al., 2019）实现，在单块NVIDIA Tesla V100 GPU上独立训练。所有代码和演示可在以下仓库获取并复现：https://github.com/AdityaLab/pinnsformer。


### 4.2 缓解PINNs的失效模式
主要评估聚焦于验证PINNsFormer相比PINNs的泛化能力优势，尤其在已知会挑战PINNs泛化能力的PDE上。实验选择两类PDE：对流方程和1D反应方程——这些方程对传统基于MLP的PINNs构成显著挑战，常导致“PINNs失效模式”（Mojgani et al., 2022; Daw et al., 2022; Krishnapriyan et al., 2021）。在失效模式下，优化过程会陷入局部极小值，导致逼近结果过度平滑，与真实解偏差较大。

评估目标是验证PINNsFormer相比标准PINNs及其变体，在缓解PINNs失效模式方面的泛化能力提升。评估结果总结于表1（PDE的详细公式见附录B）；PINNs和PINNsFormer在对流方程上的预测结果和绝对误差图如图4所示（所有预测图见附录C）。

| 模型        | 对流方程     |       |       | 1D反应方程   |       |       |
| ----------- | ------------ | ----- | ----- | ------------ | ----- | ----- |
|             | 损失（Loss） | rMAE  | rRMSE | 损失（Loss） | rMAE  | rRMSE |
| PINNs       | 0.016        | 0.778 | 0.840 | 0.199        | 0.982 | 0.981 |
| QRes        | 0.015        | 0.746 | 0.816 | 0.199        | 0.979 | 0.977 |
| FLS         | 0.012        | 0.674 | 0.771 | 0.199        | 0.984 | 0.985 |
| PINNsFormer | 3.7e-5       | 0.023 | 0.027 | 3.0e-6       | 0.015 | 0.030 |

**表1：对流方程和1D反应方程的求解结果。PINNsFormer在训练损失、rMAE和rRMSE上均持续优于所有基线模型。**

评估结果表明，在两种场景下，PINNsFormer均显著优于所有基线模型：不仅实现了最低的训练损失和测试误差，也是唯一能缓解失效模式的方法。相比之下，其他基线模型均陷入全局极小值，无法有效优化目标损失。这些结果验证了PINNsFormer相比传统PINNs及现有变体，在泛化能力和逼近精度上的显著优势。

（图4：PINNs（上）和PINNsFormer（下）在对流方程上的预测结果（左）和绝对误差（右）。相比PINNs，PINNsFormer成功缓解了失效模式。）

关于PINNsFormer的一个额外考量是其相比PINNs的计算和内存开销。基于MLP的PINNs以高效著称，而PINNsFormer采用基于Transformer的架构处理序列数据，自然会产生更高的计算和内存成本。然而，实验评估表明，这种开销是可接受的——得益于PINNsFormer仅依赖线性层，避免了卷积或循环等复杂操作。例如，当伪序列长度$k=5$时，计算成本约为PINNs的2.92倍，内存占用约为PINNs的2.15倍（详细数据见附录A）。这种开销与PINNsFormer带来的显著性能提升相比，是合理的。


### 4.3 集成多种学习方案的灵活性
| 模型              | 1D波动方程   |       |       |
| ----------------- | ------------ | ----- | ----- |
|                   | 损失（Loss） | rMAE  | rRMSE |
| PINNs             | 1.93e-2      | 0.326 | 0.335 |
| PINNsFormer       | 1.38e-2      | 0.270 | 0.283 |
| PINNs + NTK       | 6.34e-3      | 0.140 | 0.149 |
| PINNsFormer + NTK | 4.21e-3      | 0.054 | 0.058 |

**表2：1D波动方程的求解结果（集成NTK方法）。PINNsFormer与NTK结合的模型在所有指标上均优于其他方法。**

尽管PINNs及其各种架构变体在某些场景下面临挑战，但前期研究已探索出多种复杂的优化方案来缓解这些问题，包括学习率退火（Wang et al., 2021）、增广拉格朗日方法（Lu et al., 2021）和神经正切核（NTK）方法（Wang et al., 2022a）。这些改进的PINNs在特定场景下表现出显著提升。值得注意的是，当这些优化策略应用于PINNsFormer时，可轻松集成以进一步提升性能。例如，将NTK方法应用于PINNs已被证明能有效求解1D波动方程；本文验证，将NTK与PINNsFormer结合，可进一步提升逼近精度。详细结果如表2所示（PDE的详细公式见附录B，预测图见附录C）。

评估结果验证了PINNsFormer与NTK方法结合的灵活性和有效性：性能呈现逐步提升的趋势——从标准PINNs到PINNsFormer，从PINNs+NTK到PINNsFormer+NTK。本质上，PINNsFormer是PINNs的一种架构变体，而多数学习方案从优化角度设计，与神经网络架构无关。这种固有灵活性使PINNsFormer能与多种学习方案灵活结合，为真实应用中PDE的准确求解提供实用且可定制的解决方案。


### 4.4 在高维PDE上的泛化能力
（图5：PINNs和PINNsFormer在纳维-斯托克斯方程上的训练损失随迭代次数变化曲线。）

在前几节中，我们验证了PINNsFormer在缓解PINNs失效模式方面的泛化能力优势，但这些实验中的PDE解析解通常较为简单。在实际物理系统中，需求解更高维和更复杂的PDE。因此，评估PINNsFormer在这类高维PDE上的泛化能力至关重要——尤其是当PINNsFormer配备自注意力等先进机制时。

实验基于Raissi等人（2019）的既定设置，评估PINNsFormer与PINNs在纳维-斯托克斯方程上的性能。训练损失如图5所示，结果如表3所示（2D纳维-斯托克斯方程的详细公式见附录B，预测图见附录C）。

| 模型        | 纳维-斯托克斯方程 |       |       |
| ----------- | ----------------- | ----- | ----- |
|             | 损失（Loss）      | rMAE  | rRMSE |
| PINNs       | 6.72e-5           | 13.08 | 9.08  |
| QRes        | 2.24e-4           | 6.41  | 4.45  |
| FLS         | 9.54e-6           | 3.98  | 2.77  |
| PINNsFormer | 6.66e-6           | 0.384 | 0.280 |

**表3：纳维-斯托克斯方程的求解结果。PINNsFormer在所有指标上均优于所有基线模型。**

评估结果表明，在高维PDE上，PINNsFormer相比PINNs具有显著优势：  
1. 无论是训练损失还是验证误差，PINNsFormer均优于PINNs及其MLP变体；  
2. PINNsFormer在训练过程中收敛速度显著更快，这在一定程度上补偿了其每轮迭代更高的计算成本；  
3. PINNs及其MLP变体虽能预测出压力的大致形状，但随着时间推移，预测值与真实值的幅值偏差会逐渐增大；而PINNsFormer在不同时间区间内，均能保持预测压力的形状和幅值与真实值一致——这种一致性源于PINNsFormer通过基于Transformer的架构和自注意力机制捕捉时间依赖性的能力。


## 5 结论
本文提出一种基于Transformer的PINNs新框架PINNsFormer，旨在捕捉PDE解逼近过程中的时间依赖性。该框架引入伪序列生成器，将向量化输入转换为伪时间序列；并结合改进的编码器-解码器层和新的Wavelet激活函数。实验评估表明，在多种场景下（包括处理PINNs失效模式、求解高维PDE、集成不同PINNs学习方案），PINNsFormer均持续优于传统PINNs。此外，PINNsFormer保持了计算简洁性，使其成为真实应用中的实用选择。

除PINNsFormer外，Wavelet激活函数也为更广泛的机器学习社区提供了潜力。本文通过概要证明，基于解的傅里叶分解，配备Wavelet的双隐藏层无限宽度神经网络能够逼近任意目标解。我们鼓励从理论和实验角度进一步探索Wavelet激活函数的潜力——其应用范围不仅限于PINNs，还可扩展到多种架构和任务中。


## 致谢
本研究部分得到以下资助：美国国家科学基金会（NSF，资助号：Expeditions CCF-1918770、CAREER IIS-2028586、Medium IIS-1955883、Medium IIS-2106961、PIPP CCF-2200269）、美国疾病控制与预防中心（CDC）MInD项目、Meta教师捐赠基金，以及佐治亚理工学院（Georgia Tech）和乔治亚理工研究院（GTRI）提供的资金和计算资源。

（注：原文中“Published as a conference paper at ICLR 2024”为会议发表标识，已在对应章节标题后保留原文格式，确保学术完整性。）