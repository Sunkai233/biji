```markdown

```

# PDE-Transformer：用于物理模拟的高效通用Transformer
Benjamin Holzschuh  
Qiang Liu  
Georg Kohl  
Nils Thuerey  


## 摘要
本文提出PDE-Transformer，一种基于Transformer的改进架构，用于规则网格上物理模拟的代理建模（surrogate modeling）。我们将扩散Transformer（diffusion transformers）的最新架构改进与针对大规模模拟的特定调整相结合，得到了一种更具可扩展性和通用性的通用Transformer架构，该架构可作为构建物理科学领域大规模基础模型（foundation models）的骨干网络。我们在包含16种不同类型偏微分方程（PDEs）的大型数据集上证明，所提架构的性能优于计算机视觉领域的现有最优（SOTA）Transformer架构。本文提出将不同物理通道（physical channels）作为独立的时空令牌（spatio-temporal tokens）进行嵌入，这些令牌通过通道级自注意力（channel-wise self-attention）实现交互。该设计在同时学习多种PDE时，有助于保持令牌信息密度的一致性。实验表明，与从头训练相比，我们的预训练模型在多个具有挑战性的下游任务上表现更优，且性能超过了其他用于物理模拟的基础模型架构。本文源代码已开源，地址为：https://github.com/tum-pbs/pde-transformer。

（

PDE-Transformer 就是一个 **专门为物理模拟设计的 Transformer**，它能把 PDE 当作“序列问题”来处理，把不同物理量当作 token，用自注意力来建模它们之间的关系。结果证明，这种方法不仅快，而且比传统 Transformer 更适合多物理场和多 PDE 的学习。

）


**关键词**：物理模拟、偏微分方程（PDEs）、Transformer架构、可扩展性


## 1 引言
在多样化、高质量数据集上训练的大型多用途网络，在适配特定下游任务时展现出优异性能。这类被称为“基础模型”的网络已在计算机视觉（Yuan et al., 2021; Awais et al., 2025）、决策制定（Yang et al., 2023b）和时间序列预测（Liang et al., 2024）等领域获得广泛认可。自然地，这类模型也引起了科学机器学习社区的极大兴趣（Bodnar et al., 2024; Herde et al., 2024; Zhang et al., 2024a），可用于代理建模（Kim et al., 2019; Sun et al., 2020）或涉及物理模拟的逆问题（Ren et al., 2020; Holzschuh et al., 2023）等多种下游任务。在训练数据稀缺的场景中，基础模型的优势尤为突出。

机器学习在物理模拟中的应用面临多项特有挑战：首先，底层物理过程通常具有固有的多尺度特性（Smith, 1985）；其次，数据表示与模拟所用的数值方法紧密相关，涵盖从规则网格、网格（meshes）到基于粒子的表示等多种形式（Anderson et al., 2020）；此外，尽管数值模拟会生成海量数据，但这些数据往往需要进一步预处理才能作为机器学习模型的输入；最后，该领域的机器学习模型需直接与传统数值方法竞争，因此要么需在精度或速度上超越传统求解器且具备高可靠性，要么需提供传统求解器难以获取的解决方案（例如不确定性估计（Geneva & Zabaras, 2019; Jacobsen et al., 2025; Liu & Thuerey, 2024; Kohl et al., 2024）或处理部分输入数据（Wu et al., 2024a））。

为解决上述问题，本文提出一种专为科学机器学习设计的多用途Transformer架构——PDE-Transformer：该架构可在不同类型PDE、不同分辨率、不同域范围（domain extents）、不同边界条件之间实现泛化，并包含针对PDE和任务特定信息的深度条件机制（deep conditioning mechanisms）。该模型适用于二维规则网格，可通过监督学习训练为高效代理模型，也可作为扩散模型（diffusion model）用于解的后验分布较广的下游任务。具体而言，本文的贡献如下：

> **图1**：PDE-Transformer是针对科学数据设计的Transformer模型。上图展示了在包含16种不同PDE动态的大型数据集上，仅给定初始条件时，模型在20个时间步后的自回归预测结果。模型未知额外模拟参数（如粘度、域范围等），需从观测数据中推断。PDE-Transformer特别适合通过预训练应对分布外（out-of-distribution）下游任务。
- 对现有最优扩散Transformer架构进行增强，使其适配PDE和物理模拟任务，例如引入用于高效多尺度建模的令牌下采样与上采样（token down- and upsampling），以及用于提升高分辨率数据扩展性的移位窗口注意力（shifted window attention）；
- 修改注意力操作，解耦时空轴与物理通道轴之间的令牌交互，从而提升精度并增强对不同PDE的泛化能力；
- 对PDE-Transformer进行缩放和修改时，针对精度-计算量权衡（accuracy-compute tradeoffs）开展详细的消融实验；
- 证明在通用PDE集上预训练后，PDE-Transformer在具有挑战性的下游任务中表现出优异的泛化能力。

（

作者做了几件事：

1. **改进了扩散 Transformer**，加入专门为 PDE 设计的机制：
   - token 的下采样和上采样（方便多尺度建模）。
   - 移位窗口注意力（shifted window attention），提升处理高分辨率数据的能力。
2. **修改注意力机制**：把时空维度和物理通道维度的交互解耦开，提高精度和泛化能力。
3. **做了消融实验**：详细研究了模型规模、精度和计算开销的平衡。
4. **证明了泛化能力**：在一个包含 16 种 PDE 的大数据集上预训练后，PDE-Transformer 在困难的下游任务里表现优异，尤其是在 **分布外任务 (out-of-distribution)** 上效果很好。

）


## 2 相关工作
### 2.1 Transformer
Transformer已成为深度学习领域的主流架构之一。尽管Transformer最初用于自然语言处理中的序列到序列模型（Vaswani et al., 2017），但通过将图像分割为补丁（patches）并将补丁与位置信息共同作为令牌嵌入（Dosovitskiy et al., 2021, ViT），Transformer已成功应用于计算机视觉领域。研究表明，在视觉任务中，视觉Transformer（ViT）的扩展性优于传统卷积神经网络，且能更高效地利用加速器资源（Maurício et al., 2023; Takahashi et al., 2024; Rodrigo et al., 2024）。基于Transformer的大型架构已催生出如GPT（Radford et al., 2018, 2019; Brown et al., 2020）和BERT（Devlin et al., 2019）等基础模型。


### 2.2 扩散Transformer
尽管Transformer在计算机视觉中的早期应用聚焦于视觉识别任务，但潜在扩散模型（latent diffusion models）的成功也推动了基于Transformer的扩散模型（即扩散Transformer（Peebles & Xie, 2023, DiT））的发展。与直接对数据空间建模不同，潜在扩散模型（Ho et al., 2020; Rombach et al., 2022）利用变分自编码器（variational autoencoder）对数据进行嵌入，并在生成的潜在空间（latent space）中操作。此外，扩散Transformer具备基于自适应层归一化（adaptive layer normalization）的强大条件机制（Perez et al., 2018），可将类别标签或文本编码作为额外输入。

在计算机视觉任务中，扩散Transformer使用的补丁大小远小于许多早期Transformer模型。尽管更小的补丁大小能提升扩散模型的性能，但由于扩散Transformer架构的全局自注意力机制导致计算量随令牌数量呈二次方增长，其仅能在潜在空间中处理高维数据，使得训练和推理的计算成本较高。本文则聚焦于直接对原始数据建模，无需引入任何预训练自编码器。


### 2.3 注意力机制
Transformer的全局自注意力操作是其计算瓶颈，目前主要有两种解决方案：窗口注意力（windowed attention）将自注意力计算限制在非重叠的局部窗口内（Liu et al., 2021），且在不同层中移位窗口以避免窗口边界处的不连续性；类似地，轴向注意力（axial attention）中，令牌仅沿特定轴（如同一图像行或列）进行交互（Ho et al., 2019）。当在分层架构中结合下采样与上采样时，所得模型在保持或提升性能的同时，能降低计算成本。

此外，还可通过对注意力机制进行算法修改，使注意力计算量呈线性增长（Wang et al., 2020; Cao, 2021），但这类模型的性能通常低于二次方复杂度的自注意力。在物理模拟领域，Transformer和注意力机制的重要性也得到了认可：伽辽金Transformer（Galerkin Transformers, Cao, 2021）采用线性化注意力变体，去除了softmax归一化，可与可学习的层级（layer-wise）Petrov-Galerkin投影相关联；多物理适配Transformer（Multiple Physics Pertaining, McCabe et al., 2023, MPP）基于轴向ViT设计自定义Transformer骨干，沿时间和空间维度计算轴向注意力。


### 2.4 PDE学习
尽管物理残差（physical residuals）带来了艰巨的学习任务（Raissi et al., 2019; Bruna et al., 2024），但神经网络仍可通过多种方式与现有PDE求解器结合：学习修正项（Um et al., 2020; Dresdner et al., 2023; Thuerey et al., 2021）、学习闭合模型（Duraisamy et al., 2019; Sirignano & MacArt, 2023）或学习算法组件（如计算模板（Bar-Sinai et al., 2019; Bar & Sochen, 2019; Kochkov et al., 2021））。

神经算子（neural operators）的研究也引发了广泛关注（Lu et al., 2021; Kovachki et al., 2023），其可实现函数空间之间的映射。注意力机制可通过将节点间距离作为权重扩展到任意网格，当分辨率提高时收敛到积分核算子（Li et al., 2023a, b）。因此，Transformer和注意力机制可被广泛泛化，适用于任意输入和输出点（Wu et al., 2024a）。尽管神经算子具备良好的理论特性，但其灵活性也导致难以扩展到高分辨率和大量数据点。可扩展算子Transformer（scalable operator transformer, Herde et al., 2024, scOT）是一种具有移位窗口的分层视觉Transformer，用于注意力计算。与scOT不同，PDE-Transformer从设计之初就是为PDE优化的改进型扩散Transformer，性能显著优于scOT。


### 2.5 预训练
网络预训练可采用多种策略：自回归预测（Radford et al., 2018）、掩码重建（Devlin et al., 2019; He et al., 2022）和对比学习（Chen et al., 2020）。在PDE领域，基于自回归下一步预测的预训练可纳入额外模拟参数（Gupta & Brandstetter, 2023; Takamoto et al., 2023; Subramanian et al., 2023），也可仅基于历史快照（Herde et al., 2024）。此外，预训练在神经算子（Li et al., 2021; Goswami et al., 2022; Wang et al., 2022）和域迁移（Xu et al., 2023）的PDE学习中也有探索。最近，已有研究在多种PDE动态上同时进行预训练（Subramanian et al., 2023; Yang et al., 2023a; McCabe et al., 2023）。


## 3 PDE-Transformer
### 符号说明
用$\mathcal{S}$表示时空系统，该系统包含$n$个物理量$u(\mathbf{x},t):\Omega_{\mathcal{S}} \times [0,T] \to \mathbb{R}^n$，其中$\Omega_{\mathcal{S}} \subset \mathbb{R}^2$为空间域。假设数据在时间和空间上离散，即系统可表示为序列$[\mathbf{u}_0^{\mathcal{S}}, \mathbf{u}_{\Delta t}^{\mathcal{S}}, \dots, \mathbf{u}_T^{\mathcal{S}}]$，其中每个快照$\mathbf{u}_t^{\mathcal{S}}$采样自空间离散化确定的点。用向量$\mathbf{c}$表示系统$\mathcal{S}$的额外信息（如PDE类型、模拟超参数等）。网络用$\mathcal{M}_{\Theta}$表示，由权重$\Theta$参数化。


### 自回归预测
在自回归预测任务中，目标是基于$T_p$个前序快照$\mathbf{u}_{t-T_p \Delta t}^{\mathcal{S}}, \dots, \mathbf{u}_{t-\Delta t}^{\mathcal{S}}$预测快照$\mathbf{u}_t^{\mathcal{S}}$。本文中，自回归预测的输入和输出分别简记为$\mathbf{u}_{\text{in}} = [\mathbf{u}_{t-T_p \Delta t}^{\mathcal{S}}, \dots, \mathbf{u}_{t-\Delta t}^{\mathcal{S}}]$和$\mathbf{u}_{\text{out}} = [\mathbf{u}_t^{\mathcal{S}}]$。$\mathbf{u}_{\text{in}}$和$\mathbf{u}_{\text{out}}$的定义可灵活适配不同任务：例如，无条件生成模型中$\mathbf{u}_{\text{in}} = \emptyset$且$\mathbf{u}_{\text{out}} = [\mathbf{u}_t^{\mathcal{S}}]$；时间插值任务中$\mathbf{u}_{\text{in}} = [\mathbf{u}_{t-\Delta t}^{\mathcal{S}}, \mathbf{u}_{t+\Delta t}^{\mathcal{S}}]$且$\mathbf{u}_{\text{out}} = [\mathbf{u}_t^{\mathcal{S}}]$。

## 3.1 设计空间

![image-20250901121548295](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20250901121548295.png)

本文对扩散Transformer骨干（Peebles & Xie, 2023, DiT）进行增强，得到PDE-Transformer，其架构如图2所示。下文将逐步介绍DiT架构及本文的修改。

> **图2**：PDE-Transformer架构总览。多尺度架构结合了令牌的上采样与下采样，以及相同分辨率Transformer阶段之间的跳跃连接（skip connections）。注意力操作限制在令牌的局部窗口内，窗口在相邻Transformer块之间移位。条件信息（conditionings）被嵌入后，用于缩放和偏移中间令牌表示。混合通道（MC）版本将不同物理通道嵌入同一令牌；分离通道（SC）版本将不同物理通道独立嵌入，不同物理通道的令牌仅通过通道维度上的轴向自注意力（axial self-attention）交互。通道类型（如速度、密度等）是条件信息的一部分，且每个通道的条件信息不同。

(

##### 整体框架

- **左边的图**：这是整个 PDE-Transformer 的主干结构。
   输入数据（比如流体速度场、密度场等物理量）先经过一个 **Tokenizer（分词器）**，把它们变成 Transformer 可以处理的「令牌」（tokens）。
   然后数据进入一系列 **Transformer Stage**，逐层处理，最后通过 **Projection** 得到输出。
   另外，还有一个 **MLP（多层感知机）** 用来嵌入条件信息（conditioning），比如 PDE 的类型、边界条件、粘性系数等。

------

##### 两种通道设计

右边放大的是 Transformer 里的核心「通道交互」模块，有 **两种变体**：

1. **混合通道 (Mixed channel, MC)**
   - 把不同的物理通道（如速度、压力、密度）混合在一起，打包成一个大 token。
   - 使用 **Shifted Window MHSA（移位窗口多头自注意力）** 在局部区域里做注意力计算，捕捉空间关系。
   - 这种方式的好处是可以直接让不同物理量在空间上互相交流，理解它们的耦合关系。
2. **分离通道 (Separate channel, SC)**
   - 每个物理通道（速度、压力等）单独作为一个令牌，不混在一起。
   - 先用 **Channel-wise MHSA（按通道的自注意力）** 来处理物理通道之间的关系，再用 **Shifted Window MHSA** 来处理空间维度上的关系。
   - 这种方式能保证不同物理量在学习时各自独立，但还能通过专门的操作进行交互。

------

##### 为什么要这样设计？

- **多尺度处理**：图里有 **下采样和上采样**（token down-/upsampling），这样可以先压缩数据，再恢复细节，像 U-Net 一样，提升大规模模拟的效率。
- **条件信息 (conditioning)**：在不同阶段嵌入 PDE 类型、边界条件等，保证模型能泛化到不同物理场景。
- **注意力改造**：传统 Transformer 注意力只在时间或空间维度做交互，这里加上「通道维度」的交互，能更好处理多个物理量之间的耦合关系。

)


### 3.1.1 补丁划分（Patching）
DiT在固定维度的潜在空间中操作，而PDE-Transformer直接对原始数据操作。对于单物理通道数据，给定补丁大小$p$，尺寸为$T \times H \times W$的输入会被划分为$\frac{H}{p} \cdot \frac{W}{p}$个尺寸为$T \times p \times p$的补丁。通过线性层将这些补丁嵌入为表示时空子域的令牌，每个补丁映射到$d$维向量。

补丁大小是关键超参数，直接控制令牌表示的粒度：补丁大小减半会使令牌数量增至4倍，显著增加计算量。本文定义“扩展率”$E(p):=\frac{d}{p^2 T}$，用于描述输入数据的扩展比例。低扩展率更有利于可扩展性（令牌数量更少），但补丁大小与模型性能存在明显相关性（更小补丁性能更优）。第4节将探讨物理模拟中精度与计算成本的权衡。


### 3.1.2 多尺度架构（Multi-scale architecture）
尽管U型架构（如经典UNet（Ronneberger et al., 2015））已取得显著成功，但DiT的原始设计为追求架构简洁性，未采用令牌下采样与上采样（Peebles & Xie, 2023）。U型架构的分层结构与自然界中特征的多尺度特性相符，且能引入强归纳偏置（inductive bias）。已有研究将U型架构与Transformer骨干结合（Bao et al., 2023; Hoogeboom et al., 2023; Tian et al., 2024）。

本文在每个Transformer阶段末尾通过PixelShuffle和PixelUnshuffle层实现令牌的下采样与上采样。与Bao et al. (2023)和Hoogeboom et al. (2023)不同，本文依赖自适应层归一化进行条件控制；Tian et al. (2024)也采用U型设计，但在自注意力的查询-键-值（query-key-value, QKV）元组上进行令牌下采样。本文发现这种设计虽能略微提升性能，但会因加速器利用率不佳导致训练和推理时间增加，因此在PDE数据的实现中未在自注意力内部进行令牌下采样。


### 3.1.3 移位窗口（Shifted Windows）
为避免DiT中全局自注意力的二次方计算量增长，本文采用SwinTransformer（Liu et al., 2021）中的移位窗口多头自注意力（shifted window multi-head self-attention, MHSA）操作，将令牌间的自注意力限制在局部窗口内。每个块中，窗口大小为$w$的窗口包含$w \times w$个时空令牌。为避免相邻层窗口边界处的不连续性，窗口会移位$\frac{w}{2}$个令牌。第4节将评估窗口大小对精度-计算量权衡的影响。

与DiT不同，本文未在令牌嵌入中添加绝对位置信息，而是在计算注意力分数时，结合窗口内令牌的对数间隔相对位置（log-spaced relative positions）与前馈神经网络（Liu et al., 2022）。这一设计提升了PDE学习的平移不变性，并增强了对不同窗口分辨率的泛化能力。


### 3.1.4 混合与分离通道表示（Mixed and separate channel representations）
PDE可能涉及多个物理量，因此需要不同数量的物理通道——这与DiT架构固定通道数的设计有本质区别。不同物理通道的动态特性和尺度可能差异显著：一种简单策略是将最大通道数$C_{\text{max}}$作为输入的额外维度，嵌入层将$T \times C_{\text{max}} \times p \times p$的时空补丁映射到$d$维令牌嵌入；若数据通道数少于$C_{\text{max}}$，则用零填充。这是PDE-Transformer的混合通道（MC）版本。

然而，MC版本会使扩展率$E(p)$降低$\frac{1}{C_{\text{max}}}$倍，且会混合具有不同物理意义的通道，导致令牌表示过度压缩，进而降低性能和泛化能力。为此，本文提出：使计算量随通道数线性增长，且每个令牌的扩展率不受通道数影响。具体实现为：将每个通道独立嵌入，并在每个块中通过额外的通道级轴向MHSA操作实现不同通道令牌间的交互；窗口自注意力不在不同通道的令牌间计算。这种设计称为分离通道（SC）版本，第4节将详细评估两种版本在预训练数据集和微调任务中的性能。


### 3.1.5 条件机制（Conditioning mechanism）
与DiT类似，本文采用adaLN-Zero块（Goyal et al., 2017; Perez et al., 2018; Dhariwal & Nichol, 2021）：将所有条件的嵌入向量相加，通过前馈网络回归得到缩放和偏移向量；此外，初始化缩放和偏移向量使每个残差块等价于恒等函数，以加速训练。

DiT使用类别标签和扩散时间作为条件：当PDE-Transformer作为扩散模型训练时，同样引入扩散时间作为条件；而类别标签对应本文中的PDE类型。此外，在SC版本中，每个物理通道均使用通道类型（如密度、涡度）的标签嵌入。所有标签均采用10%概率的丢弃（dropout），因此模型可同时用于条件和无条件场景。将条件机制扩展到额外模拟参数的过程非常直观。


### 3.1.6 边界条件（Boundary conditions）
本文明确考虑模拟域的周期性和非周期性边界条件：在移位注意力窗口时，令牌沿$x$轴和$y$轴滚动，模拟周期性边界条件；若无需周期性，则可通过在注意力分数计算中屏蔽令牌交互，在架构中显式禁用周期性。


### 3.1.7 算法改进（Algorithmic improvements）
为避免注意力熵不受控增长导致的不稳定性（Dehghani et al., 2023），本文采用RMS归一化（RMSNorm, Zhang & Sennrich, 2019）对自注意力的Q和K进行归一化。此外，本文发现Peebles & Xie (2023)中扩散Transformer的训练配置在本文数据集上会导致训练不稳定和损失峰值，因此将学习率从$1.0 \cdot 10^{-4}$调整为$4.0 \cdot 10^{-5}$，并采用AdamW优化器，结合$10^{-15}$因子的权重衰减（weight decay）进行bf16混合精度训练（Esser et al., 2024）。本文还发现，基于梯度指数移动平均（EMA）的梯度裁剪（gradient clipping）可消除损失曲线中剩余的峰值，确保训练稳定。训练方法的更多细节见附录A。

(

##### 3.1.1 补丁划分 (Patching)

- 就像把一张大地图切成小方块。
- 原始的物理数据（比如速度场）被分成许多小补丁（patch），每个补丁再转成「令牌」给 Transformer 处理。
- 补丁大小 $p$ 决定了切得细不细：补丁小 → 更精细，但令牌数量爆炸，计算量增加。
- 定义了一个“扩展率”指标 $E(p)$，衡量压缩后的信息量。
   👉 类似视频压缩：补丁越小，信息保留越多，但计算开销也更大。

------

##### 3.1.2 多尺度架构 (Multi-scale)

- 自然界的现象往往有「大结构 + 小细节」，比如龙卷风里既有大气旋，也有小漩涡。
- 为了模仿这个特点，引入 **U型结构**（像 UNet），可以在不同分辨率下分析和合并信息。
- PDE-Transformer 里用 PixelShuffle / PixelUnshuffle 来做下采样（缩小）和上采样（放大），更高效。
   👉 就像先缩小地图看整体，再放大局部看细节。

------

##### 3.1.3 移位窗口 (Shifted Windows)

- 原始的 Transformer 注意力计算需要全局对比，计算量是平方级，非常贵。
- PDE-Transformer 借鉴 Swin Transformer：把注意力限制在「小窗口」里（局部区域）。
- 为避免边界不连续，窗口每次都会平移一半位置（shifted window）。
- 还加了「相对位置编码」，让模型更适合处理物理场里的平移不变性。
   👉 就像看一张超大地图，不是全图都对比，而是分块看，块之间还会错开，以保证连贯性。

------

##### 3.1.4 混合与分离通道 (Mixed vs Separate Channels)

- PDE 可能同时有多个物理量（速度、压力、密度……），怎么输入给模型？
- **混合通道 (MC)**：把所有通道打包在一起处理。缺点是会把不同物理量的信息压缩得太狠，效果不好。
- **分离通道 (SC)**：每个物理量单独嵌入，注意力在空间和通道两个维度上分别计算。这样既保持了独立性，又能交互。
   👉 就像处理视频的「RGB三通道」：要么混在一起，要么分开学，再互相沟通。

------

##### 3.1.5 条件机制 (Conditioning)

- PDE 模拟不仅有数据，还需要「条件」：比如 PDE 类型、边界条件、粘度系数等。
- PDE-Transformer 用 **adaLN-Zero** 来嵌入这些条件，相当于给每一层加上特定的缩放/偏移。
- 还加了随机丢弃（dropout），保证模型能适应「有条件」和「无条件」两种场景。
   👉 就像导航时，有时候你告诉它限速信息，有时候没有，模型都要能跑。

------

##### 3.1.6 边界条件 (Boundary Conditions)

- 模拟时必须考虑边界是 **周期的**（循环）还是 **非周期的**（墙）。
- PDE-Transformer 在注意力窗口移动时，可以选择「循环滚动」还是「屏蔽边界」。
   👉 就像打游戏时，地图是无边界循环的（走到右边出来在左边），还是有限边界的（撞到墙停下）。

------

##### 3.1.7 算法改进 (Algorithmic improvements)

- 为了让训练更稳定，做了几项技术改进：
  - 用 RMSNorm 代替常见的 LayerNorm，避免注意力发散。
  - 调小学习率，换成 AdamW 优化器，保证收敛稳定。
  - 用 **梯度 EMA 裁剪**，防止损失曲线突然爆炸。
     👉 就像调汽车引擎：降低油门灵敏度，加上防抖控制，让车开得更稳。

)


## 3.2 监督与扩散训练（Supervised and Diffusion Training）
### 3.2.1 监督训练
PDE-Transformer可通过监督学习或作为扩散模型训练：对于解具有确定性的任务（如训练确定性求解器的代理模型），可使用MSE损失进行监督训练，实现一步快速推理。此时，网络直接通过以下MSE损失训练：
$$\mathcal{L}_S = \mathbb{E}\left[\|\mathcal{M}_{\Theta}(\mathbf{u}_{\text{in}}, \mathbf{c}) - \mathbf{u}_{\text{out}}\|_2^2\right] \tag{1}$$


### 3.2.2 扩散训练
若解具有非确定性，则扩散训练更适用——其可从完整后验分布中采样，而非学习平均解。本文采用扩散模型的流匹配（flow matching, Lipman et al., 2023; Liu et al., 2023）公式（Ho et al., 2020）进行训练：给定输入$\mathbf{u}_{\text{in}}$和条件$\mathbf{c}$，将噪声分布$p_0 = \mathcal{N}(0, I)$中的样本$\mathbf{x}_0$通过常微分方程（ODE）$\text{d}\mathbf{x}_t = v(\mathbf{x}_t, t)\text{d}t$映射到后验分布$p_1$中的样本$\mathbf{x}_1$。网络$\mathcal{M}_{\Theta}$通过回归生成$p_0$和$p_1$之间概率路径的向量场，学习速度$v$。

概率路径上的样本通过前向过程生成：
$$\mathbf{x}_t = t \mathbf{u}_{\text{out}} + \left[1 - (1 - \sigma_{\text{min}})t\right] \epsilon \tag{2}$$
其中$t \in [0,1]$，$\epsilon \sim \mathcal{N}(0, I)$，超参数$\sigma_{\text{min}} = 10^{-4}$。记$\mathbf{c}_t = [\mathbf{c}, t]$和$\mathbf{u}_{\text{in},t} = [\mathbf{u}_{\text{in}}, \mathbf{x}_t]$，通过以下损失回归速度$v$：
$$\mathcal{L}_{\text{FM}} = \mathbb{E}\left[\|\mathcal{M}_{\Theta}(\mathbf{u}_{\text{in},t}, \mathbf{c}_t) - \mathbf{u}_{\text{out}} + (1 - \sigma_{\text{min}})\epsilon\|_2^2\right] \tag{3}$$

训练完成后，基于$\mathbf{u}_{\text{in}}$和$\mathbf{c}$的后验采样过程如下：采样$\mathbf{x}_0 \sim \mathcal{N}(0, I)$，求解ODE $\text{d}\mathbf{x}_t = \mathcal{M}(\mathbf{u}_{\text{in},t}, \mathbf{c}_t)\text{d}t$（从$t=0$到$t=1$）。本文采用显式欧拉法，并在第4节中实验探索PDE采样的最优步长$\Delta t$。

(

##### 3.2.1 监督训练（Supervised Training）

- **场景**：当 PDE 的解是**确定的**，比如输入一定、输出也唯一，那么用普通的监督学习就行。
- **方法**：模型输入初始条件 $\mathbf{u}_{\text{in}}$ 和额外信息 $\mathbf{c}$，输出预测解 $\mathbf{u}_{\text{out}}$。
- **目标**：让预测结果尽量接近真实解，所以用 **均方误差 (MSE)** 来衡量差距：

$$
\mathcal{L}_S = \mathbb{E}\Big[\|\mathcal{M}_{\Theta}(\mathbf{u}_{\text{in}}, \mathbf{c}) - \mathbf{u}_{\text{out}}\|_2^2\Big]
$$

👉 简单来说，就是让模型学会“一步到位”给出正确解。

------

##### 3.2.2 扩散训练（Diffusion Training）

- **场景**：当 PDE 的解 **不唯一** 或存在**随机性**（比如流体中带有噪声或不同可能结果），监督学习学到的“平均解”没意义。
- **解决办法**：用 **扩散模型**。扩散模型的思想是：
  - 从随机噪声（高斯分布 $\mathcal{N}(0,I)$）出发；
  - 通过一个「流动过程」（ODE）逐渐把噪声推到真实解的分布；
  - 模型的任务就是学会这个过程里的「速度场」 $v$，告诉我们该怎么一步步从噪声走向解。

)


## 4 实验
本文评估PDE-Transformer在自回归预测任务中的性能（前序快照数$T_p=1$），实验分为两部分：首先，在包含多种PDE的大型预训练集上，将PDE-Transformer与其他现有最优Transformer架构对比，重点关注精度、训练时间和计算量，并通过消融实验验证PDE-Transformer的设计选择；其次，将预训练网络在三个具有挑战性的下游任务（涉及新边界条件、不同分辨率、物理通道和域大小）上微调，验证其对分布外数据的泛化能力。

本文模型采用三种配置（S、B、L），对应不同的令牌嵌入维度$d$（分别为96、192、384），记为PDE-S（S配置的PDE-Transformer）。除非特别说明，补丁大小$p$和窗口大小$w$默认设为4和8，且使用混合通道（MC）版本。


### 4.1 预训练数据集
#### 4.1.1 训练设置
预训练数据集包含16种线性和非线性PDE，包括柯尔莫哥洛夫流（Kolmogorov flow）、伯格斯方程（Burgers’ equation）、格雷-斯科特方程（Gray Scott equation）的多种变体等。数据集基于APEBench（Koehler et al., 2024）构建，详细描述见附录C。每种PDE包含600条轨迹，每条轨迹有30个模拟步，数据集按固定比例划分为训练集、验证集和测试集。

数据通过光谱求解器生成，原始分辨率为$2048 \times 2048$，下采样至$256 \times 256$，PDE具有1或2个物理通道。训练较大模型时采用梯度累积，保持批量大小不变；评估时使用权重的EMA（衰减系数0.999）。


#### 4.1.2 评估指标
评估采用归一化均方根误差（nRMSE），定义为：
$$\text{nRMSE} = \frac{1}{M} \sum_{i=1}^M \frac{\text{MSE}(\hat{\mathbf{u}}_{\text{out}}, \mathbf{u}_{\text{out}})}{\text{MSE}(0, \mathbf{u}_{\text{out}})} \tag{4}$$
其中$\hat{\mathbf{u}}_{\text{out}}$为网络预测结果，$M$为测试集中的轨迹数。对于系统$\mathcal{S}$，可自回归生成完整轨迹，定义时间$t$处的nRMSE为预测状态$\hat{\mathbf{u}}_t^{\mathcal{S}}$与参考状态$\mathbf{u}_t^{\mathcal{S}}$的误差，每个时间步的误差均在测试集中所有系统上取平均。


#### 4.1.3 窗口注意力与多尺度架构
本文训练了多种具有代表性的Transformer模型进行对比：
- DiT（Peebles & Xie, 2023）：无令牌下采样/上采样；
- UDiT（Tian et al., 2024）：采用U型架构；
- scOT（Herde et al., 2024）：神经算子Transformer，含分层架构和移位窗口注意力；
- FactFormer（Li et al., 2023b）：基于轴向因子核积分的Transformer；
- 现代UNet（Ronneberger et al., 2015; Ho et al., 2020）；
- 本文提出的PDE-Transformer。

所有模型均采用各自架构的S配置，补丁大小$p=4$（对应$64 \times 64$时空令牌），通过nRMSE评估自回归预测的1步和10步误差，结果如表1所示。

**表1**：在4块H100 GPU上训练100个epoch的S配置模型性能对比

| 模型       | 1步nRMSE  | 10步nRMSE | 训练时间（小时） | 参数数量（百万） |
| ---------- | --------- | --------- | ---------------- | ---------------- |
| DiT-S      | 0.066     | 0.78      | 13h 4m           | 39.8             |
| UDiT-S     | 0.042     | 0.39      | 18h 30m          | 58.9             |
| scOT-S     | 0.051     | 0.59      | 21h 11m          | 39.8             |
| FactFormer | 0.069     | 0.65      | 12h 25m          | 3.8              |
| UNet       | 0.075     | 0.68      | 48h 00m          | 35.7             |
| **PDE-S**  | **0.044** | **0.36**  | **7h 42m**       | **33.2**         |

由表1可知，UDiT-S和PDE-S性能最优，但两者在训练时间上存在显著差异：PDE-S训练速度更快（7小时42分钟）。尽管DiT-S的计算量（GFlops）更高，但其训练时间（13小时4分钟）仍远低于UDiT-S（18小时30分钟），这是因为DiT架构在现代GPU硬件上的加速器利用率优于UDiT。此外，DiT-S训练过程中会出现损失峰值且无法恢复（与学习率和梯度裁剪策略无关），表明DiT架构在PDE数据集上存在稳定性问题，因此DiT-S的评估使用验证损失最低的检查点（checkpoint）。


#### 4.1.4 域大小与模型参数的高效缩放
图3展示了补丁大小$p=4$时，PDE-S、UDiT-S和DiT-S在不同输入大小下的推理计算量（GFlops）和GPU内存需求（GB）。由于全局自注意力，DiT-S在大域上的计算成本极高；UDiT-S的整体扩展性更优，但由于注意力算子内用于令牌上/下采样的卷积层，其GPU内存需求增长较快；PDE-S在U型架构中仅使用少量卷积层进行上/下采样，在GFlops和GPU内存方面均实现了最优缩放。

> **图3**：补丁大小$p=4$时的域大小缩放。$256 \times 256$输入分辨率对应$64 \times 64$令牌，每个点的面积代表推理时的GPU内存（批量大小为1）。

本文训练了PDE-Transformer的S、B、L三种配置（仅增加令牌嵌入维度$d$，架构保持一致），结果如表2所示。更大的令牌嵌入维度可提升性能：当$d$翻倍时，推理计算量约增至4倍，但训练时间增长速度较慢——这得益于现代GPU上自注意力算子中矩阵乘法的高效加速器利用率。不同配置的监督损失和流匹配损失见附录图7。

**表2**：令牌嵌入维度$d$的缩放对比（监督训练的1步nRMSE，4块H100 GPU训练100个epoch的时间）

| 配置 | $d$（嵌入维度） | 1步nRMSE | 训练时间（小时） | 推理计算量（GFlops） |
| ---- | --------------- | -------- | ---------------- | -------------------- |
| S    | 96              | 0.045    | 7h 42m           | 19.62                |
| B    | 192             | 0.038    | 10h 40m          | 76.55                |
| L    | 384             | 0.035    | 20h 8m           | 302.34               |


#### 4.1.5 不同物理通道的轴向注意力
图4（底部）对比了三种配置下PDE-Transformer的混合通道（MC）和分离通道（SC）版本：SC版本的灵活性未导致推理精度下降，但由于令牌数量随通道数线性增长，其计算量增加。鉴于SC版本的根本性灵活性优势，后续实验聚焦于SC版本，并在4.2节的下游任务中验证其泛化能力。


#### 4.1.6 监督学习与概率学习
图4（顶部）对比了监督训练与基于流匹配的扩散训练（及后验采样）：扩散型PDE-Transformer从后验分布采样，而监督训练旨在预测后验均值。尽管不同PDE有更适合的评估指标，但由于数据集包含多种PDE，本文仅使用nRMSE对比——监督训练的性能始终优于扩散版本的单样本预测（测试集中每条轨迹仅生成一个样本）。尽管扩散版本的精度略低，但仍接近监督基线，且其从后验采样的能力在许多下游任务和实际工程应用中极具价值。此外，本文实验了采样步长$\Delta t$的影响（见附录B图8），发现步长越多，性能越优。

> **图4**：顶部：监督训练与流匹配训练及后验采样的nRMSE评估；底部：混合通道（MC）与通道维度轴向注意力分离通道（SC）的nRMSE对比。


#### 4.1.7 补丁与窗口大小的影响
本文以S配置的监督训练模型为例，评估补丁大小$p$和窗口大小$w$的影响：
1. **固定$p=4$，改变$w$**：更大的窗口会扩大自注意力的感受野，降低训练损失，但计算量增加（图5左）；然而，测试集nRMSE无明显提升，表明存在过拟合，因此小窗口大小足以捕捉数据集中PDE的代表性行为。
2. **保持感受野恒定（$p \cdot w = 32$），改变$p$和$w$**：更小的$p$可提升性能，但计算量显著增加（图5右）；精度-计算量权衡的平衡点为$p=4$、$w=8$。

> **图5**：窗口大小$w$（左）和补丁大小$p$（右）对性能与计算量（GFlops）的影响。左图：固定$p=4$，增大自注意力窗口；右图：选择$p$和$w$使$p \cdot w = 32$（窗口覆盖相同空间域）。


### 4.2 下游任务
本节在三个具有挑战性的下游任务（来自Well库（Ohana et al., 2024））上评估模型性能：活性物质（active matter）、瑞利-贝纳德对流（Rayleigh-Bénard convection, RBC）、剪切流（shear flow）。这些任务的动态特性远复杂于预训练数据集，涉及不同边界条件（周期与非周期）和几何形状（正方形与非正方形域），任务细节见附录E.1。附录B.4还初步评估了PDE-Transformer与低秩适配（LoRA, Hu et al., 2022）结合的性能。

#### 4.2.1 对比模型与实验设置
下游任务的对比模型包括：
- scOT-S（Herde et al., 2024）；
- 伽辽金Transformer（Galerkin Transformer, Cao, 2021）；
- OFormer（Li et al., 2023a）；
- 傅里叶神经算子（Fourier Neural Operator, Kossaifi et al., 2024, FNO）。

所有基线模型均从头训练；PDE-Transformer的S配置分为“从头初始化”和“预训练权重初始化”两组。除OFormer（内存消耗随模型大小显著增加）外，所有模型的可训练参数数量相近。需注意，scOT和伽辽金Transformer限于正方形模拟域，因此不适用于RBC和剪切流任务，模型细节见附录A.3。


#### 4.2.2 下游任务性能结果
图6展示了各模型在下游任务上的平均滚动nRMSE（rollout nRMSE）：预训练的PDE-S在所有Well任务中均实现了更高的预测精度，nRMSE平均比第二优模型（FNO）低42%，证明PDE-Transformer的性能持续优于所有基线模型。值得注意的是，尽管预训练数据来自光谱求解器的理想化数据，但预训练仍提升了复杂动态下游任务的性能；此外，预训练模型在非周期边界条件（RBC）和非正方形域（RBC和剪切流）上也表现出性能提升，滚动预测误差的详细数值见附录B.2。

> **图6**：不同模型在下游任务上的平均滚动nRMSE。

此外，Poseidon模型（Herde et al., 2024）提供了scOT-B配置的预训练权重。为公平对比，本文评估了预训练PDE-B与预训练scOT-B（两者大小相近）的性能：PDE-B的精度显著提升，累积nRMSE比scOT-B低75%（scOT-B在长期滚动预测中不稳定），模型预测的平均滚动nRMSE见附录图9。上述结果充分证明了本文提出的架构和预训练方法的优势。


#### 4.2.3 通道表示对下游任务的影响
表3对比了PDE-S的SC和MC版本在下游任务中的性能：无预训练时，两种版本的性能相近；但预训练后，SC版本在下游任务中的性能提升显著更高——在三个挑战性任务中，SC版本的性能提升是MC版本的2.7倍至4.4倍。这一结果进一步表明，本文提出的通道独立表示使网络能保留更多预训练知识，且无需对PDE特定层进行大量训练或微调。

**表3**：PDE-Transformer分离通道（SC）与混合通道（MC）版本的性能对比（数值为前20步滚动预测nRMSE的平均值）

| 任务                   | 通道类型 | 从头训练 | 预训练 | 性能提升 |
| ---------------------- | -------- | -------- | ------ | -------- |
| 活性物质               | SC       | 0.494    | 0.455  | 7.89%    |
|                        | MC       | 0.493    | 0.479  | 2.84%    |
| 瑞利-贝纳德对流（RBC） | SC       | 0.155    | 0.104  | 32.90%   |
|                        | MC       | 0.147    | 0.130  | 11.56%   |
| 剪切流                 | SC       | 0.199    | 0.125  | 37.19%   |
|                        | MC       | 0.178    | 0.163  | 8.43%    |


## 5 局限性
PDE-Transformer目前限于二维规则网格，且本文聚焦于自回归预测任务。未来需进一步验证和扩展PDE-Transformer，以处理噪声数据、部分观测数据、数据同化（data assimilation）或逆问题等场景。

将PDE-Transformer扩展到非规则网格的方向包括：
1. 结合图神经算子（GNO, Li et al., 2020）层作为编码器和解码器，替代补丁划分（patchification），实现从给定几何形状到潜在规则网格的映射；
2. 将注意力窗口泛化为图的局部邻域，并将令牌上/下采样替换为相应的图池化（graph pooling）和上采样操作。


## 6 结论
本文提出PDE-Transformer，一种多用途Transformer模型，用于解决物理模拟中的关键挑战。其多尺度架构基于扩散Transformer骨干改进，将令牌交互限制在局部窗口内，且不牺牲PDE动态的学习性能。在精度-计算量权衡方面，PDE-Transformer的性能超过现有最优Transformer架构，在高分辨率数据上展现出优异的可扩展性。通过解耦不同物理通道的令牌嵌入，PDE-Transformer的性能进一步提升（尤其在复杂下游任务的微调中）。其高可扩展性使PDE-Transformer成为高分辨率模拟数据基础模型的优秀骨干网络。未来，本文计划将PDE-Transformer从二维扩展到三维模拟。


## 致谢
本研究得到欧洲研究理事会（ERC）整合型资助项目SpaTe（项目编号：CoG-2019-863850）的资助。作者感谢巴伐利亚科学院莱布尼茨超级计算中心（LRZ）提供的AI服务基础设施LRZ AI Systems及相关科学支持，该设施由巴伐利亚州科学与艺术部（StMWK）资助。


## 影响声明
本文旨在推进机器学习领域的发展，其研究成果具有多种潜在社会影响，但无需特别强调某一具体影响。


## 参考文献
Aleem et al. (2024)  
Aleem, S., Dietlmeier, J., Arazo, E., and Little, S. 基于ConvLora和AdaBN的自训练域自适应（ConvLora and AdaBN based domain adaptation via self-training）. In IEEE国际生物医学成像研讨会（ISBI 2024），希腊雅典，2024年5月27-30日，第1-5页. IEEE, 2024. doi: 10.1109/ISBI56570.2024.10635661. URL https://doi.org/10.1109/ISBI56570.2024.10635661.

Anderson et al. (2020)  
Anderson, D., Tannehill, J. C., Pletcher, R. H., Munipalli, R., and Shankar, V. 计算流体力学与传热学（Computational fluid mechanics and heat transfer）. CRC Press, 2020.

Awais et al. (2025)  
Awais, M., Naseer, M., Khan, S., Anwer, R. M., Cholakkal, H., Shah, M., Yang, M.-H., and Khan, F. S. 基础模型定义计算机视觉新纪元：综述与展望（Foundation models defining a new era in vision: a survey and outlook）. IEEE Transactions on Pattern Analysis and Machine Intelligence, pp. 1–20, 2025. doi: 10.1109/TPAMI.2024.3506283.

Bao et al. (2023)  
Bao, F., Nie, S., Xue, K., Cao, Y., Li, C., Su, H., and Zhu, J. 万物皆可为词：扩散模型的ViT骨干（All are worth words: A ViT backbone for diffusion models）. In IEEE/CVF计算机视觉与模式识别会议（CVPR），第22669-22679页. IEEE, 2023. URL https://doi.org/10.1109/CVPR52729.2023.02171.

Bar & Sochen (2019)  
Bar, L. and Sochen, N. 用于PDE正问题与逆问题的无监督深度学习算法（Unsupervised deep learning algorithm for PDE-based forward and inverse problems）. arXiv预印本 arXiv:1904.05417, 2019.

Bar-Sinai et al. (2019)  
Bar-Sinai, Y., Hoyer, S., Hickey, J., and Brenner, M. P. 学习数据驱动的偏微分方程离散化方法（Learning data driven discretizations for partial differential equations）. Proceedings of the National Academy of Sciences, 116(31):15344–15349, 2019. ISSN 0027-8424, 1091-6490. doi: 10.1073/pnas.1814058116.

Bodnar et al. (2024)  
Bodnar, C., Bruinsma, W. P., Lucic, A., Stanley, M., Brandstetter, J., Garvan, P., Riechert, M., Weyn, J., Dong, H., Vaughan, A., et al. Aurora：大气基础模型（Aurora: A foundation model of the atmosphere）. arXiv预印本 arXiv:2405.13063, 2024.

Brown et al. (2020)  
Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. 语言模型是少样本学习者（Language models are few-shot learners）. Advances in Neural Information Processing Systems, 33:1877–1901, 2020.

Bruna et al. (2024)  
Bruna, J., Peherstorfer, B., and Vanden-Eijnden, E. 基于主动学习的高维发展方程神经伽辽金格式（Neural Galerkin schemes with active learning for high-dimensional evolution equations）. J. Comput. Phys., 496:112588, 2024. doi: 10.1016/J.JCP.2023.112588. URL https://doi.org/10.1016/j.jcp.2023.112588.

Buehler & Buehler (2024)  
Buehler, E. L. and Buehler, M. J. X-LoRA：低秩适配专家混合模型——大型语言模型的灵活框架及其在蛋白质力学与分子设计中的应用（X-LoRA: Mixture of low-rank adapter experts, a flexible framework for large language models with applications in protein mechanics and molecular design）, 2024. URL https://arxiv.org/abs/2402.07148.

Burns et al. (2020)  
Burns, K. J., Vasil, G. M., Oishi, J. S., Lecoanet, D., and Brown, B. P. Dedalus：光谱方法数值模拟的灵活框架（Dedalus: A flexible framework for numerical simulations with spectral methods）. Physical Review Research, 2(2):023068, 2020. doi: 10.1103/PhysRevResearch.2.023068. URL https://doi.org/10.1103/PhysRevResearch.2.023068.

Cao (2021)  
Cao, S. 选择Transformer：傅里叶还是伽辽金（Choose a transformer: Fourier or Galerkin）. In 神经信息处理系统进展（NeurIPS 2021），第34卷, 2021. URL https://openreview.net/forum?id=ssohLcmn4-r.

Chen et al. (2020)  
Chen, T., Kornblith, S., Norouzi, M., and Hinton, G. E. 视觉表示对比学习的简单框架（A simple framework for contrastive learning of visual representations）. In 第37届国际机器学习会议（ICML 2020），2020年7月13-18日，虚拟会议，《机器学习研究会议论文集》第119卷，第1597-1607页. PMLR, 2020. URL http://proceedings.mlr.press/v119/chen20j.html.

Dehghani et al. (2023)  
Dehghani, M., Djolonga, J., Mustafa, B., Padlewski, P., Heek, J., Gilmer, J., Steiner, A. P., Caron, M., Geirhos, R., Alabdulmohsin, I., Jenatton, R., Beyer, L., Tschannen, M., Arnab, A., Wang, X., Ruiz, C. R., Minderer, M., Puigcerver, J., Evci, U., Kumar, M., van Steenkiste, S., Elsayed, G. F., Mahendran, A., Yu, F., Oliver, A., Huot, F., Bastings, J., Collier, M., Gritsenko, A. A., Birodkar, V., Vasconcelos, C. N., Tay, Y., Mensink, T., Kolesnikov, A., Pavetic, F., Tran, D., Kipf, T., Lucic, M., Zhai, X., Keysers, D., Harmsen, J. J., and Houlsby, N. 将视觉Transformer扩展到220亿参数（Scaling vision transformers to 22 billion parameters）. In Krause, A., Brunskill, E., Cho, K., Engelhardt, B., Sabato, S., and Scarlett, J.（编），国际机器学习会议（ICML 2023），2023年7月23-29日，美国夏威夷檀香山，《机器学习研究会议论文集》第202卷，第7480-7512页. PMLR, 2023. URL https://proceedings.mlr.press/v202/dehghani23a.html.

Devlin et al. (2019)  
Devlin, J., Chang, M., Lee, K., and Toutanova, K. BERT：用于语言理解的深度双向Transformer预训练（BERT: Pre-training of deep bidirectional transformers for language understanding）. In Burstein, J., Doran, C., and Solorio, T.（编），第2019届北美计算语言学协会会议论文集：人类语言技术（NAACL-HLT 2019），2019年6月2-7日，美国明尼苏达州明尼阿波利斯，第1卷（长文与短文），第4171-4186页. 计算语言学协会, 2019. doi: 10.18653/V1/N19-1423. URL https://doi.org/10.18653/v1/n19-1423.

Dhariwal & Nichol (2021)  
Dhariwal, P. and Nichol, A. Q. 扩散模型在图像合成上超越GAN（Diffusion models beat GANs on image synthesis）. In Ranzato, M., Beygelzimer, A., Dauphin, Y. N., Liang, P., and Vaughan, J. W.（编），神经信息处理系统进展34：2021年度会议（NeurIPS 2021），2021年12月6-14日，虚拟会议，第8780-8794页, 2021.

Dosovitskiy et al. (2021)  
Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., and Houlsby, N. 一图胜千言：用于图像识别的大尺度Transformer（An image is worth 16x16 words: Transformers for image recognition at scale）. In 国际学习表示会议（ICLR）, 2021. URL https://openreview.net/forum?id=YicbFdNTTy.

Dresdner et al. (2023)  
Dresdner, G., Kochkov, D., Norgaard, P. C., Zepeda-Nunez, L., Smith, J., Brenner, M., and Hoyer, S. 学习修正光谱方法以模拟湍流（Learning to correct spectral methods for simulating turbulent flows）. Transactions on Machine Learning Research, 2023. ISSN 2835-8856. URL https://openreview.net/forum?id=wNBARGxoJn.

Duraisamy et al. (2019)  
Duraisamy, K., Iaccarino, G., and Xiao, H. 数据时代的湍流建模（Turbulence modeling in the age of data）. Annual Review of Fluid Mechanics, 51(1):357–377, 2019年1月. doi: 10.1146/annurev-fluid-010518-040547. URL https://doi.org/10.1146%2Fannurev-fluid-010518-040547.

Esser et al. (2024)  
Esser, P., Kulal, S., Blattmann, A., Entezari, R., Müller, J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F., Podell, D., Dockhorn, T., English, Z., and Rombach, R. 缩放整流流Transformer以实现高分辨率图像合成（Scaling rectified flow transformers for high-resolution image synthesis）. In 第41届国际机器学习会议（ICML 2024），奥地利维也纳，2024年7月21-27日. OpenReview.net, 2024. URL https://openreview.net/forum?id=FPnUhsQJ5B.

Falcon & The PyTorch Lightning team (2019)  
Falcon, W. and The PyTorch Lightning team. PyTorch Lightning, 2019年3月. URL https://github.com/Lightning-AI/lightning.

Geneva & Zabaras (2019)  
Geneva, N. and Zabaras, N. 基于贝叶斯深度神经网络的雷诺平均湍流模型形式不确定性量化（Quantifying model form uncertainty in Reynolds-averaged turbulence models with Bayesian deep neural networks）. Journal of Computational Physics, 383:125–147, 2019. ISSN 0021-9991. doi: https://doi.org/10.1016/j.jcp.2019.01.021. URL https://www.sciencedirect.com/science/article/pii/S0021999119300464.

Goswami et al. (2022)  
Goswami, S., Kontolati, K., Shields, M. D., and Karniadakis, G. E. 用于条件偏移下偏微分方程的深度迁移算子学习（Deep transfer operator learning for partial differential equations under conditional shift）. Nature Machine Intelligence, 4(12):1155–1164, 2022.

Goyal et al. (2017)  
Goyal, P., Dollár, P., Girshick, R. B., Noordhuis, P., Wesolowski, L., Kyrola, A., Tulloch, A., Jia, Y., and He, K. 高精度大批量SGD：1小时训练ImageNet（Accurate, large minibatch SGD: Training ImageNet in 1 hour）. CoRR, abs/1706.02677, 2017. URL http://arxiv.org/abs/1706.02677.

Gupta et al. (2024a)  
Gupta, D., Bhatti, A., and Parmar, S. 超越LoRA：探索时间序列基础模型的高效微调技术（Beyond LoRA: Exploring efficient fine-tuning techniques for time series foundational models）, 2024a. URL https://arxiv.org/abs/2409.11302.

Gupta et al. (2024b)  
Gupta, D., Bhatti, A., Parmar, S., Dan, C., Liu, Y., Shen, B., and Lee, S. 时间序列基础模型的低秩适配用于分布外模态预测（Low-rank adaptation of time series foundational models for out-of-domain modality forecasting）. In Hung, H., Oertel, C., Soleymani, M., Chaspari, T., Dibeklioglu, H., Shukla, J., and Truong, K. P.（编），第26届国际多模态交互会议论文集（ICMI 2024），哥斯达黎加圣何塞，2024年11月4-8日，第382-386页. ACM, 2024b. doi: 10.1145/3678957.3685724. URL https://doi.org/10.1145/3678957.3685724.

Gupta & Brandstetter (2023)  
Gupta, J. K. and Brandstetter, J. 迈向多时空尺度通用PDE建模（Towards multi-spatiotemporal-scale generalized PDE modeling）. Trans. Mach. Learn. Res., 2023, 2023. URL https://openreview.net/forum?id=dPSTDbGtBY.

He et al. (2022)  
He, K., Chen, X., Xie, S., Li, Y., Dollár, P., and Girshick, R. B. 掩码自编码器是可扩展的视觉学习者（Masked autoencoders are scalable vision learners）. In IEEE/CVF计算机视觉与模式识别会议（CVPR 2022），美国路易斯安那州新奥尔良，2022年6月18-24日，第15979-15988页. IEEE, 2022. doi: 10.1109/CVPR52688.2022.01553. URL https://doi.org/10.1109/CVPR52688.2022.01553.

Herde et al. (2024)  
Herde, M., Raonic, B., Rohner, T., Käppeli, R., Molinaro, R., de Bézenac, E., and Mishra, S. Poseidon：用于PDE的高效基础模型（Poseidon: Efficient foundation models for PDEs）. In Globersons, A., Mackey, L., Belgrave, D., Fan, A., Paquet, U., Tomczak, J. M., and Zhang, C.（编），神经信息处理系统进展38：2024年度会议（NeurIPS 2024），加拿大不列颠哥伦比亚省温哥华，2024年12月10-15日, 2024.

Ho et al. (2019)  
Ho, J., Kalchbrenner, N., Weissenborn, D., and Salimans, T. 多维Transformer中的轴向注意力（Axial attention in multidimensional transformers）. CoRR, abs/1912.12180, 2019. URL http://arxiv.org/abs/1912.12180.

Ho et al. (2020)  
Ho, J., Jain, A., and Abbeel, P. 去噪扩散概率模型（Denoising diffusion probabilistic models）. In Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., and Lin, H.（编），神经信息处理系统进展，第33卷，第6840-6851页. Curran Associates, Inc., 2020.

Holzschuh et al. (2023)  
Holzschuh, B. J., Vegetti, S., and Thuerey, N. 基于得分匹配求解物理逆问题（Solving inverse physics problems with score matching）. In Oh, A., Naumann, T., Globerson, A., Saenko, K., Hardt, M., and Levine, S.（编），神经信息处理系统进展36：2023年度会议（NeurIPS 2023），美国路易斯安那州新奥尔良，2023年12月10-16日, 2023.

Hoogeboom et al. (2023)  
Hoogeboom, E., Heek, J., and Salimans, T. 简单扩散：高分辨率图像的端到端扩散（Simple diffusion: End-to-end diffusion for high resolution images）. In Krause, A., Brunskill, E., Cho, K., Engelhardt, B., Sabato, S., and Scarlett, J.（编），国际机器学习会议（ICML 2023），2023年7月23-29日，美国夏威夷檀香山，《机器学习研究会议论文集》第202卷，第13213-13232页. PMLR, 2023. URL https://proceedings.mlr.press/v202/hoogeboom23a.html.

Hu et al. (2022)  
Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W. LoRA：大型语言模型的低秩适配（LoRA: Low-rank adaptation of large language models）. In 第十届国际学习表示会议（ICLR 2022），虚拟会议，2022年4月25-29日. OpenReview.net, 2022. URL https://openreview.net/forum?id=nZeVKeeFYf9.

Jacobsen et al. (2025)  
Jacobsen, C., Zhuang, Y., and Duraisamy, K. CoCoGen：用于正逆问题的物理一致条件得分生成模型（CoCoGen: Physically consistent and conditioned score-based generative models for forward and inverse problems）. SIAM J. Sci. Comput., 47(2):399, 2025. doi: 10.1137/24M1636071. URL https://doi.org/10.1137/24m1636071.

Kim et al. (2019)  
Kim, B., Azevedo, V. C., Thuerey, N., Kim, T., Gross, M., and Solenthaler, B. Deep Fluids：参数化流体模拟的生成网络（Deep Fluids: A Generative Network for Parameterized Fluid Simulations）. Comput. Graph. Forum, 38(2):12, 2019. URL http://www.byungsoo.me/project/deep-fluids/.

Kochkov et al. (2021)  
Kochkov, D., Smith, J. A., Alieva, A., Wang, Q., Brenner, M. P., and Hoyer, S. 机器学习加速计算流体力学（Machine learning–accelerated computational fluid dynamics）. Proceedings of the National Academy of Sciences, 118(21):e2101784118, 2021. doi: 10.1073/pnas.2101784118. URL https://www.pnas.org/doi/abs/10.1073/pnas.2101784118.

Koehler et al. (2024)  
Koehler, F., Niedermayr, S

# 附录A 训练配置
## A.1 训练超参数
表4总结了训练过程中使用的关键超参数。我们采用PyTorch Lightning（Falcon & The PyTorch Lightning team, 2019）支持的分布式数据并行（Distributed Data Parallel, DDP）策略，在多块GPU上训练模型。为有效管理内存消耗并在不同硬件配置下保持一致的有效批量大小，我们引入了梯度累积技术。此外，除下游任务中的FNO和OFormer模型外，大多数模型均启用混合精度训练。排除这两个模型是因为其架构中的快速傅里叶变换与混合精度设置存在固有不兼容问题。

**表4**：训练的主要超参数

| 超参数             | 预训练               | 下游任务                                                     |
| ------------------ | -------------------- | ------------------------------------------------------------ |
| 有效批量大小       | 256                  | 256                                                          |
| 学习率             | $4.00 \cdot 10^{-5}$ | 活性物质与RBC：$1.00 \cdot 10^{-4}$；剪切流：$4.00 \cdot 10^{-5}$ |
| 优化器             | AdamW                | AdamW                                                        |
| 训练轮次（Epochs） | 100                  | 2000                                                         |


## A.2 EMA梯度裁剪
本研究采用梯度范数的指数移动平均（Exponential Moving Average, EMA）来稳定训练过程。我们使用两个不同系数计算梯度范数的EMA值，具体如算法1所示。系数较大的EMA更注重历史值，用作裁剪阈值；而系数较小的EMA则作为梯度裁剪的目标值。需注意，裁剪系数$\kappa$必须设置为大于1，以确保$g_1$能有效跟踪梯度范数的变化。与传统梯度裁剪（采用固定阈值和裁剪值）相比，EMA梯度裁剪通过动态调整阈值和裁剪值，提供了更灵活的方案。这种灵活性尤为重要，因为不同模型规模和训练阶段的梯度范数差异显著。

**算法1 EMA梯度裁剪**  
输入：第一个EMA系数$\beta_1$、第二个EMA系数$\beta_2$、裁剪阈值系数$\alpha$、裁剪值系数$\kappa$。  
初始化：$\beta_1=0.99$，$\beta_2=0.999$，$\alpha=2$，$\kappa=1.1$，$i=0$，$g_1=0$，$g_2=0$。  
循环：  
1. 从训练步骤中获取梯度$\mathbf{g}$；  
2. 若$i \neq 0$且$|\mathbf{g}| > \alpha \cdot \frac{g_2}{1-\beta_2^i}$，则$\mathbf{g} = \kappa \cdot \mathbf{g} \cdot \frac{g_1}{1-\beta_1^i}$；  
3. 更新$g_1 = \beta_1 \cdot g_1 + (1-\beta_1) \cdot |\mathbf{g}|$；  
4. 更新$g_2 = \beta_1 \cdot g_2 + (1-\beta_1) \cdot |\mathbf{g}|$；  
5. $i = i + 1$；  
直到训练结束。


## A.3 模型细节
表5总结了不同模型的网络规模。scOT模型的规模定义遵循Poseidon基础模型（Herde et al., 2024）中的定义；同时，我们在官方实现（Kossaifi et al., 2024）中通过设置$n_{\text{modes_height}}=16$、$n_{\text{modes_width}}=16$和$\text{hidden_channels}=192$，对FNO的规模进行了调整。对于伽辽金Transformer，我们同样在官方实现（Cao, 2021）中修改了$n_{\text{hidden}}=96$、$\text{num_encoder_layers}=5$、$\text{dim_feedforward}=128$和$\text{freq_dim}=432$，以扩展网络规模。我们注意到，OFormer在训练过程中所需内存远高于其他模型，因此不得不修改其原始实现（Li et al., 2023a）中的$\text{in_emb_dim}=24$和$\text{out_seq_emb_dim}=48$，使其内存消耗与其他模型相近，尽管其网络规模仍处于不同量级。

**表5**：不同模型的网络规模

| 模型              | 可训练参数数量 |
| ----------------- | -------------- |
| PDE-S             | 46.57M         |
| scOT-S            | 39.90M         |
| FNO               | 42.72M         |
| 伽辽金Transformer | 38.82M         |
| OFormer           | 0.12M          |
| 预训练PDE-B       | 178.97M        |
| 预训练scOT-B      | 152.70M        |

表6列出了PDE-Transformer架构的超参数。

**表6**：PDE-Transformer的架构超参数

| 超参数名称                         | 取值                  |
| ---------------------------------- | --------------------- |
| 窗口大小（window_size）            | 8                     |
| 深度（depth）                      | [2, 4, 4, 6, 4, 4, 2] |
| 注意力头数（num_heads）            | 16                    |
| MLP比例（mlp_ratio）               | 4.0                   |
| 类别丢弃概率（class_dropout_prob） | 0.1                   |
| QKV偏置（qkv_bias）                | True                  |
| 激活函数（activation）             | GELU                  |


# 附录B 细节与补充结果
## B.1 训练方法
PDE-Transformer不同配置的训练损失收敛细节如图7所示。该图表明，在监督训练和基于扩散的训练方法中，B和L配置均能持续产生更低的训练损失。此外，左图显示扩散训练的损失波动更小，训练过程更稳定。

> **图7**：S、B、L三种配置在监督训练（左）和扩散训练（右）下的训练损失。

图8展示了采用流匹配的PDE-Transformer L配置在不同滚动步数下的nRMSE。当流积分步数超过约20步后，nRMSE均值开始稳定在较低水平，且步数增加时性能持续提升。

> **图8**：PDE-Transformer L配置下，nRMSE与推理步数的关系。我们采用显式欧拉法求解ODE以进行采样。


## B.2 下游任务
图9展示了与预训练Poseidon模型scOT-B随时间变化的详细对比结果。scOT-B和PDE-Transformer B的规模相近，在轨迹的最初几帧中性能相当。然而，scOT-B的预测误差随时间快速累积，而PDE-B即使在高难度任务中仍能生成稳定轨迹，误差持续保持较低水平。这种差异体现在图9左侧所示的$t=8$时刻快照中：scOT-B已偏离至以振荡为主的状态，而PDE-B的解仍与参考解高度接近。

> **图9**：预训练权重下，PDE-Transformer与scOT在活性物质任务上的平均轨迹nRMSE。左侧展示$t=8$时刻的代表性帧。


## B.3 数值结果
表7列出了图6中nRMSE的具体数值。此外，图10展示了每个下游任务在滚动预测过程中nRMSE的变化情况，结果表明预训练PDE-Transformer在整个滚动预测过程中始终优于基线模型。

**表7**：不同模型在下游任务上的平均滚动nRMSE

| 模型              | 活性物质 | 瑞利-贝纳德对流（RBC） | 剪切流 |
| ----------------- | -------- | ---------------------- | ------ |
| FNO               | 0.509    | 0.269                  | 0.274  |
| OFormer           | 0.752    | 0.520                  | 0.780  |
| scOT-S            | 0.546    | -                      | -      |
| 伽辽金Transformer | 0.638    | -                      | -      |
| PDE-S（从头训练） | 0.494    | 0.155                  | 0.199  |
| PDE-S（预训练）   | 0.455    | 0.104                  | 0.125  |

> **图10**：不同模型在各下游任务上的平均滚动nRMSE。


## B.4 基于LoRA的微调
低秩适配（Low-Rank Adaptation, LoRA）（Hu et al., 2022）是近年来解决基础模型微调计算挑战的热门技术。与其他参数高效微调（Parameter-Efficient Fine-Tuning, PEFT）方法类似，LoRA在大幅减少微调阶段可训练参数数量的同时，仅导致极小的性能损失。LoRA最初在大型语言模型微调中取得成功（Hu et al., 2022; Mao et al., 2025），随后在基于扩散的图像模型（Luo et al., 2023; Smith et al., 2024）中得到应用，目前已在计算机视觉（Aleem et al., 2024; Lin et al., 2024）、持续学习（Wistuba et al., 2023; Wei et al., 2025）和时间序列预测（Gupta et al., 2024b, a）等其他领域获得广泛关注。在科学深度学习领域，LoRA也已应用于蛋白质和材料工程（Buehler & Buehler, 2024; Zeng et al., 2024）。然而，LoRA在偏微分方程（PDE）基础模型中的应用仍较少探索。

给定网络输入$\mathbf{x}$和原始预训练权重$\mathbf{W}_0 \in \mathbb{R}^{d \times k}$，LoRA引入低秩矩阵$\Delta \mathbf{W}$来计算网络输出$\mathbf{y}$：
$$\mathbf{y} = \mathbf{W}_0 \mathbf{x} + \Delta \mathbf{W}\mathbf{x} = \mathbf{W}_0 \mathbf{x} + \frac{\alpha}{r} \mathbf{B}\mathbf{A}\mathbf{x} \tag{5}$$
其中，$\mathbf{B} \in \mathbb{R}^{d \times r}$，$\mathbf{A} \in \mathbb{R}^{r \times k}$，$r$为秩，$\alpha$（本研究中$\alpha=r$）为用于缩放低秩矩阵的系数。通过设置$r < \frac{d \cdot k}{d + k}$，并在微调过程中冻结预训练权重，可显著减少微调阶段的可训练参数数量。我们对网络中线性层和卷积层的权重启用LoRA，而保持偏差等其他参数不变。

图11展示了不同秩下PDE-Transformer各规模模型的参数数量变化。PDE-Transformer-S、-B、-L的参数数量均随秩的增加呈线性增长。

> **图11**：不同LoRA秩下，PDE-Transformer各规模模型的可训练参数数量。

我们在下游任务上评估了LoRA的性能：训练规模为B的PDE-Transformer，设置$r=96$（满足Transformer模型理论所需的最小秩（Zeng & Lee, 2024））。最终微调后的网络包含42.06M可训练参数，规模与S型模型相近。图12展示了B型模型基于LoRA微调的平均滚动预测性能：在活性物质任务中，LoRA-B模型优于无预训练的LoRA-S模型，表现出与预训练S型模型相近的优势；但在RBC任务中，LoRA-B模型与无预训练模型性能相当，未能达到预训练模型的水平。这些结果表明，尽管基于LoRA微调的模型性能可与同规模从头训练模型相当，但并非总能达到同规模全预训练模型的性能。

> **图12**：LoRA微调在下游任务上的性能。


# 附录C 预训练数据集
我们选择的数据集来源旨在确保涵盖广泛的PDE类型、高空间分辨率，且每个数据集内的模拟包含不同物理量。为模拟线性、反应扩散和非线性PDE，我们采用Exponax求解器（Koehler et al., 2024），该求解器实现了一系列指数时间差分龙格-库塔（Exponential Time Differencing Runge-Kutta, ETDRK）方法，可高效、统一地求解不同PDE。我们未直接使用作者提供的基准数据集APEBench，而是自主构建数据集，以确保更高的分辨率和更丰富的物理行为多样性（而非仅改变初始条件）。ETDRK方法在傅里叶空间中运行，因此不支持非周期域或复杂边界。在所有数据集处理中，我们均使用物理求解器接口（而非基于无量纲难度的接口），以便为任务嵌入提供更简洁、统一的物理量处理方式。

**表8**：使用Exponax模拟的数据集概述（上半部分，Koehler et al., 2024），包含线性PDE、反应扩散PDE和非线性PDE。每个数据集的维度通过模拟次数$s$、时间步$t$、物理场（或通道）$f$以及空间维度$x$和$y$描述。除指定变量外，每个模拟（$s$）的初始条件均不同。

| 数据集     | $s$  | $t$  | $f$  | $x$  | $y$  | $s$间变化的变量      | 测试集                  |
| ---------- | ---- | ---- | ---- | ---- | ---- | -------------------- | ----------------------- |
| diff       | 600  | 30   | 1    | 2048 | 2048 | 粘度（$x$、$y$方向） | $s \in [500, 600)$      |
| fisher     | 600  | 30   | 1    | 2048 | 2048 | 扩散系数、反应系数   | $s \in [500, 600)$      |
| sh         | 600  | 30   | 1    | 2048 | 2048 | 反应系数、临界值     | $s \in [500, 600)$      |
| gs-alpha   | 100  | 30   | 2    | 2048 | 2048 | 仅初始条件           | 独立集：$s=30$，$t=100$ |
| gs-beta    | 100  | 30   | 2    | 2048 | 2048 | 仅初始条件           | 独立集：$s=30$，$t=100$ |
| gs-gamma   | 100  | 30   | 2    | 2048 | 2048 | 仅初始条件           | 独立集：$s=30$，$t=100$ |
| gs-delta   | 100  | 30   | 2    | 2048 | 2048 | 仅初始条件           | $s \in [80, 100)$       |
| gs-epsilon | 100  | 30   | 2    | 2048 | 2048 | 仅初始条件           | 独立集：$s=30$，$t=100$ |
| gs-theta   | 100  | 30   | 2    | 2048 | 2048 | 仅初始条件           | $s \in [80, 100)$       |
| gs-iota    | 100  | 30   | 2    | 2048 | 2048 | 仅初始条件           | $s \in [80, 100)$       |
| gs-kappa   | 100  | 30   | 2    | 2048 | 2048 | 仅初始条件           | $s \in [80, 100)$       |
| burgers    | 600  | 30   | 2    | 2048 | 2048 | 粘度                 | $s \in [500, 600)$      |
| kdv        | 600  | 30   | 2    | 2048 | 2048 | 域范围、粘度         | $s \in [500, 600)$      |
| ks         | 600  | 30   | 1    | 2048 | 2048 | 域范围               | 独立集：$s=50$，$t=200$ |
| decay-turb | 600  | 30   | 1    | 2048 | 2048 | 粘度                 | 独立集：$s=50$，$t=200$ |
| kolm-flow  | 600  | 30   | 1    | 2048 | 2048 | 粘度                 | 独立集：$s=50$，$t=200$ |


## C.1 线性PDE
Exponax求解器框架（Koehler et al., 2024）提供了多种ETDRK方法，可高效模拟不同PDE。所选线性PDE相对简单且存在解析解，是构建更复杂PDE的重要基础。此处的每个线性PDE均可解释为受不同物理过程影响的标量（如密度）。除非特别说明，否则区间采样均采用均匀随机采样。图13展示了下文所述每个数据集的示例可视化结果。

除数据集间的物理参数差异外，每个模拟的初始条件均随机生成。默认情况下，初始条件通过以下三种方式构建（由Exponax实现，均匀随机选择一种）：  
1. 随机截断傅里叶级数初始化器：将多个傅里叶级数叠加，直至某一频率截止阈值，该阈值从$[2, 11)$中均匀随机选择整数；  
2. 高斯随机场初始化器：在傅里叶空间中生成幂律谱（能量随波数呈多项式衰减），幂律指数从$[2.3, 3.6)$中均匀随机选择；  
3. 扩散噪声初始化器：生成白噪声正态分布张量，随后进行扩散处理，得到的谱随强度率呈指数二次衰减，强度率从$[0.00005, 0.01)$中均匀随机选择。  

对于所有初始化器，生成的初始条件值均归一化，确保最大绝对值为1。对于向量物理量，每个向量分量的初始化器独立随机选择。


### 扩散方程（diff）
该数据集模拟密度场因扩散产生的空间耗散过程。在本设置中，扩散系数（又称粘度）随坐标轴方向变化。从视觉上看，该过程类似于密度场随时间模糊，高频信息被抑制。尽管看似简单，但扩散过程广泛存在于物理、生物、经济和统计等领域。

- 维度：$s=600$，$t=30$，$f=1$，$x=2048$，$y=2048$  
- 初始条件：随机截断傅里叶级数/高斯随机场/扩散噪声  
- 边界条件：周期性  
- 数据存储时间步：0.01  
- 模拟空间域大小：$[0, 1] \times [0, 1]$  
- 物理场：密度  
- 变化参数：粘度（$x$、$y$方向独立，取值范围$[0.005, 0.05)$）  
- 验证集：从$s \in [0, 500)$中随机选取15%的序列  
- 测试集：$s \in [500, 600)$的所有序列  

> **图13**：diff数据集的随机模拟示例。


## C.2 反应扩散PDE
Exponax求解器（Koehler et al., 2024）同样用于模拟不同的反应扩散PDE。这类PDE最常见于局部化学反应场景，但也存在于生物、物理或地质等领域，可用于建模行波和图案形成，通常描述一种或多种物质的浓度变化。


### 费希尔-克普方程（fisher）
该数据集包含基于费希尔-克普（Fisher-KPP）方程的反应扩散系统模拟，描述物质浓度随时间和空间的变化：反应过程由反应系数控制，空间扩散过程由扩散系数控制。该方程可应用于波传播、种群动态、生态学或等离子体物理等场景。图14展示了fisher数据集的示例可视化结果。

- 维度：$s=600$，$t=30$，$f=1$，$x=2048$，$y=2048$  
- 初始条件：随机截断傅里叶级数/高斯随机场/扩散噪声（并裁剪至$[0, 1]$范围）  
- 边界条件：周期性  
- 数据存储时间步：0.005  
- 模拟空间域大小：$[0, 1] \times [0, 1]$  
- 物理场：浓度  
- 变化参数：扩散系数（$[0.00005, 0.01)$）、反应系数（$[5, 15)$）  
- 验证集：从$s \in [0, 500)$中随机选取15%的序列  
- 测试集：$s \in [500, 600)$的所有序列  


### 斯威夫特-霍恩伯格方程（sh）
该数据集包含斯威夫特-霍恩伯格（Swift-Hohenberg）方程的模拟，描述特定图案形成过程，可用于解释弯曲弹性双层材料中的褶皱形态（如人类指纹形成——皮肤层间应力导致特征性褶皱）。图14展示了sh数据集的示例可视化结果。

- 维度：$s=600$，$t=30$，$f=1$，$x=2048$，$y=2048$  
- 初始条件：随机截断傅里叶级数/高斯随机场/扩散噪声  
- 边界条件：周期性  
- 数据存储时间步：0.5（模拟包含5个子步）  
- 模拟空间域大小：$[0, 20\pi] \times [0, 20\pi]$  
- 物理场：浓度  
- 变化参数：反应系数（$[0.4, 1)$）、临界值（$[0.8, 1.2)$）  
- 验证集：从$s \in [0, 500)$中随机选取15%的序列  
- 测试集：$s \in [500, 600)$的所有序列  

> **图14**：fisher（上）和sh（下）数据集的随机模拟示例。


### 格雷-斯科特方程（gs）
该数据集描述两种化学物质随时间的反应与扩散过程：物质$s_a$（浓度$c_a$）被反应消耗，同时按供给率补充；反应产物$s_b$（浓度$c_b$）按清除率从域中移除。根据供给率和清除率的配置，模拟会产生差异显著的稳态或非稳态行为及图案。因此，我们构建了多个子集：4个时间稳态配置（gs-delta、gs-theta、gs-iota、gs-kappa，最终状态基本不再变化）和4个时间非稳态配置（gs-alpha、gs-beta、gs-gamma、gs-epsilon，状态随时间持续演化）。对于非稳态情况，我们单独构建了具有更长时间滚动的测试集。图15展示了稳态配置的示例可视化结果，图16和17展示了非稳态配置及对应测试集的示例。更多细节可参考Pearson (1993)。

所有模拟中，物质的扩散系数固定为$d_a=0.00002$和$d_b=0.00001$。此外，这些数据集采用随机高斯斑点初始化器：在域中心60%区域（gs-kappa为20%）生成4个位置和方差随机的高斯斑点，且$c_a$的初始值为$c_b$的补集（即$c_a=1-c_b$）。


#### 稳态配置（gs-delta、gs-theta、gs-iota、gs-kappa）
- 维度（每个配置）：$s=100$，$t=30$，$f=2$，$x=2048$，$y=2048$  
- 初始条件：随机高斯斑点  
- 边界条件：周期性  
- 模拟时间步：1.0（所有配置）  
- 数据存储时间步：  
  - gs-delta：130.0  
  - gs-theta：200.0  
  - gs-iota：240.0  
  - gs-kappa：300.0  
- 预热步数（丢弃，按数据存储时间步计）：  
  - gs-delta：0  
  - gs-theta：0  
  - gs-iota：0  
  - gs-kappa：15  
- 模拟空间域大小：$[0, 2.5] \times [0, 2.5]$  
- 物理场：浓度$c_a$、浓度$c_b$  
- 变化参数：供给率和清除率由配置决定（即配置内仅初始条件变化）：  
  - gs-delta：供给率0.028，清除率0.056  
  - gs-theta：供给率0.040，清除率0.060  
  - gs-iota：供给率0.050，清除率0.0605  
  - gs-kappa：供给率0.052，清除率0.063  
- 验证集：从$s \in [0, 80)$中随机选取15%的序列  
- 测试集：$s \in [80, 100)$的所有序列  


#### 非稳态配置（gs-alpha、gs-beta、gs-gamma、gs-epsilon）
- 维度（每个配置）：$s=100$，$t=30$，$f=2$，$x=2048$，$y=2048$  
- 初始条件：随机高斯斑点  
- 边界条件：周期性  
- 模拟时间步：1.0（所有配置）  
- 数据存储时间步：  
  - gs-alpha：30.0  
  - gs-beta：30.0  
  - gs-gamma：75.0  
  - gs-epsilon：15.0  
- 预热步数（丢弃，按数据存储时间步计）：  
  - gs-alpha：75  
  - gs-beta：50  
  - gs-gamma：70  
  - gs-epsilon：300  
- 模拟空间域大小：$[0, 2.5] \times [0, 2.5]$  
- 物理场：浓度$c_a$、浓度$c_b$  
- 变化参数：供给率和清除率由配置决定（即配置内仅初始条件变化）：  
  - gs-alpha：供给率0.008，清除率0.046  
  - gs-beta：供给率0.020，清除率0.046  
  - gs-gamma：供给率0.024，清除率0.056  
  - gs-epsilon：供给率0.020，清除率0.056  
- 验证集：从$s \in [0, 100)$中随机选取15%的序列  
- 测试集：每个配置单独生成模拟，$s=30$，$t=100$，$f=2$，$x=2048$，$y=2048$  

> **图15**：格雷-斯科特反应扩散系统稳态配置的随机模拟示例：gs-delta、gs-theta、gs-iota、gs-kappa。

> **图16**：格雷-斯科特反应扩散系统非稳态配置的随机模拟示例：gs-alpha、gs-beta、gs-gamma、gs-epsilon。

> **图17**：gs-alpha、gs-beta、gs-gamma、gs-epsilon非稳态配置的长滚动测试集随机模拟示例。


## C.3 非线性PDE
非线性PDE的研究难度通常较大，甚至其解析解的存在性都是难题。此外，大多数通用方法无法跨场景应用，单个非线性PDE通常需作为独立问题研究。Exponax求解器（Koehler et al., 2024）提供了处理部分非线性PDE的工具，下文还将介绍其他数据来源。


### 伯格斯方程（burgers）
该数据集包含伯格斯方程的模拟，其类似于平流扩散问题，但描述的是流场自身因平流和扩散产生的变化，可能导致尖锐间断（激波）形成，模拟难度较大。伯格斯方程还可应用于非线性声学和交通流等场景。图18展示了burgers数据集的示例可视化结果。

- 维度：$s=600$，$t=30$，$f=2$，$x=2048$，$y=2048$  
- 初始条件：随机截断傅里叶级数/高斯随机场/扩散噪声  
- 边界条件：周期性  
- 数据存储时间步：0.01（模拟包含50个子步）  
- 模拟空间域大小：$[0, 1] \times [0, 1]$  
- 物理场：速度（$x$、$y$方向）  
- 变化参数：粘度（$[0.00005, 0.0003)$）  
- 验证集：从$s \in [0, 500)$中随机选取15%的序列  
- 测试集：$s \in [500, 600)$的所有序列  


### 科特韦格-德弗里斯方程（kdv）
该数据集包含周期性域上科特韦格-德弗里斯（Korteweg-de Vries, KdV）方程的模拟，用于建模浅水波。该方程的挑战在于能量会向高空间频率传递，形成形状和传播速度不变的孤立波（孤子）。所有模拟中，对流系数固定为-6，弥散系数固定为1。图18展示了kdv数据集的示例可视化结果。

- 维度：$s=600$，$t=30$，$f=2$，$x=2048$，$y=2048$  
- 初始条件：随机截断傅里叶级数/高斯随机场/扩散噪声  
- 边界条件：周期性  
- 数据存储时间步：0.05（模拟包含10个子步）  
- 模拟空间域大小：每个模拟不同  
- 物理场：速度（$x$、$y$方向）  
- 变化参数：域范围（$x$、$y$方向相同，$[30, 120)$）、粘度（$[0.00005, 0.001)$）  
- 验证集：从$s \in [0, 500)$中随机选取15%的序列  
- 测试集：$s \in [500, 600)$的所有序列  


### 库拉托莫-希瓦辛斯基方程（ks）
该数据集包含周期性域上库拉托莫-希瓦辛斯基（Kuramoto-Sivashinsky, KS）方程的模拟，用于建模燃烧过程中的热扩散火焰不稳定性，也可应用于反应扩散系统。该方程以混沌行为著称——初始条件略有差异的时间轨迹会随时间显著偏离。模拟中丢弃了初始瞬态阶段。为研究模型对混沌行为的处理能力，ks数据集采用具有更长滚动的测试集。图18展示了ks数据集的示例可视化结果。

- 维度：$s=600$，$t=30$，$f=1$，$x=2048$，$y=2048$  
- 初始条件：随机截断傅里叶级数/高斯随机场/扩散噪声  
- 边界条件：周期性  
- 数据存储时间步：0.5（模拟包含5个子步）  
- 预热步数（丢弃，按数据存储时间步计）：200  
- 模拟空间域大小：每个模拟不同  
- 物理场：密度  
- 变化参数：域范围（$x$、$y$方向相同，$[10, 130)$）  
- 验证集：从$s \in [0, 600)$中随机选取15%的序列  
- 测试集：单独生成模拟，$s=50$，$t=200$，$f=1$，$x=2048$，$y=2048$  

> **图18**：burgers、kdv、ks数据集及ks长滚动测试集的随机模拟示例。


### 衰减湍流（decay-turb）
该数据集包含周期性域上流函数-涡量形式的纳维-斯托克斯方程模拟，展现旋转湍流涡旋随时间衰减的过程。为更明显地观察衰减现象，该数据集采用具有更长滚动的测试集。图19展示了decay-turb数据集的示例可视化结果。

- 维度：$s=600$，$t=30$，$f=1$，$x=2048$，$y=2048$  
- 初始条件：随机截断傅里叶级数/高斯随机场/扩散噪声  
- 边界条件：周期性  
- 数据存储时间步：3.0（模拟包含500个子步）  
- 模拟空间域大小：$[0, 1] \times [0, 1]$  
- 物理场：涡量  
- 变化参数：粘度（$[0.00005, 0.0001)$）  
- 验证集：从$s \in [0, 600)$中随机选取15%的序列  
- 测试集：单独生成模拟，$s=50$，$t=200$，$f=1$，$x=2048$，$y=2048$  


### 柯尔莫哥洛夫流（kolm-flow）
该数据集包含周期性域上流函数-涡量形式的纳维-斯托克斯方程模拟。与上述衰减湍流不同，该模拟引入额外的强迫项，持续向系统注入能量以维持涡旋，导致时空混沌行为。模拟中丢弃了初始瞬态阶段（初始状态演变为条纹图案再形成涡旋的过程）。为测试模型对流动混沌行为的处理能力，该数据集采用具有更长滚动的测试集。图19展示了kolm-flow数据集的示例可视化结果。

- 维度：$s=600$，$t=30$，$f=1$，$x=2048$，$y=2048$  
- 初始条件：随机截断傅里叶级数/高斯随机场/扩散噪声  
- 边界条件：周期性  
- 数据存储时间步：0.3（模拟包含1500个子步）  
- 预热步数（丢弃，按数据存储时间步计）：50  
- 模拟空间域大小：$[0, 1] \times [0, 1]$  
- 物理场：涡量  
- 变化参数：粘度（$[0.0001, 0.001)$）  
- 验证集：从$s \in [0, 600)$中随机选取15%的序列  
- 测试集：单独生成模拟，$s=50$，$t=200$，$f=1$，$x=2048$，$y=2048$  

> **图19**：decay-turb、kolm-flow数据集及各自长滚动测试集的随机模拟示例。


# 附录D 自回归预测
表9和表10分别列出了所有数据集在1步和20步滚动预测下的nRMSE值。下文图20、21、22、23展示了PDE-L（采用MSE损失和混合通道（MC）训练）在测试数据集上从$t=0$到$t=27$的轨迹自回归预测结果。

**表9**：预训练数据集1步预测的nRMSE（nRMSE1）

| 方法       | diff   | burgers | kdv    | ks     | fisher | gs-alpha | gs-beta | gs-gamma | gs-delta | gs-epsilon | gs-theta | gs-iota | gs-kappa | sh     | decay-turb | kolm-flow |
| ---------- | ------ | ------- | ------ | ------ | ------ | -------- | ------- | -------- | -------- | ---------- | -------- | ------- | -------- | ------ | ---------- | --------- |
| DiT-S      | 0.0528 | 0.0262  | 0.0553 | 0.0609 | 0.0310 | 0.0388   | 0.0405  | 0.0942   | 0.0475   | 0.0284     | 0.0402   | 0.0365  | 0.0556   | 0.0856 | 0.2570     | 0.1209    |
| UDiT-S     | 0.0370 | 0.0191  | 0.0435 | 0.0178 | 0.0242 | 0.0224   | 0.0300  | 0.0302   | 0.0149   | 0.0161     | 0.0125   | 0.0150  | 0.0193   | 0.0519 | 0.2487     | 0.0715    |
| scOT-S     | 0.0674 | 0.0358  | 0.0536 | 0.0240 | 0.0230 | 0.0254   | 0.0400  | 0.0449   | 0.0271   | 0.0232     | 0.0212   | 0.0215  | 0.0311   | 0.0589 | 0.2114     | 0.0987    |
| FactFormer | 0.1440 | 0.0455  | 0.0823 | 0.0407 | 0.0231 | 0.0256   | 0.0347  | 0.0547   | 0.0343   | 0.0172     | 0.0255   | 0.0244  | 0.0410   | 0.0816 | 0.2248     | 0.1441    |
| UNet       | 0.0559 | 0.0392  | 0.0606 | 0.0469 | 0.0335 | 0.0441   | 0.0575  | 0.0845   | 0.0413   | 0.0416     | 0.0308   | 0.0355  | 0.0429   | 0.0829 | 0.2885     | 0.2385    |
| PDE-S      | 0.0370 | 0.0215  | 0.0480 | 0.0216 | 0.0247 | 0.0248   | 0.0295  | 0.0316   | 0.0172   | 0.0175     | 0.0129   | 0.0141  | 0.0231   | 0.0651 | 0.2361     | 0.0873    |
| PDE-B      | 0.0348 | 0.0162  | 0.0456 | 0.0145 | 0.0230 | 0.0270   | 0.0298  | 0.0256   | 0.0141   | 0.0140     | 0.0092   | 0.0095  | 0.0165   | 0.0640 | 0.2206     | 0.0578    |
| PDE-L      | 0.0349 | 0.0135  | 0.0455 | 0.0111 | 0.0235 | 0.0196   | 0.0260  | 0.0184   | 0.0113   | 0.0076     | 0.0063   | 0.0064  | 0.0134   | 0.0647 | 0.2113     | 0.0447    |

**表10**：预训练数据集20步预测的nRMSE（nRMSE20）

| 方法       | diff   | burgers | kdv    | ks     | fisher | gs-alpha | gs-beta | gs-gamma | gs-delta | gs-epsilon | gs-theta | gs-iota | gs-kappa | sh     | decay-turb | kolm-flow |
| ---------- | ------ | ------- | ------ | ------ | ------ | -------- | ------- | -------- | -------- | ---------- | -------- | ------- | -------- | ------ | ---------- | --------- |
| DiT-S      | 0.2677 | 0.8003  | 0.4915 | 1.6441 | 0.7041 | 1.4206   | 1.1469  | 1.0784   | 1.5140   | 1.2283     | 1.5285   | 1.2335  | 1.2387   | 0.9253 | 0.8148     | 1.2348    |
| UDiT-S     | 0.2035 | 0.1982  | 0.3157 | 0.9865 | 0.6144 | 0.6983   | 0.7592  | 0.7958   | 1.0605   | 0.3591     | 1.0323   | 0.9341  | 0.7869   | 0.6882 | 1.0018     | 0.8575    |
| scOT-S     | 0.8773 | 0.3948  | 0.4351 | 1.1377 | 0.7017 | 0.8794   | 0.9355  | 0.9100   | 1.1182   | 0.4704     | 1.0945   | 1.0103  | 1.0584   | 0.6872 | 1.0455     | 0.9866    |
| FactFormer | 1.9913 | 0.6704  | 0.8224 | 1.3988 | 0.6370 | 0.8579   | 0.8107  | 0.9878   | 1.1288   | 0.5425     | 1.0834   | 1.0160  | 1.2319   | 0.8426 | 1.2972     | 1.1670    |
| UNet       | 0.5263 | 0.7142  | 0.5829 | 1.3061 | 0.7327 | 1.0608   | 1.1585  | 0.9748   | 1.0159   | 0.7716     | 0.9879   | 0.9197  | 0.9803   | 0.8322 | 0.9746     | 1.2597    |
| PDE-S      | 0.2095 | 0.1879  | 0.3383 | 0.9741 | 0.5689 | 0.6696   | 0.6643  | 0.7887   | 1.0686   | 0.3196     | 1.0247   | 0.7601  | 0.7272   | 0.6757 | 0.9141     | 0.9370    |
| PDE-B      | 0.2090 | 0.1142  | 0.3204 | 0.8599 | 0.6131 | 0.6186   | 0.6110  | 0.7558   | 1.0253   | 0.3636     | 0.9902   | 0.6432  | 0.6435   | 0.6354 | 0.7333     | 0.8005    |
| PDE-L      | 0.2104 | 0.0921  | 0.3004 | 0.7357 | 0.5731 | 0.5554   | 0.5924  | 0.7083   | 0.9916   | 0.1997     | 0.9382   | 0.5333  | 0.5572   | 0.6112 | 0.7011     | 0.7102    |

> **图20**：模型在diff、fisher、sh数据集上的预测结果可视化。

> **图21**：模型在gs-alpha、gs-beta、gs-gamma、gs-epsilon数据集上的预测结果可视化。

> **图22**：模型在gs-delta、gs-theta、gs-iota、gs-kappa数据集上的预测结果可视化。

> **图23**：模型在burgers、kdv、ks、decay-turb、kolm-flow数据集上的预测结果可视化。


# 附录E 下游任务
## E.1 数据集
我们选取三个具有挑战性的PDE预测任务作为下游任务：活性物质、瑞利-贝纳德对流和剪切流，所有模拟数据均来自Well数据集（Ohana et al., 2024）。图24展示了这三个任务的可视化示例。Well数据集中每个数据集均已预先划分训练集、验证集和测试集，我们从中分别随机选取42条、8条和10条轨迹用于训练、验证和测试，并将每条轨迹随机截断为30帧。各数据集细节如下：


### 活性物质（Active Matter）
该数据集模拟浸没在斯托克斯流中的杆状生物活性粒子。活性粒子将化学能转化为机械能，产生在系统中传递的应力；此外，粒子的协同作用导致流场内部出现复杂行为。数据集关键特征概述如下（更多细节见Maddu et al., 2024）：

- 边界条件：周期性  
- 数据存储时间步：0.25  
- 模拟空间域大小：$[0, 10] \times [0, 10]$  
- 空间分辨率：$x=256$，$y=256$  
- 物理场：浓度、速度（$x$、$y$方向）（Well数据集原始的取向场（$xx$、$xy$、$yx$、$yy$）和应变场（$xx$、$xy$、$yx$、$yy$）在本次测试中舍弃）  


### 瑞利-贝纳德对流（Rayleigh-Bénard Convection, RBC）
该数据集包含水平周期性域上的瑞利-贝纳德对流模拟，结合流体动力学与热力学，模拟上下板温度差导致的对流胞形成过程。浮力、热传导和粘度的共同作用使流场呈现复杂行为，包含边界层和涡旋。数据集关键特征概述如下（更多细节见Burns et al., 2020）：

- 边界条件：$x$方向周期性，$y$方向固壁  
- 数据存储时间步：0.25  
- 模拟空间域大小：$[0, 4] \times [0, 1]$  
- 空间分辨率：$x=512$，$y=128$  
- 物理场：浮力、压力、速度（$x$、$y$方向）  


### 剪切流（Shear Flow）
该数据集包含剪切流配置下周期性不可压缩纳维-斯托克斯方程的模拟：两层流体以不同速度相互滑动。预测不同雷诺数和施密特数下形成的涡旋，在汽车、生物医学和空气动力学领域具有重要应用。数据集关键特征概述如下（更多细节见Burns et al., 2020）：

- 边界条件：周期性  
- 数据存储时间步：0.1  
- 模拟空间域大小：$[0, 1] \times [0, 2]$  
- 空间分辨率：$x=512$，$y=256$  
- 物理场：密度、压力、速度（$x$、$y$方向）  

> **图24**：活性物质、瑞利-贝纳德对流、剪切流数据集的随机模拟示例。


## E.2 自回归预测
图25、26、27展示了预训练PDE-S（采用分离通道（SC））在各数据集训练集上的自回归预测结果。

> **图25**：活性物质任务。预训练PDE-S（SC）的自回归预测结果。

> **图26**：瑞利-贝纳德对流任务。预训练PDE-S（SC）的自回归预测结果。

> **图27**：剪切流任务。预训练PDE-S（SC）的自回归预测结果。