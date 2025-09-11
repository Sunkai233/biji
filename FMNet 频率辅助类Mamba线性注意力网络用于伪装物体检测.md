# FMNet: 频率辅助类Mamba线性注意力网络用于伪装物体检测

**arXiv:2503.11030v2 [cs.CV] 31 May 2025**

**作者**: 邓明^1,^*^, 孙思晋^2,4,^*^, 李子豪^1^, 胡晓川^3^, 吴星^1,^†^  
^1^ 上海大学  
^2^ 新加坡科技研究局  
^3^ 电子科技大学  
^4^ 新加坡国立大学  
^*^ 平等贡献 ^†^ 通讯作者  

## 摘要

伪装物体检测（Camouflaged Object Detection, COD）由于伪装物体与其周围环境的高度相似性而具有挑战性，这使得识别变得复杂。现有的方法主要依赖于空间局部特征，难以捕捉全局信息，而Transformer方法则增加了计算成本。为解决这一问题，提出了**频率辅助类Mamba线性注意力网络（FMNet）**，该网络利用频率域学习高效捕捉全局特征，缓解物体与背景之间的模糊性。FMNet引入了**多尺度频率辅助类Mamba线性注意力（MFM）模块**，通过多尺度结构整合频率和空间特征，以处理尺度变化并降低计算复杂性。此外，**金字塔频率注意力提取（PFAE）模块**和**频率反向解码器（FRD）**增强了语义并重构特征。实验结果表明，FMNet在多个COD数据集上优于现有方法，展示了其在性能和效率方面的优势。代码可在以下链接获取：`https://github.com/Chranos/FMNet`。

**关键词**：伪装物体检测，语义增强，类Mamba线性注意力，频率辅助

{

### 通俗解释版摘要

**伪装物体检测（COD）** 的难点在于：伪装物体和背景长得几乎一样，像变色龙趴在树叶上，很难分辨。
 传统方法一般只关注局部细节（局部纹理、边缘），容易被“迷惑”；而基于 Transformer 的方法虽然能捕捉全局信息，但计算量太大，效率不高。

为此，作者提出了一种新模型 **FMNet（频率辅助类 Mamba 线性注意力网络）**。
 核心思路是：

- **利用频率信息（在频率域看图像）** 来更高效地捕捉全局特征，把“背景与物体之间的微妙差异”挖掘出来。
- 网络的关键模块：
  1. **MFM 模块（多尺度频率辅助类 Mamba 线性注意力）**：结合不同尺度的空间特征和频率特征，既能适应大小变化，又能降低计算开销。
  2. **PFAE 模块（金字塔频率注意力提取）**：进一步增强语义信息。
  3. **FRD 模块（频率反向解码器）**：帮助恢复和重建清晰的目标特征。

**实验结果**：在多个 COD 数据集上，FMNet 的检测效果比现有方法更好，同时运行效率也更高。

代码开源在 GitHub 👉 `https://github.com/Chranos/FMNet`

}

## I. 引言

伪装物体检测（COD）是一项极具挑战性的任务，旨在准确检测与周围环境高度相似的隐藏物体。COD在医疗影像中的病灶分割和工业场景中的缺陷检测等领域具有重要的应用价值。

传统的COD方法主要依赖于手工特征提取[1][2]。虽然这些方法在特定场景下取得了一定性能，但其鲁棒性有限。随着开源COD数据集[3][4][5]的可用性，基于深度学习的COD方法通过自动提取丰富特征显示出显著优势[3][6]。近期研究表明，优化特征表示可以有效应对伪装物体在目标尺寸变化、环境遮挡和边界模糊等方面的挑战。已经提出了多尺度特征提取[7][8]和边界引导[9]等策略。然而，大多数现有方法仍局限于优化局部特征，难以有效捕捉全局信息。这种局限性在检测具有显著尺寸变化或遮挡的伪装物体时尤为明显。此外，空间域特征容易受到复杂背景的干扰，这通常是由于过分强调局部细节或单个像素位置所致。

![图1. 与传统注意力机制方法的比较](attachment://fig1_comparison.png)

Transformer方法[10]因其建模长距离依赖的能力而在伪装物体检测中得到广泛应用。然而，Transformer的高计算成本和复杂网络结构[11][12]显著限制了其实用性。

频率域特征[13][14]因其固有的全局建模能力而在伪装物体检测中受到广泛关注，这类特征能有效抑制背景噪声并提高伪装物体的语义清晰度。这一优势在边界模糊和遮挡场景中尤为明显。然而，频繁在频率域和空间域之间的转换会导致计算复杂度和参数开销的增加。

近年来，Mamba方法[15][16]凭借其高效的注意力机制和轻量级设计显著降低了计算成本，并显示出巨大的应用潜力。然而，Mamba方法在伪装物体检测中的潜力尚未被充分探索。

基于上述讨论，提出了一种名为**频率辅助类Mamba线性注意力网络（FMNet）**的新方法。该方法整合了频率域和空间域的特征信息，引入多尺度策略进一步提取全局信息，并利用**类Mamba线性注意力（MLLA）**[17]优化基于Transformer的COD方法。我们的方法与基于Transformer的COD方法的比较如图1所示。本文的主要贡献总结如下：

- 提出了FMNet，与传统的Transformer方法不同，通过引入多尺度和频率辅助策略显著提升伪装物体检测的性能。
- 设计了**多尺度频率辅助类Mamba线性注意力（MFM）模块**，协同频率域和空间域特征，提供更全面的图像特性理解，同时有效降低计算复杂性。
- 创新开发了**金字塔频率注意力提取（PFAE）模块**和**频率反向解码器（FRD）模块**，以增强频率域特征的表示并整合多层次信息，进一步提升检测性能。

{

#### 频率域特征

频率特征（比如傅里叶变换后的表示）在 COD 中也逐渐受到关注：

- 天然适合做 **全局建模**
- 能抑制背景噪声，让目标更清晰
- 特别适合处理 **边界模糊** 和 **遮挡** 的情况

缺点是：频繁在 **空间域 ↔ 频率域** 来回切换，计算代价比较高。

------

#### Mamba方法

Mamba 是一种新兴的高效注意力机制：

- 轻量级，计算成本低
- 已经在一些任务里展现了潜力
- 但在 COD 里几乎没人探索过

------

#### 我们的工作

基于这些背景，作者提出了 **FMNet（频率辅助类 Mamba 线性注意力网络）**。

核心创新点：

1. **FMNet 框架**：结合 **频率特征 + 空间特征**，多尺度处理 → 更强的全局建模能力，但计算成本比 Transformer 小。
2. **MFM 模块**：多尺度频率辅助类 Mamba 注意力，把频率和空间结合起来，兼顾细节与整体。
3. **PFAE 模块**（金字塔频率注意力提取）：进一步强化频率特征表示。
4. **FRD 模块**（频率反向解码器）：在解码阶段充分利用频率信息，帮助重建更清晰的目标。

}

## II. 方法

### A. 概述

FMNet的整体框架如图2所示。给定输入图像 \( I_c \in \mathbb{R}^{H \times W \times 3} \)，采用结合Transformer和Mamba的混合骨干网络[18]高效提取初始特征 \( E_i \)，其中每个特征图的分辨率逐渐缩小至原始大小的 \( \frac{1}{2^{i+1}} \)。受先前工作[14][19]的启发，采用PFAE模块提取多尺度融合特征 \( E_5 \)。为高效建模特征域的长距离依赖，使用MFM模块优化全局上下文并生成优化特征 \( F_i \)。最后，FRD模块聚合这些多层次特征以生成最终特征图 \( G_i \。

![图2. FMNet框架概览](attachment://fig2_fmnet_framework.png)

### B. 金字塔频率注意力提取

PFAE模块如图2所示，通过在频率域整合注意力机制，更有效地提取高频特征。具体来说，输入特征 \( E_4 \) 首先通过 \( 1 \times 1 \) 卷积减少通道数，得到 \( \hat{E}_4 \)。随后，加载四个具有不同膨胀率的膨胀卷积层作为四个分支：

\[
\tilde{I}_n = C1AC_z(\hat{E}_4 + I_{n-1}), \quad z = 2^n - 1, \quad n \geq 2
\]

每个分支包含一个频率域注意力模块，该模块使用快速傅里叶变换（FFT）生成查询 \( Q \)、键 \( K \) 和值 \( V \)。在重塑后，查询和键进行点乘生成转置注意力图 \( A'_f \):

\[
Q, K, V = fft(\hat{E}_4), \quad A'_f = \tilde{Q} \odot \tilde{K}
\]

其中 \( \hat{Q} \) 和 \( \hat{K} \) 是对查询 \( Q \) 和键 \( K \) 应用重塑操作的结果。由于 \( A_f \) 是复数类型，分别激活其实部和虚部，然后合并：

\[
A_{re}^f = \frac{A'_f + coj(A'_f)}{2}, \quad A_{im}^f = \frac{A'_f - coj(A'_f)}{2i}
\]

\[
A_f = \Theta(\text{Sof}(A_{re}^f), \text{Sof}(A_{im}^f))
\]

其中 \( \Theta(\cdot, \cdot) \) 表示将实部和虚部组合成复数的函数，\( \text{Sof}(\cdot) \) 表示Softmax函数。随后，注意力图 \( A_f \) 与 \( V \) 进行点乘实现加权优化，并通过逆快速傅里叶变换（IFFT）转换回原始域。在此过程中，引入了**频率权重模块（FWM）**（将在II-C节讨论）执行残差连接。最后，应用 \( 1 \times 1 \) 卷积，并与FFT变换前的特征进行残差连接，生成混合特征 \( I_n \):

\[
I_n = C1\Phi(\|ifft(A_f \odot V)\|, FWM) + \tilde{I}_n
\]

其中 \( \Phi \) 表示拼接操作。通过卷积操作生成输出 \( E_5 \):

\[
E_5 = C3C1(\Phi(J_1, J_2, J_3, J_4) + P_{128}^4)
\]

其中 \( C_k \) 表示核大小为 \( k \times k \) 的卷积操作。

### C. 多尺度频率辅助类Mamba线性注意力

多头注意力模块[10]将输入查询、键和值分解为 \( N_h \) 部分，沿通道维度执行自注意力学习，公式如下：

\[
A = \sigma \left( \frac{Q_i K^T_i}{\sqrt{D_h}} \right) V_i
\]

其中 \( \sigma \) 表示Sigmoid函数，对于输入特征图 \( X \in \mathbb{R}^{H \times W \times D} \)，计算复杂性为 \( O(H^2 W^2 D) \)。显然，随着图像分辨率的增加，复杂性迅速增加。因此，许多采用类似模块的伪装物体检测网络不可避免地面临高计算成本[12][20]。

受解决这一问题的启发[21][15]，提出了一个名为**多尺度频率辅助类Mamba线性注意力（MFM）**的模块。

**线性化注意力**：线性注意力[21]用线性归一化替换非线性Softmax函数，并在查询 \( q_j \) 和键 \( k_k \) 中引入额外的核函数 \( \phi \)。输出 \( y_j \) 可实现为自回归模型，表达如下：

\[
S_j = S_{j-1} + K_j^\top V_j, \quad Z_j = Z_{j-1} + K_j^\top, \quad y_j = \frac{Q_j S_j}{Q_j Z_j}
\]

**选择性状态空间模型**：经典状态空间模型通过隐藏状态 \( h(t) \in \mathbb{R}^{d \times 1} \) 将输入 \( x(t) \in \mathbb{R} \) 映射到输出 \( y(t) \in \mathbb{R} \)。Mamba[15]引入了选择性状态空间模型，使用零阶保持（ZOH）离散化参数。MLLA[17]将Mamba重写为以下等效形式：

\[
h_j = \tilde{A}_j \odot h_{j-1} + B_j (\Delta_j \odot x_j), \quad y_j = C_j h_j + D \odot x_j
\]

其中 \( \odot \) 表示逐元素乘法，\( B_j, C_j, \Delta_j \) 由输入 \( x_j, \Delta_j \in \mathbb{R}^{1 \times C}, y_j \in \mathbb{R}^{1 \times C} \) 派生。可以看出，公式(8)和(9)有密切关系，具体为：\( h_j \sim S_j \in \mathbb{R}^{d \times C}, B_j \sim K_j^\top \in \mathbb{R}^{d \times 1}, x_j \sim V_j \in \mathbb{R}^{1 \times d}, C_j \sim Q_j \in \mathbb{R}^{1 \times d} \)，\( \tilde{A}_i \) 在选择性状态空间模型（SSM）中扮演遗忘门的角色。MLLA模块继承了Mamba的优点，采用类似Mamba的结构并引入遗忘门机制。同时，为了更好地适应视觉任务，MLLA用位置编码LePE[26]、RoPE[27]和CPE[28]替换传统遗忘门：

\[
LePE(x) = x + DWConv(x)W_L
\]

\[
RoPE(x_m, \theta_i) = x_m \cdot (\cos(m \theta_i) + \sin(m \theta_i))
\]

\[
L_a = Att(RoPE(Q), RoPE(K), V + LePE(V))
\]

其中 \( W_L \) 是可学习权重矩阵，\( DWConv \) 表示深度卷积，\( x_m \) 是输入的第 \( m \) 维，\( \theta_i \) 是与位置相关的角度。线性注意力模块的复杂性为 \( O(NCd) = O(HWDD_h) \)。

**频率辅助多尺度MLLA**：MFM模块如图3所示，为清晰起见，省略了重塑操作。通过优化MLLA，设计了一个类似[29]的多尺度结构，具有可接受的复杂性。输入特征 \( E_i \) 首先通过CPE和层归一化处理，得到张量 \( \tilde{E}_i \)。然后，沿通道维度分割并通过 \( 1 \times 1 \) 卷积处理，之后通过 \( n \times n \) 深度卷积生成不同尺度的张量。这些张量通过线性注意力层后拼接在一起。整个过程表示为：

\[
A_i^n = L_a(R(\sigma(D_n C_1(\tilde{E}_i))))
\]

\[
A_i = \Phi(A_i^3, A_i^5), \quad F_i^1 = L(A_i \odot R(\sigma(C_1(\tilde{E}_i))))
\]

其中 \( R \) 表示重塑操作，\( D_n \) 表示深度卷积核，\( C_1 \) 表示大小为1的卷积核，\( \Phi \) 表示拼接操作，\( F_i^1 \) 是第一阶段的特征图输出。

还设计了一个**频率权重模块（FWM）**，通过频率残差连接增强频率域信息的表示：

\[
FWM = \|ifft(W(fft(X)) * fft(X))\|
\]

其中 \( W(\cdot) \) 表示一系列操作，包括卷积、批归一化、GELU、卷积和Sigmoid函数，依次应用。通过多尺度MLLA模块后，得到 \( F_i^2 \):

\[
F_i^2 = CPE(F_i^1 + FWM(E_i) + E_i)
\]

最终，MFM的整体输出 \( F_i \) 可通过以下公式获得：

\[
F_i = F_i^2 + LN(Mlp(F_i^2)) + FWM(F_i^2)
\]

其中 \( LN \) 表示层归一化。

![图3. 多尺度频率辅助类Mamba线性注意力（MFM）细节](attachment://fig3_mfm_details.png)

### D. 频率反向解码器

FRD模块如图4所示，与现有COD方法[6][12]不同。FRD的输入包括两部分：辅助输入和主输入 \( F_i \)。高级特征图 \( G_{i+1}(F_5) \) 用作辅助输入。首先，应用双线性插值和通道维度扩展，确保辅助输入与主输入的大小和通道数匹配。随后，辅助输入与主输入沿通道维度拼接，得到特征 \( G_i^1 \):

\[
G_i^1 = \Phi(F_i, Ex(G_{i+1}), \dots, Ex(Z)), \quad Z = \{G_4, F_5\}
\]

对辅助特征应用反向注意力，生成频率-空间混合反向注意力图 \( RA \)，然后用于生成频率优化的反向特征 \( G_i^2 \):

\[
RA = \sum (1 - \sigma(G_{i+1})) + (1 - \sigma(\|fft(G_{i+1})\|))
\]

\[
G_i^2 = RA * F_i
\]

最后，\( G_i^1 \) 经过一系列卷积操作，并与 \( G_i^2 \) 整合，生成最终特征 \( G_i \)，\( G_i \) 将作为下一层FRD的辅助输入：

\[
G_i = C3\Phi(G_i^1, Con(G_i^2)) + F_i
\]

![图4. 频率反向解码器细节](attachment://fig4_frd_details.png)

### E. 损失函数

与[6][12]类似，使用加权二元交叉熵（BCE）和加权交并比（IoU）[30]作为损失函数，监督多层次特征 \( G_i \)。损失函数定义如下：

\[
L_{all} = \sum_{i=1}^5 2^{1-i} \left( L_w^{bce}(G_i, GT) + L_w^{iou}(G_i, GT) \right)
\]

其中 \( L_w^{bce} \) 和 \( L_w^{iou} \) 分别表示加权BCE和IoU损失函数。

{

#### A. 整体框架

FMNet 的整体框架（图2）可以理解为 **“频率+空间的双通道特征处理流水线”**。

流程是：

1. 输入图像 $I_c$ 先通过一个 **混合骨干网络**（结合 Transformer 和 Mamba）得到多层特征 $E_i$，分辨率逐步降低。
2. **PFAE 模块** 从这些特征中提取多尺度频率信息，生成融合特征 $E_5$。
3. **MFM 模块**（多尺度频率辅助 Mamba 线性注意力）进一步建模全局上下文，得到优化后的特征 $F_i$。
4. **FRD 模块**（频率反向解码器）把多层次特征聚合起来，输出最终预测图 $G_i$。

------

#### B. 金字塔频率注意力提取（PFAE）

PFAE 的目标：在 **频率域** 提取更有效的特征。

- 输入：特征 $E_4$，先用 $1 \times 1$ 卷积降维。
- 然后分成 4 个分支，每个分支用不同膨胀率的卷积（多尺度感受野）。

数学表达：
$$
\tilde{I}_n = C1AC_z(\hat{E}_4 + I_{n-1}), \quad z=2^n-1
$$

- 每个分支进入 **FFT 频率注意力模块**：

  - 把特征做 FFT → 得到 $Q,K,V$

  - 点乘得到注意力图 $A_f'$：
    $$
    A_f' = \tilde{Q}\odot \tilde{K}
    $$

  - 分别取实部和虚部：
    $$
    A_{re}^f = \frac{A_f' + \text{conj}(A_f')}{2}, \quad
    A_{im}^f = \frac{A_f' - \text{conj}(A_f')}{2i}
    $$

  - 对实部和虚部分别做 Softmax，再合并成复数注意力图 $A_f$。

  - 与 $V$ 相乘，再做 IFFT 转回空间域。

最后：

- 加入 **FWM（频率权重模块）** 做残差增强
- 拼接多分支特征，输出 $E_5$

------

#### C. 多尺度频率辅助 Mamba 线性注意力（MFM）

**问题背景**：

- 普通多头注意力复杂度很高，计算量 $O(H^2W^2D)$，图像大时不可用。
- 线性注意力（Linear Attention）可以降低复杂度。

**线性注意力**：
$$
S_j = S_{j-1} + K_j^\top V_j, \quad 
Z_j = Z_{j-1} + K_j^\top, \quad
y_j = \frac{Q_j S_j}{Q_j Z_j}
$$
它通过累积形式避免了对所有像素两两计算。

**Mamba 思路**：

- 引入状态空间模型（SSM）
- 形式类似注意力，但用“遗忘门”机制管理信息流：

$$
h_j = \tilde{A}_j \odot h_{j-1} + B_j (\Delta_j \odot x_j), \quad
y_j = C_j h_j + D \odot x_j
$$

**改进：MLLA**（类 Mamba 线性注意力）：

- 结合 Mamba 的轻量特性

- 加入位置编码（LePE, RoPE, CPE）让其适应视觉任务
  $$
  L_a = Att(RoPE(Q), RoPE(K), V + LePE(V))
  $$

**MFM 模块设计**：

1. 输入特征 $E_i$ 先归一化、卷积
2. 用多尺度深度卷积生成不同感受野的特征
3. 分别送进线性注意力层，再拼接

公式：
$$
A_i^n = L_a(R(\sigma(D_n C_1(\tilde{E}_i))))
$$

1. 加入 **FWM（频率权重模块）**：

$$
FWM = \|ifft(W(fft(X)) * fft(X))\|
$$

1. 残差增强：

$$
F_i^2 = CPE(F_i^1 + FWM(E_i) + E_i)
$$

1. 输出：

$$
F_i = F_i^2 + LN(Mlp(F_i^2)) + FWM(F_i^2)
$$

------

#### D. 频率反向解码器（FRD）

**目标**：利用频率特征引导解码，让目标区域更清晰。

步骤：

1. 主输入：MFM 输出的特征 $F_i$
2. 辅助输入：高层特征 $G_{i+1}$，插值+扩展后与 $F_i$ 拼接：

$$
G_i^1 = \Phi(F_i, Ex(G_{i+1}), \dots)
$$

1. 计算 **反向注意力图**：同时利用空间域和频率域信息

$$
RA = \sum (1 - \sigma(G_{i+1})) + (1 - \sigma(\|fft(G_{i+1})\|))
$$

1. 用 RA 加权主输入：

$$
G_i^2 = RA * F_i
$$

1. 融合：

$$
G_i = C3\Phi(G_i^1, Con(G_i^2)) + F_i
$$

------

#### E. 损失函数

采用加权 BCE + IoU：
$$
L_{all} = \sum_{i=1}^5 2^{1-i} \Big( L_w^{bce}(G_i, GT) + L_w^{iou}(G_i, GT) \Big)
$$
特点：对浅层特征赋予更大权重（$2^{1-i}$），保证细节与语义兼顾。

------

#### 总结

- **PFAE** → 在频率域提取多尺度特征
- **MFM** → 结合 Mamba 线性注意力 + 多尺度频率增强，低算力下实现全局建模
- **FRD** → 利用频率反向注意力改进解码，增强目标和背景的区分度
- **损失函数** → 多层次监督，兼顾边界和区域

}

## III. 实验

### A. 实验设置

FMNet在CAMO[4]、COD10K[3]和NC4K[5]数据集上进行评估，使用CAMO[4]和COD10K[3]数据集中的4000张图像进行训练。FMNet在三块NVIDIA GTX 4090 GPU（24GB）上训练，并使用预训练的MambaVision作为编码器提取初始特征。输入图像调整为 \( 416 \times 416 \)，并应用随机水平翻转和裁剪等数据增强技术。训练时，批量大小设为30，训练100个周期。初始学习率设为 \( 1 \times 10^{-4} \)，使用Adam优化器，每50个周期将学习率降低10倍。采用以下指标评估方法性能：S-测度（\( S_m \)）、E-测度（\( E_m \)）、平均F-测度（\( F_\phi \)）和平均绝对误差（\( M \)）。

### B. 与最先进方法的比较

FMNet与8种最先进的方法进行比较，包括JSOCOD[22]、UGTR[23]、ZoomNet[8]、SINet-V2[7]、FRINet[24]、FSPNet[12]、VSCode[25]和GLCONet[11]。本文中所有预测结果由原始作者提供或从开源代码获得。

FMNet与其他SOTA方法的定量比较总结在表I中。结果显示，该方法在各个评估指标上优于其他方法，展示了卓越的性能。此外，表III提供了FMNet与其他方法在参数和FLOPs方面的比较。可以看出，FMNet相较于复杂网络显示出效率优势。

![表I. 与三个COD数据集上的最先进方法的比较](attachment://table1_comparison.png)

视觉比较如图5所示，包括大物体、小物体和遮挡物体的分割结果。结果表明，FMNet在所有场景下都展示了鲁棒性能。

![图5. 与最先进方法的预测图视觉比较](attachment://fig5_visual_comparison.png)

### C. 消融研究

FMNet方法的组件定量结果如表II所示。具体验证了PFAE、MFM和FRD的有效性及其对参数和FLOPs的影响。实验结果表明，将这些模块添加到基线中显著提高了预测性能。从表II(e)、表II(f)和表II(g)可以看出，这三个组件彼此兼容性良好。此外，效率消融研究显示，MLLA模块消耗最多参数和FLOPs，同时对性能提升贡献最大，进一步证明其是优化的关键部分。

![表II. 消融分析](attachment://table2_ablation.png)

![表III. FMNet与其他COD方法的效率分析](attachment://table3_efficiency.png)

## IV. 结论

本文提出了一种用于伪装物体检测的新方法FMNet，其核心是高性能的多尺度频率辅助模块（MFM）。FMNet整合了空间域和频率域的特征信息，实现了更精确的物体分割。此外，设计了金字塔频率注意力提取（PFAE）模块以提取多尺度特征，以及频率反向解码器（FRD）进行跨层聚合和反向优化，进一步提升模型性能。广泛的比较实验表明，FMNet在多个基准数据集上优于现有SOTA方法。未来，我们将改进FMNet对复杂环境的适应性，并优化其在边缘设备上的部署，以增强其实际应用。

## 致谢

本研究得到国家自然科学基金（No. 62172267号）和教育部硅酸盐文物保护重点实验室（上海大学）项目（No. SCRC2023ZZ02ZD）的资助。

## 参考文献

[1] Chennamsetty Pulla Rao, Aavula Guruva Reddy, and C. B. Rama Rao, “Camouflaged object detection for machine vision applications,” *International Journal of Speech Technology*, vol. 23, pp. 327–335, 2020.  
[2] Meirav Galun, Eitan Sharon, Ronen Basri, and Achi Brandt, “Texture segmentation by multiscale aggregation of filter responses and shape elements,” *Proceedings Ninth IEEE International Conference on Computer Vision*, pp. 716–723 vol.1, 2003.  
[3] Deng-Ping Fan, Ge-Peng Ji, Guolei Sun, Ming-Ming Cheng, Jianbing Shen, and Ling Shao, “Camouflaged object detection,” *2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 2774–2784, 2020.  
[4] Trung-Nghia Le, Tam V. Nguyen, Zhongliang Nie, Minh-Triet Tran, and Akihiro Sugimoto, “Anabranch network for camouflaged object segmentation,” *Comput. Vis. Image Underst.*, vol. 184, pp. 45–56, 2019.  
[5] Yunqiu Lyu, Jing Zhang, Yuchao Dai, Aixuan Li, Bowen Liu, Nick Barnes, and Deng-Ping Fan, “Simultaneously localize, segment and rank the camouflaged objects,” *2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 11586–11596, 2021.  
[6] Hongwei Zhu, Peng Li, Haoran Xie, Xu Yan, Dong Liang, Dapeng Chen, Mingqiang Wei, and Jing Qin, “I can find you! boundary-guided separated attention network for camouflaged object detection,” in *AAAI Conference on Artificial Intelligence*, 2022.  
[7] Deng-Ping Fan, Ge-Peng Ji, Ming-Ming Cheng, and Ling Shao, “Concealed object detection,” *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 44, no. 10, pp. 6024–6042, 2022.  
[8] Youwei Pang, Xiaoqi Zhao, Tian-Zhu Xiang, Zhang Lihe, and Huchuan Lu, “Zoom in and out: A mixed-scale triplet network for camouflaged object detection,” *2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 2150–2160, 2022.  
[9] Yujia Sun, Shuo Wang, Chenglizhao Chen, and Tian-Zhu Xiang, “Boundary-guided camouflaged object detection,” *ArXiv*, vol. abs/2207.00794, 2022.  
[10] Ashish Vaswani, Noam M. Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin, “Attention is all you need,” in *Neural Information Processing Systems*, 2017.  
[11] Yanguang Sun, Hanyu Xuan, Jian Yang, and Lei Luo, “Glconet: Learning multisource perception representation for camouflaged object detection,” *IEEE Transactions on Neural Networks and Learning Systems*, pp. 1–14, 2024.  
[12] Zhou Huang, Hang Dai, Tian-Zhu Xiang, Shuo Wang, Huaixin Chen, Jie Qin, and Huan Xiong, “Feature shrinkage pyramid for camouflaged object detection with transformers,” *2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 5557–5566, 2023.  
[13] Runmin Cong, Mengyao Sun, Sanyi Zhang, Xiaofei Zhou, Wei Zhang, and Yao Zhao, “Frequency perception network for camouflaged object detection,” *Proceedings of the 31st ACM International Conference on Multimedia*, 2023.  
[14] Yanguang Sun, Chunyan Xu, Jian Yang, Hanyu Xuan, and Lei Luo, “Frequency-spatial entanglement learning for camouflaged object detection,” *ArXiv*, vol. abs/2409.01686, 2024.  
[15] Albert Gu and Tri Dao, “Mamba: Linear-time sequence modeling with selective state spaces,” *ArXiv*, vol. abs/2312.00752, 2023.  
[16] Yue Liu, Yunjie Tian, Yuzhong Zhao, Hongtian Yu, Lingxi Xie, Yaowei Wang, Qixiang Ye, and Yunfan Liu, “Vmamba: Visual state space model,” *ArXiv*, vol. abs/2401.10166, 2024.  
[17] Dongchen Han, Ziyi Wang, Zhuofan Xia, Yizeng Han, Yifan Pu, Chunjiang Ge, Jun Song, Shiji Song, Bo Zheng, and Gao Huang, “Demystify mamba in vision: A linear attention perspective,” in *NeurIPS*, 2024.  
[18] Ali Hatamizadeh and Jan Kautz, “Mambavision: A hybrid mamba-transformer vision backbone,” *arXiv preprint arXiv:2407.08083*, 2024.  
[19] Xiaoqi Zhao, Lihe Zhang, Youwei Pang, Huchuan Lu, and Lei Zhang, “A single stream network for robust and real-time rgb-d salient object detection,” in *European Conference on Computer Vision*, 2020.  
[20] Haiyang Mei, Ge-Peng Ji, Ziqi Wei, Xin Yang, Xiaopeng Wei, and Deng-Ping Fan, “Camouflaged object segmentation with distraction mining,” *2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 8768–8777, 2021.  
[21] Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and Franccois Fleuret, “Transformers are rnns: Fast autoregressive transformers with linear attention,” in *International Conference on Machine Learning*, 2020.  
[22] Aixuan Li, Jing Zhang, Yun-Qiu Lv, Bowen Liu, Tong Zhang, and Yuchao Dai, “Uncertainty-aware joint salient object and camouflaged object detection,” *2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 10066–10076, 2021.  
[23] F. Yang, Qiang Zhai, Xin Li, Rui Huang, Ao Luo, Hong Cheng, and Deng-Ping Fan, “Uncertainty-guided transformer reasoning for camouflaged object detection,” *2021 IEEE/CVF International Conference on Computer Vision (ICCV)*, pp. 4126–4135, 2021.  
[24] Chenxi Xie, Changqun Xia, Tianshu Yu, and Jia Li, “Frequency representation integration for camouflaged object detection,” *Proceedings of the 31st ACM International Conference on Multimedia*, 2023.  
[25] Ziyang Luo, Nian Liu, Wangbo Zhao, Xuguang Yang, Dingwen Zhang, Deng-Ping Fan, Fahad Khan, and Junwei Han, “Vscode: General visual salient and camouflaged object detection with 2d prompt learning,” *2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 17169–17180, 2023.  
[26] Xiaoyi Dong, Jianmin Bao, Dongdong Chen, Weiming Zhang, Nenghai Yu, Lu Yuan, Dong Chen, and Baining Guo, “Cswin transformer: A general vision transformer backbone with cross-shaped windows,” *2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 12114–12124, 2021.  
[27] Jianlin Su, Yu Lu, Shengfeng Pan, Bo Wen, and Yunfeng Liu, “Roformer: Enhanced transformer with rotary position embedding,” *ArXiv*, vol. abs/2104.09864, 2021.  
[28] Xiangxiang Chu, Zhi Tian, Bo Zhang, Xinlong Wang, and Chunhua Shen, “Conditional positional encodings for vision transformers,” in *International Conference on Learning Representations*, 2021.  
[29] Syed Waqas Zamir, Aditya Arora, Salman Hameed Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang, “Restormer: Efficient transformer for high-resolution image restoration,” *2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 5718–5729, 2021.  
[30] Md.Atiqur Rahman and Yang Wang, “Optimizing intersection-over-union in deep neural networks for image segmentation,” in *International Symposium on Visual Computing*, 2016.