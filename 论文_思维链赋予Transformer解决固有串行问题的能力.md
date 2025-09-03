# 思维链赋予Transformer解决固有串行问题的能力

**作者**：Zhiyuan Li¹'²，Hong Liu¹，Denny Zhou³，Tengyu Ma¹

¹斯坦福大学，²芝加哥丰田技术学院，³谷歌

## 摘要

指导模型生成一系列中间步骤，即思维链（CoT），是提高大语言模型（LLM）在算术和符号推理任务准确性的高效方法。然而，CoT背后的机制仍不清楚。本研究通过表达能力的角度为仅解码器的transformer理解CoT的力量提供理论见解。从概念上讲，CoT赋予模型执行固有串行计算的能力，这在transformer中原本是缺失的，特别是当深度较低时。给定输入长度n，先前的研究表明，具有有限精度poly(n)嵌入大小的常深度transformer在没有CoT的情况下只能解决TC⁰中的问题。我们首先展示了常深度transformer在常位精度下的更严格的表达能力上界，只能解决AC⁰中的问题，这是TC⁰的真子集。然而，通过T步CoT，使用常位精度和$$O(\log n)$$嵌入大小的常深度transformer可以解决任何由大小为T的布尔电路可解决的问题。

在经验上，启用CoT极大地提高了对并行计算困难任务的准确性，包括排列群的复合、迭代平方和电路值问题，特别是对于低深度transformer。

{

#### 核心发现

**问题背景：** 我们知道让AI"展示思考过程"（CoT）能让它算数学题更准确，但不知道为什么这么有效。

**研究发现：**

#### 1. Transformer的天生局限

- **普通Transformer**：就像一个"并行处理器"，所有计算同时进行
- **问题**：有些任务必须"一步一步来"，比如长除法，你不能跳过中间步骤
- **结果**：普通Transformer在这类任务上表现很差

#### 2. CoT的神奇作用

当你让AI"说出思考步骤"时：

- **变成了串行处理**：AI被迫一步步思考
- **突破了限制**：原来解决不了的问题现在能解决了
- **就像给计算器加了"草稿纸"**

#### 3. 理论证明

**没有CoT时：**

- 只能解决很简单的问题（AC⁰类）
- 就像只能做基本的加减法

**有CoT时：**

- 能解决复杂得多的问题
- 步骤越多，能解决的问题越难
- 就像有了草稿纸，能做复杂的长除法

}

## 1 引言

大语言模型（LLM）在复杂推理任务中表现出卓越的能力，如数学问题解决和代码生成，远远超越了标准监督机器学习技术。解锁这些高级推理能力的关键在于使LLM能够在最终确定最终答案之前生成中间步骤，即思维链（CoT）。这可以通过各种方法实现，包括使用丰富中间步骤的示例训练或指导调整模型，或通过少样本CoT提示。

一个自然的解释是中间步骤提供了关于任务和有效解决方法的额外信息，以便模型可以模仿。然而，有趣的是，生成思维步骤的功效扩展到零样本CoT提示，其中LLM仅通过"让我们逐步思考"的提示指导，甚至使用少样本示例中的错误推理步骤。这些观察表明，CoT提示的形式与其内容同样重要（如果不是更重要的话），因为仅仅指导LLM生成中间步骤就有帮助。

本文旨在研究为什么CoT的形式能够改善LLM的推理能力。我们的假设是CoT允许执行更多的串行计算，而普通的transformer在没有CoT的情况下无法做到这一点。我们通过有无CoT的表达能力角度制定和分析这一假设。我们采用电路复杂性的语言来讨论transformer的能力。先前的工作已经表明，标准的仅解码器transformer（直接输出答案）是高效的并行计算机，只能表达在阈值电路的$$O(1)$$并行运行时间内可计算的函数，TC⁰，这是一种允许高效并行计算AND、OR、NOT和MAJORITY函数的多输入计算模型。

我们首先为常精度transformer的表达能力展示了更严格的上界（定理3.1）——它只能表达TC⁰的真子集类AC⁰，其中不允许MAJORITY门。我们的上界也更现实，因为它处理浮点数的舍入问题或迭代加法，而大多数先前的结果本质上只适用于定点数加法。

然后我们证明配备CoT的transformer——允许transformer在回答问题之前自回归地生成一系列中间标记——可以解决本质上需要串行计算的复杂问题（假设复杂性理论中的众所周知的猜想）。直觉上，没有CoT时，transformer进行的串行计算数量受深度限制（在本工作中被视为固定常数），而通过T个中间步骤，可能的串行计算数量提升到T。注意T可以随着序列长度的增加而轻易增加，而深度是依赖于架构的固定数字。

具体地，我们在定理3.3中证明，具有T个中间步骤和嵌入维度为序列长度对数的常精度transformer可以表达任何由大小为T的电路可计算的函数。取T为序列长度的多项式，结果表明具有多项式多中间步骤的transformer能够计算P/poly中的所有电路，这是P的超类。定理3.3还暗示具有线性多中间步骤的transformer可以计算所有正则语言，包括不可解群的复合，如五元素上的排列群S₅，它不属于AC⁰且被广泛猜想超出TC⁰。因此，多项式多CoT步骤使有界深度和精度的transformer严格更强大。

我们在定义3.4中正式定义了transformer可以用一定数量CoT步骤解决的问题类，并在图1中总结我们的理论结果。有趣的是，我们还证明对数多CoT步骤不允许transformer计算超出AC⁰的函数（定理3.1）。

为了证实我们的理论分析，我们经验性地评估transformer在解决四个核心问题上的能力：模算术、排列复合、迭代平方和电路值问题。我们学习transformer使用大量合成数据解决这些任务，有无CoT，或有额外提示但无CoT。模算术属于TC⁰，意味着它可以轻易地并行解决。Liu等人(2022a)表明它可由具有对数精度的常深度transformer解决，确实在经验上深度1对奇偶问题（模2加法）就足够了。其他三个任务都被猜想需要本质上的串行计算。如预期，普通transformer要么需要巨大的深度来解决这些任务（因为深度是transformer串行计算数量的上界），要么根本无法解决这些任务。另一方面，只要深度超过小阈值，CoT就可以解决这些任务。这些实验证明CoT可以提供更多串行计算来解决复杂推理任务。

{

#### 关键洞察

**不是AI变聪明了，而是"工作方法"变了！**

- **形式比内容重要**：即使推理步骤是错的，光是"一步步思考"这个过程就有帮助
- **串行 vs 并行**：有些问题天生需要按顺序处理，不能并行
- **更多步骤 = 更强能力**：允许的中间步骤越多，能解决的问题越复杂

这就解释了为什么简单加一句提示词就能让AI表现提升这么多——本质上是改变了AI的"计算模式"！

}

## 2 记号和预备知识

我们使用$$\mathbb{N}$$和$$\mathbb{R}$$分别表示自然数集和实数集。对任何$$n \in \mathbb{N}^+$$，我们定义$$[n] \triangleq {1, 2, \ldots, n}$$。我们定义$$\text{relu}(x) \triangleq \max(x, 0)$$。对向量x，我们使用$$x_{a:b}$$表示包含x从位置a到位置b坐标的向量。对矩阵M，我们定义$$M_{a_1:b_1,a_2:b_2}$$表示通过选择从$$a_1$$到$$b_1$$行、从$$a_2$$到$$b_2$$列的子矩阵。

给定两个非负函数f, g，我们说$$f(n) = O(g(n))$$（相应地$$f(n) = \Omega(g(n))$$）当且仅当存在$$C > 0$$，使得对所有$$n \geq 0$$，$$f(n) \leq Cg(n)$$（相应地$$f(n) \geq Cg(n)$$）。我们使用$$\text{poly}(n) \triangleq {T : \mathbb{N} \to \mathbb{N} | \exists k > 0, T(n) = O(n^k)}$$表示最多多项式增长率的函数集。

我们使用$$\phi(x) = \sum_{i=1}^{|x|} 2^{|x|-i}x_i$$表示二进制字符串x表示的二进制数的值。我们使用$$\text{bin}_k(x)$$表示使用k个二进制位对自然数x的通常二进制编码，使得$$\phi(\text{bin}_k(x)) = x$$，使用$$\text{sbin}_k(x)$$表示有符号二进制编码，即$$2\text{bin}_k(x) - (1, \ldots, 1)$$。

对任何$$n \in \mathbb{N}^+$$，我们定义$$\text{softmax} : \mathbb{R}^n \to \mathbb{R}^n$$为$$(\text{softmax}(x))*i = \exp(x_i)/\sum*{i=1}^n \exp(x_i)$$，对任何$$x \in \mathbb{R}^n$$和$$i \in [n]$$。我们使用$$\odot$$表示两个向量的逐元素乘积。我们使用$$a|b$$或$$(a, b)$$表示两个向量a和b的连接。

### 2.1 仅解码器Transformer

给定词汇表V，具有参数θ和最大输入长度$$n_{\max}$$的仅解码器transformer将输入标记序列$$(x_1, \ldots, x_n) \in V^n$$映射到V上的概率分布，对所有$$n \leq n_{\max}$$，记作$$p_\theta(\cdot | x_1, \ldots, x_n)$$。我们还通过在V中最大化$$p_\theta(\cdot | x_1, \ldots, x_n)$$的标记定义函数$$\text{TF}*\theta(x)$$，即$$\text{TF}*\theta(x_1, \ldots, x_n) \triangleq \arg\max_{y \in V} p_\theta(y | x_1, \ldots, x_n)$$。

**下一标记生成器**：给定词汇表V，具有参数θ和最大输入长度$$n_{\max}$$的下一标记生成器是从$$\bigcup_{n=1}^{n_{\max}} V^n$$到V的映射。我们在本工作中感兴趣的主要下一标记生成器是仅解码器transformer，$$\text{TF}_\theta(x_1, \ldots, x_n)$$，其中$$x_i \in V$$对所有$$i \in [n]$$。

我们还递归定义$$\text{TF}*\theta^i(x_1, \ldots, x_n) \triangleq \text{TF}*\theta^{i-1}(x_1, \ldots, x_n, \text{TF}*\theta(x_1, \ldots, x_n))$$，对每个正整数i和n满足$$i + n \leq n*{\max} - 1$$，基础情况为$$\text{TF}*\theta^1(x_1, \ldots, x_n) \triangleq \text{TF}*\theta(x_1, \ldots, x_n)$$。换句话说，对所有$$0 \leq i \leq n_{\max} - n - 1$$，具有i步CoT的输出是$$x_{n+i+1} = \text{TF}*\theta^{i+1}(x_1, \ldots, x_n) = \text{TF}*\theta(x_1, \ldots, x_n, x_{n+1}, \ldots, x_{n+i})$$。

**Transformer架构概述**：我们在本文中考虑的仅解码器transformer模型与GPT风格架构非常相似，由四个部分组成：标记嵌入层（TE）、位置编码层（PE）、输出线性层（OUTPUT）和L个相同层的堆栈作为"解码器"，其中L也称为模型的深度。每个解码器层有两个子层：多头自注意力层（ATTN）和位置相关的全连接前馈网络（FF）。

**自注意力机制**：给定注意力参数$$\theta_{\text{ATTN}} = (W_Q, W_K, W_V, W_O) \in \mathbb{R}^{d \times d} \times \mathbb{R}^{d \times d} \times \mathbb{R}^{d \times d} \times \mathbb{R}^{d \times d}$$，我们在算法3中定义仅解码器transformer的带掩码注意力层。

**算法1：因果自注意力ATTN**

- 输入：参数$$\theta_{\text{ATTN}} = (W_Q, W_K, W_V, W_O)$$，输入嵌入$$h = (h_1, \ldots, h_n) \in \mathbb{R}^{nd}$$
- 输出：输出嵌入$$h' = (h'*1, \ldots, h'\*n) \triangleq \text{ATTN}\*{\theta*{\text{ATTN}}}(h_1, \ldots, h_n)$$

1. $$q_i \triangleq W_Q h_i, k_i \triangleq W_K h_i, v_i \triangleq W_V h_i, \forall i \in [n]$$
2. $$s_i \triangleq \text{softmax}(\langle q_i, k_1 \rangle, \ldots, \langle q_i, k_i \rangle) | (0, \ldots, 0)$$
3. $$h'*i \triangleq W_O \sum*{j=1}^n (s_i)_j v_j$$

**前馈网络**：给定全连接前馈网络层的参数$$\theta_{\text{FF}} = (W_1, b_1, W_2, b_2) \in \mathbb{R}^{d \times d} \times \mathbb{R}^d \times \mathbb{R}^{d \times d} \times \mathbb{R}^d$$，我们定义全连接前馈层$$\text{FF}*{\theta*{\text{FF}}} : \mathbb{R}^d \to \mathbb{R}^d$$为$$\text{FF}*{\theta*{\text{FF}}}(h) \triangleq W_2 \text{relu}(W_1 h + b_1) + b_2$$。

**算法2：仅解码器Transformer，$$\text{TF}\*\theta$$和$$p\*\theta$$**

输入：Transformer参数$$\theta = (\theta_{\text{PE}}, \theta_{\text{TE}}, \theta_{\text{OUTPUT}}, {\theta_{\text{ATTN}}^{(l)}, \theta_{\text{FF}}^{(l)}}_{l=0}^{L-1})$$和输入标记$$x = (x_1, \ldots, x_n) \in V^n$$

输出：对所有$$i \in [n]$$的输出分布$$p_\theta(\cdot | x_1, \ldots, x_i)$$和输出标记$$\text{TF}_\theta(x)$$

1. $$h_i^{(0)} \leftarrow \theta_{\text{TE}}(x_i) + \theta_{\text{PE}}(i), \forall i \in [n]$$
2. **for** $$l = 0, \ldots, L-1$$ **do**
3. $$(h_1^{(l+0.5)}, \ldots, h_n^{(l+0.5)}) \leftarrow (h_1^{(l)}, \ldots, h_n^{(l)}) + \text{ATTN}*{\theta*{\text{ATTN}}^{(l)}}(h_1^{(l)}, \ldots, h_n^{(l)})$$
4. $$h_i^{(l+1)} \leftarrow h_i^{(l+0.5)} + \text{FF}*{\theta*{\text{FF}}^{(l)}}(h_i^{(l+0.5)}), \forall i \in [n]$$
5. **end for**
6. $$p_\theta(\cdot | x_1, \ldots, x_i) \leftarrow \text{OUTPUT}*{\theta*{\text{OUTPUT}}}(h_i^{(L)}), \forall i \in [n]$$
7. $$\text{TF}*\theta(x) \leftarrow \arg\max_y p*\theta(y | x_1, \ldots, x_n)$$

### 2.2 电路复杂性

**问题**：在本文中，我们考虑以下问题概念：给定输入标记序列，输出一个标记作为答案。数学上，给定词汇表V，我们称映射$$L : \bigcup_{k \in \mathbb{N}^+} V^k \to V$$为问题。如果正确答案总是0或1，我们称L为决策问题。在电路复杂性中，这样的L也被称为语言。

虽然电路复杂性的标准定义只处理二进制字符串，但给定任何有限词汇表V，我们总是可以用二进制表示替换V中的每个标记，输入长度只会增加一个常数因子。因此，我们可以自然地将现有复杂性类扩展到任意有限词汇表。

**P**：类P包含所有可由确定性图灵机在多项式时间内解决的问题。

**布尔电路**：n个变量上的布尔电路是有向无环图，其中节点是AND、OR或NOT门。入度为0的门是输入，被分配n个布尔变量之一。给定输入，电路根据传入门的值计算每个非输入门的值，并在输出门输出一个数字。

**SIZE[T(n)]**：给定任何函数T，SIZE[T(n)]表示当输入长度为n时可由具有$$O(T(n))$$个门的布尔电路解决的问题类。形式上，问题L在SIZE[T(n)]中当且仅当存在电路序列$${C_n}$$，使得每个电路$$C_n$$有n个输入和1个输出，每个电路$$C_n$$的大小至多为$$O(T(n))$$，且对所有字符串x，x在L中当且仅当$$C_{|x|}(x) = 1$$。

**P/poly**：我们定义类P/poly为可由多项式大小电路族解决的问题集，即$$\text{P/poly} \triangleq \bigcup_{k \in \mathbb{N}^+} \text{SIZE}[n^k]$$。由于任何时间界为$$T(n)$$的图灵机可由大小为$$T(n) \log T(n)$$的电路模拟，我们知道$$\text{P} \subseteq \text{P/poly}$$。

**NC、AC和TC**：类NC包含所有可在小并行运行时间——输入长度的多项式对数——和多项式数量处理器内解决的问题。形式上，对正整数k，问题L在$$\text{NC}^k$$中当且仅当存在多项式$$p(n)$$和电路族$${C_n}$$，使得每个电路$$C_n$$有n个输入和1个输出，门的扇入至多为2，每个电路$$C_n$$的大小至多为$$p(n)$$，每个电路$$C_n$$的深度为$$O((\log n)^k)$$，且对所有字符串x，x在L中当且仅当$$C_{|x|}(x) = 1$$。最后我们定义$$\text{NC} = \bigcup_{k \in \mathbb{N}} \text{NC}^k$$。

类$$\text{AC}^k$$对每个$$k \in \mathbb{N}^+$$的定义几乎与$$\text{NC}^k$$相同，除了$$\text{AC}^k$$中的AND和OR门允许无界扇入。类$$\text{TC}^k$$与$$\text{AC}^k$$相比允许更强大的门类型——MAJORITY。MAJORITY门可以有无界扇入，定义为$$\text{MAJORITY}(x_1, \ldots, x_n) = \lfloor \frac{1}{2} + \frac{(\sum_{i=1}^n x_i) - 1/2}{n} \rfloor$$。

对所有自然数i，有$$\text{NC}^i \subseteq \text{AC}^i \subseteq \text{TC}^i \subseteq \text{NC}^{i+1}$$。因此$$\text{NC} = \text{AC} = \text{TC}$$，都代表可在多项式对数时间内用多项式并行处理器解决的问题类。

{

这部分是论文的**"数学工具箱"**，让我用通俗的语言解释：

#### 2.1 Transformer是什么

#### 基本概念

想象Transformer是个**"智能翻译机"**：

- **输入**：一串文字（比如"今天天气很好"）
- **输出**：下一个最可能的词（比如"呢"）

#### 工作流程

就像工厂的**流水线**：

1. **标记嵌入**：把每个字变成数字向量（就像给每个字打标签）

2. **位置编码**：告诉AI每个字在第几位（就像给字编号）

3. 多层处理

   ：

   - **注意力层**：让AI看看前面的字，理解上下文
   - **前馈网络**：基于理解做计算

4. **输出层**：预测下一个字

#### CoT模式

- **普通模式**：看完输入直接给答案
- **CoT模式**：先生成中间步骤，再基于这些步骤继续生成

#### 2.2 电路复杂性（问题难度分类）

这就像给**数学题按难度分类**：

#### 基本分类

**P类**：

- 普通计算机能在"合理时间"内解决
- 就像小学数学题，时间够就能算出来

**AC⁰类**：

- 只能用**简单运算**（加、减、与、或）
- 就像只会四则运算的计算器
- **不能做复杂判断**（比如判断奇偶需要看所有位）

**TC⁰类**：

- 比AC⁰强一点，可以做**投票判断**
- 就像会"少数服从多数"的计算器
- 能判断"大部分是1还是0"

**P/poly类**：

- 非常强大，几乎能解决所有"实际问题"
- 就像超级计算机

#### 关键对比

```
AC⁰ ⊂ TC⁰ ⊂ P/poly
(最弱)   (中等)   (很强)
```

**论文的核心发现**：

- **没有CoT的Transformer**：只有AC⁰的能力（很弱）
- **有CoT的Transformer**：能达到P/poly的能力（很强）

}

## 3 具有思维链(CoT)的Transformer表达能力理论

### 3.1 有限精度建模

在实践中，transformer的训练和推理通常使用16位或32位浮点数完成。因此在本文中，我们主要关注常精度transformer的计算模型，其中每个算术运算的输出被舍入到遵循IEEE 754标准的固定数字位数可表示的最近浮点数（定义3.2），从而避免了先前工作中的不现实的无限精度假设。

下面我们给出浮点数和舍入操作的正式定义。回忆$$\phi(a) = \sum_{i=1}^k 2^{k-i}a_i$$表示对任何$$k \in \mathbb{N}^+$$，$$a \in {0,1}^k$$表示的二进制数的值。

**定义3.1（浮点表示）**：设e为指数位数，s为尾数位数。$$(e + 2s + 1)$$位二进制字符串$$a = (a_1, a_2, \ldots, a_{e+2s+1}) \in {0,1}^{e+2s+1}$$是数字$$\phi_{e,s}(a) \triangleq \text{sign}(a) \cdot 2^{\text{exponent}(a)} \cdot \text{significand}(a)$$的具有e位指数和2s精度的浮点二进制表示，其中符号为$$\text{sign}(a) \triangleq 2a_1 - 1$$，尾数为$$\text{significand}(a) \triangleq 2^{-s}\phi(a_{2:2s+1})$$，指数为$$\text{exponent}(a) \triangleq \phi(a_{2s+2:2s+e+1}) - 2^{\max(0,e-1)}$$。

我们进一步使用$$F_{e,s}$$表示使用e位指数和2s位精度（尾数）可表示的所有浮点数，即$$F_{e,s} \triangleq {S \cdot 2^{-s+E} | -2^{2s} + 1 \leq S \leq 2^{2s} - 1, -2^{\max(0,e-1)} \leq E \leq 2^e - 1 - 2^{\max(0,e-1)}, E, S \in \mathbb{N}}$$。我们定义$$B_{e,s} \triangleq \max F_{e,s}$$。

**定义3.2（正确舍入）**：对任何$$x \in \mathbb{R}$$和任何包含0的$$\mathbb{R}$$的闭子集F，我们定义正确舍入$$\text{round}(x, F)$$为F中最接近x的数。我们通过选择绝对值较小的数来打破平局。

特别地，我们用$$\text{round}*{e,s}(\cdot) \triangleq \text{round}(\cdot, F*{e,s})$$表示具有e位指数、2s位精度的舍入操作，为了方便也记作$$[\cdot]^{e,s}$$。我们通过坐标级舍入将round和$$\text{round}_{e,s}$$的定义扩展到向量输入。

我们的浮点数概念通过删除$$\infty$$和$$-\infty$$简化了IEEE 754浮点算术标准。当发生溢出时，我们总是将输出舍入到$$F_{e,s}$$中（负）最大可表示数。对于包括加法、减法、乘法和除法在内的一元函数如exp(·)和二元函数，我们简单地通过舍入其输出来定义它们的舍入版本。每当发生除零时，我们将其视为模型输出错误结果。

接下来，我们通过将其分解为固定顺序的舍入二进制加法链来定义对两个以上数字的有限精度求和。

**定义3.3（迭代舍入求和）**：对任何$$s, n \in \mathbb{N}^+$$和向量$$x \in \mathbb{R}^n$$，我们定义到e位指数和2s位精度的迭代舍入求和为$$\text{sum}*{e,s} : \bigcup*{n \in \mathbb{N}^+} (F_{e,s})^n \to F_{e,s}$$，其中对任何$$n \in \mathbb{N}^+$$和$$x \in \mathbb{R}^n$$：

$$\text{sum}*{e,s}(x) \triangleq \left[ \left[ \left[ [x_1 + x_2]^{e,s} + x_3 \right]^{e,s} + \cdots x*{n-1} \right]^{e,s} + x_n \right]^{e,s}$$

我们进一步定义以下操作：

- 有限精度内积：$$\langle x, y \rangle^{e,s} \triangleq \text{sum}_{e,s}(x \odot y)$$
- 有限精度矩阵乘积：$$(A \times_{e,s} B)*{i,j} \triangleq \langle (A*{i,:})^T, B_{:,j} \rangle^{e,s}$$
- 有限精度softmax：$$\text{softmax}*{e,s}(x) \triangleq [[\exp(x)]^{e,s} / \text{sum}*{e,s}([\exp(x)]^{e,s})]^{e,s}$$

最后，有限精度transformer可以通过用上面列出的有限精度对应物替换所有无限精度操作来定义。（详见算法4）。我们将各个transformer层的有限精度版本的细节推迟到附录B。

### 3.2 CoT：具有CoT的常深度Transformer复杂性类

在本小节中，我们定义由具有有限精度CoT的某些仅解码器transformer可解决的所有问题组成的复杂性类。

**定义3.4（CoT）**：给定有限词汇表V和四个函数$$T(n), d(n), s(n), e(n)$$，非正式地，$$\text{CoT}[T(n), d(n), s(n), e(n)]$$是由具有常深度、s(n)位精度、e(n)位指数、嵌入大小d(n)和T(n)步CoT的transformer可解决的问题族。

形式上，我们说问题$L : \bigcup_{n \in \mathbb{N}^+} V^n \to {0,1}$在$\text{CoT}[T(n), d(n), s(n), e(n)]$中当且仅当存在整数L和三个函数$T'(n) = O(T(n))$，$d'(n) = O(d(n))$，$s'(n) = O(s(n))$，$e'(n) = O(e(n))$，使得对每个正整数n，存在一个L层仅解码器transformer，记作$\text{TF}_{\theta_n}$，具有嵌入大小$d'(n)$、$2s'(n)$位精度和$e'(n)$位指数，可以使用$T'(n)$步思维链对任何输入$x \in V^n$输出$L(x)$。数学上，这意味着：

$\text{TF}_{\theta_n}^{1+T'(n)}(x) = L(x), \forall x \in V^n \quad (1)$

我们还将CoT的定义扩展到函数类而不是单个函数。例如，$\text{CoT}[T(n), \text{poly}(n), s(n), e(n)] \triangleq \bigcup_{k \in \mathbb{N}^+} \text{CoT}[T(n), n^k, s(n), e(n)]$。

**定义3.5（T）**：我们定义$\mathbf{T}[d(n), s(n), e(n)] \triangleq \text{CoT}[0, d(n), s(n), e(n)]$为常深度、常精度仅解码器transformer可以用$O(s(n))$位精度、$O(e(n))$位指数、$O(d(n))$嵌入大小且不使用CoT（或只使用0步CoT）解决的问题。

根据定义，$\text{CoT}[T(n), d(n), s(n), e(n)]$在所有$T(n), d(n), s(n), e(n)$中都是单调的，例如，如果对所有$n \in \mathbb{N}$有$T'(n) \leq T(n)$，则$\text{CoT}[T'(n), d(n), s(n), e(n)] \subseteq \text{CoT}[T(n), d(n), s(n), e(n)]$。特别地，我们有$\mathbf{T}[d(n), s(n), e(n)] \triangleq \text{CoT}[0, d(n), s(n), e(n)] \subseteq \text{CoT}[T(n), d(n), s(n), e(n)]$。

注意上面定义的复杂性类CoT是非一致的，即它允许对每个输入大小使用不同的程序。这与之前的工作形成对比，之前的工作专注于一致的transformer类。请参考附录G的讨论。

### 3.3 Transformer表达能力的更严格上界

现有工作已经表明，常深度、多项式宽度和对数精度的transformer可以在小并行时间内模拟，即使用TC⁰电路。这些结果建立在n位二进制数的乘法和除法，以及对n个不同n位二进制整数的迭代加法都在TC⁰中这一事实上。

然而，对于使用浮点数运算的transformer，这样的TC⁰表达能力上界可能是不现实的。先前工作隐含地假设当添加多于一个浮点数时，算法首先使用任意更多精度计算精确答案而不舍入，仅在最后执行舍入。然而，在实践中，舍入发生在每两个数字之间的加法之后，对于这样的TC⁰上界是否仍然成立还是开放的。立即舍入使浮点数的迭代加法不再满足结合律，例如，$\text{round}(a + \text{round}(b + c)) \neq \text{round}(\text{round}(a + b) + c)$。整数加法的结合律在n个不同n位二进制整数的迭代加法在TC⁰中这一事实中起着关键作用。

在本节中，我们为在每个算术运算步骤后舍入直接结果的transformer提出两个新的表达能力上界。首先，我们为具有常位精度和指数的常深度transformer展示了严格强于TC⁰的上界，即AC⁰（定理3.1）。这表明当输入长度足够长时，常精度transformer最终无法计数，即使在模运算的意义上也是如此。例如，众所周知，没有AC⁰电路可以决定二进制字符串的奇偶性。

**定理3.1**：$\mathbf{T}[\text{poly}(n), 1, 1] \subseteq \text{CoT}[\log n, \text{poly}(n), 1, 1] \subseteq \text{AC}^0$

我们的第二个结果定理3.2显示，当指数位数为0（即定点数）时，即使使用定义3.2中定义的正确舍入，常深度、对数精度transformer表达能力的TC⁰上界仍然成立。

**定理3.2**：$\mathbf{T}[\text{poly}(n), \log(n), 0] \subseteq \text{CoT}[\log n, \text{poly}(n), \log(n), 0] \subseteq \text{TC}^0$

我们注意到transformer的单次前向传播可以被AC⁰电路模拟这一事实立即暗示具有$O(\log n)$步CoT的transformer输出也可以被AC⁰模拟。这是因为一般来说，可以将具有T步CoT的transformer输出视为$2^T$个不相交子电路的OR，其中每个子电路枚举所有T个CoT标记的可能值，并在所有中间标记值一致的分支中输出标记的值。这种枚举可以并行完成，因此只需要常数深度。当$T = O(\log n)$时，这只会导致电路大小的poly(n)倍爆炸，因此仍在AC⁰中。同样的论证对TC⁰也成立。

上述两个结果的主要技术困难是证明$\text{sum}*{e,s} : (F*{e,s})^n \to F_{e,s}$在e、s都是常数（相应地e = 0，s = O(\log(n))）时具有AC⁰（相应地TC⁰）电路。我们将具有舍入的$F_{e,s}$上的迭代加法视为状态空间和词汇表都是$F_{e,s}$的自动机。第一个结果是由于自动机的经典Krohn-Rhodes分解定理的新颖应用（定理C.2），其中我们使用舍入加法的性质：对所有$x, x' \in F_{e,s}, y \in F_{e,s}$，$x \geq x' \Rightarrow [x + y]^{e,s} \geq [x' + y]^{e,s}$。我们在定义D.2中将此性质形式化为有序自动机，并证明所有有序自动机都是无计数器的（定理D.3），因此可以被AC⁰电路模拟。

定理3.1的证明技术不能推广到定理3.2，因为之前构造的AC⁰电路的深度取决于自动机状态的数量，因此不是常数。我们对定理3.2的证明受到Liu等人(2022a)中名为'GridWorld'的自动机算法1的启发。

然而，对于具有对数位指数的常深度、对数精度transformer $\mathbf{T}[\text{poly}(n), \log(n), \log(n)]$甚至常位指数$\mathbf{T}[\text{poly}(n), \log(n), 1]$是否有TC⁰电路仍然是开放的。

### 3.4 CoT使Transformer更具表达力

现在我们准备提出我们的主要理论结果（定理3.3），它刻画了具有CoT和$O(\log(n))$嵌入大小的常深度、常精度transformer的表达能力。$\log(n)$嵌入大小是必要的，以确保n个输入的位置嵌入是不同的。

所有transformer表达能力的下界（有无CoT）都针对定点数证明，即不使用任何指数位。允许指数位只会使transformer更具表达力。为了方便，我们定义$\text{CoT}[T(n), d(n), s(n)] \triangleq \text{CoT}[T(n), d(n), s(n), 0]$。

**定理3.3**：对任何多项式函数$T : \mathbb{N}^+ \to \mathbb{N}^+$，$\text{SIZE}[T(n)] \subseteq \text{CoT}[T(n), \log n, 1]$。特别地，$\text{P/poly} = \text{CoT}[\text{poly}(n), \log n, 1]$。

与定理3.1和3.2相比，定理3.3显示在标准难度假设$\text{TC}^0 \subsetneq \text{P/poly}$下，允许多项式步CoT严格使常深度、常精度、仅解码器transformer更具表达力，使对数精度transformer更具表达力。

**定理3.3的证明思路**：高层证明思想是我们使用CoT中的每一步来模拟目标电路中的一个门操作，并将门输出写作下一个输入。为了做到这一点，我们使用一个位置编码为每个门存储信息，包含四个部分：当前门id、下一个门类型{AND, OR, NOT, TRUE, FALSE}，以及下一个门的两个输入门id。由于总共有poly(n)个门，$d(n) = \Theta(\log n)$嵌入大小足以存储上述信息。这里的CoT被构造为每个门按id递增顺序的值。因此，在每一步中，我们可以使用注意力拉取两个输入门的值（已经计算的或者是输入），并使用前馈网络计算当前门的值。证明思想在图2中有所说明。

从证明思路可以看出，CoT模拟任何深度电路的关键步骤是将输出标记写回到下一个输入位置。这个动作将电路中中间输出的"深度"重置为0。我们的理论解释了Wei等人(2022)中的消融实验，当模型被提示只输出等于解决问题所需标记数量的点序列（...）时，性能不比直接输出答案好。

因为每个正则语言都可以被有限状态自动机识别（定义C.1），而有限状态自动机显然可以被线性大小电路模拟，作为定理3.3的直接推论，以下成立：

**推论3.4**：每个正则语言都属于$\text{CoT}[n, \log n, 1]$。

下面我们给出一个具体的正则语言，在标准难度假设$\text{TC}^0 \subsetneq \text{NC}^1$下，常深度、多项式嵌入大小transformer只有通过CoT才能解决，即五元素上排列群S₅的字问题，在定理3.5中。

**定义3.6（群G的字问题）**：给定G中的n个元素$(g_1, \ldots, g_n)$，我们使用$L_G$表示$g_1 \circ g_2 \circ \cdots \circ g_n$是否等于G的单位元的决策问题。

为了方便，在本文中，我们将$L_G$的定义域扩展到由二进制字符串编码的群序列。定理3.5的证明是定理3.2、3.3和3.6的直接结果。

**定理3.5**：假设$\text{TC}^0 \subsetneq \text{NC}^1$，S₅的字问题$L_{S_5}$在$\text{CoT}[n, \log n, 1]$中但不在$\mathbf{T}[\text{poly}(n), \log n]$中。

**定理3.6（Barrington (1986)）**：S₅的字问题在AC⁰归约下是$\text{NC}^1$完全的。即，对$\text{NC}^1$中的任何决策问题L，存在AC⁰电路族${C_n}*{n=1}^{\infty}$（常深度，poly(n)扇出），使得对任何$n \in \mathbb{N}^+$和$x \in {0,1}^n$，$L(x) = L*{S_5}(C_n(x))$。

**定理3.5的证明**：首先$L_{S_5}$是正则语言，因此通过推论3.4属于$\text{CoT}[n, \log n, 1]$。由于$L_{S_5}$通过定理3.6是$\text{NC}^1$完全的，假设$\text{TC}^0 \subsetneq \text{NC}^1$，$L_{S_5}$不属于TC⁰。通过应用定理3.2完成证明，该定理表明$\mathbf{T}[\text{poly}(n), \log(n)] \subseteq \text{TC}^0$。

**poly(n)嵌入大小的结果**：到目前为止，我们一直专注于具有$O(\log n)$嵌入大小的transformer的表达能力，那么transformer是否也能从更大的嵌入大小（比如poly(n)）中受益是很自然的问题？我们的定理3.7通过显示对数精度（相应地常精度）常深度多项式嵌入大小仅解码器transformer具有T(n)步CoT可以模拟任何T(n)大小的电路与一些具有poly(n)输入的TC⁰（相应地AC⁰）预言门来积极回答这个问题。

形式上，给定决策问题$L : \bigcup_{n=1}^{\infty} {0,1}^n \to {0,1}$，我们使用$L_n$表示L在${0,1}^n$上的限制，这也可以视为具有n个扇入的单门。我们通过定义3.7定义可由具有一定大小门（包括预言门）的电路解决的问题。

**定义3.7（SIZE^L）**：对任何决策问题L和$T(n) \subseteq O(\text{poly}(n))$，我们定义$\text{SIZE}^L(T(n))$为决策问题$L'$的集合，使得存在$p(n) \in \text{poly}(n)$和电路${C_n}*{n=1}^{\infty}$，其中$C_n$包含至多$O(T(n))$个AND、OR、NOT和$L*{p(n)}$门。对复杂性类C，我们定义$\text{SIZE}^C(T(n)) \triangleq \bigcup_{L \in C} \text{SIZE}^L(T(n))$。

**定理3.7**：对任何$T(n) \in \text{poly}(n)$，$\text{SIZE}^{\text{TC}^0}[1+T(n)] = \text{CoT}[T(n), \text{poly}(n), \log n]$。特别地，对$T(n) = 0$，我们有$\text{TC}^0 = \text{SIZE}^{\text{TC}^0}[1] = \text{CoT}[0, \text{poly}(n), \log n] = \mathbf{T}[\text{poly}(n), \log n]$。

**定理3.8**：对任何$T(n) \in \text{poly}(n)$，$\text{SIZE}^{\text{AC}^0}[1 + T(n)] = \text{CoT}[T(n), \text{poly}(n), 1]$。特别地，对$T(n) = 0$，我们有$\text{AC}^0 = \text{SIZE}^{\text{AC}^0}[1] = \text{CoT}[0, \text{poly}(n), 1] = \mathbf{T}[\text{poly}(n), 1]$。

定理3.8显示对于$T(n) = \text{poly}(n)$步CoT，使用poly(n)嵌入大小不会比使用$\log(n)$嵌入大小提高表达能力（定理3.3），因为$\text{SIZE}^{\text{TC}^0}[\text{poly}(n)] = \text{SIZE}^{\text{AC}^0}[\text{poly}(n)] = \text{SIZE}[\text{poly}(n)]$。然而，定理3.9显示对于任何特定多项式$T(n) = n^k$步CoT，将嵌入宽度从$O(\log(n))$增加到poly(n)使transformer严格更强大。

**定理3.9**：对任何$s(n) = O(\log n)$，$\mathbf{T}[\log n, s(n)] \subsetneq \mathbf{T}[\text{poly}(n), s(n)]$，对所有$k \in \mathbb{N}$，$\text{CoT}[n^k, \log n, s(n)] \subsetneq \text{CoT}[n^k, \text{poly}(n), s(n)]$。

## 4 CoT在经验上提高低深度Transformer在固有串行问题上的表达能力

本节是对仅解码器transformer在四个不同算术问题上具有CoT的表达能力的经验研究：模算术、排列复合（S₅）、迭代平方和电路值问题。第一个问题是可并行化的，可以由具有对数精度的常深度transformer解决，而后三个在一些复杂性理论或密码学的标准难度假设下是本质上串行的。作为我们理论的预测，我们期望当CoT开启时准确性会有巨大提升。

**一般设置**：为了检查具有和不具有CoT的解码器transformer在这四类问题上的表达能力，我们在每个问题和每个不同序列长度n的在线监督设置中使用Adam从随机初始化训练transformer。在每一步，我们从分布$p_n(x)$中采样一批训练数据，其中$x = (x_1, \ldots, x_n)$是训练数据，$y = f^*(x_1, \ldots, x_n)$是标签。我们总是设置$x_n$为'='。我们考虑三种不同设置：base、cot和hint：

- **base**：优化目标简单地为$\ell_{\text{base}}(\theta) \triangleq \mathbb{E}*{x \sim p} [-\log p*\theta(f^*(x) | x)]$
- **cot**：我们手动为每个实例x设计思维链，它是V中的字符串，我们用c(x)表示。我们确保c(x)的最后一个标记总是等于答案$f^*(x)$。设$\tilde{x} \triangleq (x, c(x))$为x和c(x)的连接，m为c(x)的长度，优化目标为$\ell_{\text{cot}}(\theta) \triangleq \frac{1}{m} \mathbb{E}*{x \sim p} \sum*{i=n}^{n+m-1} [-\ln p_\theta(\tilde{x}_{i+1} | \tilde{x}_1, \ldots, \tilde{x}_i)]$
- **hint**：即使transformer在cot设置中比base设置有更好的性能，人们可能争论除了表达能力的差异外，cot设置相比base也有统计优势，因为cot提供更多标签，因此关于真实$f^*$的更多信息给模型。这激励我们设计以下损失，将思维链c(x)作为标签提供。为了简单起见，我们假设c(x)的长度等于n。形式上我们定义$\ell_{\text{hint}}(\theta) \triangleq \frac{1}{n} \mathbb{E}*{x \sim p} \sum*{i=1}^n [-\ln p_\theta(c_i(x) | x_1, \ldots, x_i)]$

**性能评估**：由于我们在每一步使用新鲜采样的合成数据训练transformer，训练准确性/损失与验证准确性/损失相同。对于base和hint设置，我们直接评估最终答案的准确性。对于cot设置，直接评估最终答案过于容易，因为它只测量transformer在给定CoT作为输入的情况下正确计算最后一步的能力。理想情况下，我们应该测量transformer在自回归生成$|c(x)|$个标记后输出的答案。但为了计算效率，我们测量transformer能够正确预测给定CoT中所有标记的概率。注意这个概率是理想度量的下界，因为存在transformer可能用错误CoT正确回答的小概率。尽管如此，即使使用这个稍微更困难的评估度量，cot设置中的transformer仍然比没有CoT时优化得更快。

由于空间限制，我们将训练细节和每个设置推迟到附录A。我们的实验结果在图3到6中呈现。

**我们的发现**：不出所料，hint设置中的准确性总是高于base设置。由于空间限制，我们将base设置的所有结果推迟到附录A。对于并行计算困难的问题，即排列复合、迭代平方和电路值问题，我们发现cot总是比hint和base更好，特别是当深度较小时，改进是巨大的。我们的实验表明开启CoT极大地提高了低深度transformer在难以并行计算的问题上的表达能力，即那些本质上串行的问题。

## 5 相关工作

尽管在经验成就方面取得了无数成功，但关于能够进行算法推理的神经网络内部工作机制的问题仍未得到解答。自注意力创建低复杂性电路的能力已被认识到，以及其形成声明性程序和图灵机的能力。此外，已经证明可以从训练的模型中提取可解释的符号计算。

Liu等人(2022a)是与我们密切相关的工作，它研究了低深度transformer对半自动机的表达能力。他们的设置对应于只使用1步CoT，我们的贡献是展示允许更多步CoT使transformer能够解决比半自动机更困难的问题，特别是那些本质上串行的问题，如电路值问题，它是P完全的。

**常精度与对数精度**：我们注意到大多数关于transformer表达能力的先前文献都专注于对数精度设置，包括先前的工作等。正如Merrill & Sabharwal (2023a)所争论的，一个主要原因是对数精度允许transformer对其余标记使用均匀注意力。然而，LLM的最近进展表明，均匀注意力对于良好性能可能不是必需的，至少对于自然语言任务是如此。例如，最成功的开源LLM之一LLAMA2接受4096个标记的序列输入并使用BF16精度，它有1个符号位、8个指数位和7个尾数位（加上一个额外的前导位）。因此，例如，BF16不能表达$2^8 = 256$和$2^8 + 2 = 258$之间的任何浮点数，这使得LLAMA2不可能计算超过257个元素的均匀注意力。

并发工作Feng等人(2023)也通过表达能力的角度研究CoT的好处，他们显示通过CoT，transformer可以解决一些特定的P完全问题。我们的结果更强，因为我们为P/poly中的每个问题给出了简单而干净的构造。我们还注意到设置的细微差别，虽然我们主要关注具有$O(\log n)$嵌入大小的常精度transformer，但他们关注具有有界嵌入大小的$O(\log(n))$精度transformer。

## 6 结论

我们通过表达能力的角度研究CoT对仅解码器transformer的能力。我们采用电路复杂性的语言并定义新的复杂性类$\text{CoT}[T(n), d(n), s(n), e(n)]$，它对应于由具有$O(T(n))$步CoT、$O(d(n))$嵌入大小和具有$O(e(n))$位指数和$O(s(n))$位尾数的浮点数的常深度、常精度仅解码器transformer可解决的问题类。我们的理论表明增加CoT的长度可以极大地使transformer更具表达力。我们还在四个算术问题上经验验证了我们的理论。我们发现对于那三个本质上串行的问题，transformer只能通过使用CoT来表达真实函数。

## 致谢

作者感谢NSF IIS 2045685的支持。作者还感谢Wei Zhan和Lĳie Chen在电路复杂性方面提供参考文献和各种启发性讨论，感谢Cyril Zhang和Bingbin Liu就Khron-Rhodes分解定理的有用讨论，以及Kaifeng Lyu的有用反馈。