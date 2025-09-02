```

```

# Communications Chemistry Article  
*Nature Portfolio 期刊*  
https://doi.org/10.1038/s42004-025-01437-x  

## 使用条件Transformer进行分子优化，通过强化学习实现反应感知的化合物探索  
**检查更新**  
Shogo Nakamura $^{1}$, Nobuaki Yasuo$^{2}$ & Masakazu Sekijima $^{3}$  

设计具有理想性质的分子是药物发现中的一项关键工作。由于深度学习的最新进展，分子生成模型已经得到发展。然而，现有的化合物探索模型常常忽视确保有机合成可行性的重要问题。为了解决这个问题，我们提出了TRACER，这是一个将分子属性优化与合成路径生成相结合的框架。该模型可以在反应类型约束下，通过条件Transformer预测给定反应物经由特定反应类型生成的产物。针对DRD2、AKT1和CXCR4的活性预测模型的分子优化结果表明，TRACER有效地生成了具有高评分的化合物。能够识别整个结构的Transformer模型捕捉了有机合成的复杂性，使其能够在考虑实际反应性约束的同时，在广阔的化学空间中进行导航。

小分子是药物发现中必不可少的模态之一。然而，新药物候选物的发现变得越来越困难，有些需要超过12年的时间来开发，平均耗资26亿美元才能进入市场$^{1,2}$。在研究的早期阶段，通常通过高通量筛选或虚拟筛选来识别对靶蛋白具有高抑制活性的先导化合物（hit compounds）。然后，通过添加构建块对这些先导化合物进行结构扩展，以探索具有所需性质的化合物$^{3}$。为了从先导化合物出发，在广阔的化学空间中找到药物候选化合物，提高DMTA（设计、合成、测试、分析）循环的效率至关重要。

深度学习模型及其泛化能力的最新进展导致了用于从头设计和先导化合物优化目的的生成模型的发展$^{4}$。然而，这些方法大多只关注“制造什么”，而没有充分考虑“如何制造”。利用各种架构（如循环神经网络（RNNs）$^{5–8}$、变分自编码器（VAEs）$^{9,10}$和生成对抗网络（GANs）$^{11,12}$）的分子生成模型，以及明确考虑与靶蛋白相互作用的分子设计方法$^{13–15}$已经被开发出来。由于这些方法没有考虑如何合成生成的分子，化学家必须从生成模型建议的大量化合物中寻找合成实验的候选者；这是转向实验台合成的一个重大障碍$^{16}$。

最常用的评估合成难易程度的评分函数SA score$^{17}$，会对PubChem数据库中罕见的片段、手性中心和某些子结构进行惩罚。最近，Chen等人开发了一种使用构建块和化学反应数据集的评分方法，以改进SA score$^{18}$。然而，这些拓扑方法非常适合高通量地粗略估计合成难易程度，无法真正考虑合成可行性$^{19}$。

尽管已经提出了逆合成模型来计算预测合成路径$^{20}$，但它们无法考虑化学反应的复杂性；因此，在合成规划过程中，人类专家的知识和经验是不可或缺的$^{21}$。尽管最近基于机器学习的评分函数$^{22–27}$取得了进展，但这些函数仅预测逆合成模型的结果。因此，这些模型的根本问题仍未解决$^{21}$。

解决生成可合成分子问题的一种方法是使用预定义的反应模板集，通过分子设计任务构建虚拟库，这些模板描述了反应物如何转化为产物$^{28–34}$。机器学习的进步导致了与优化算法（例如遗传算法或贝叶斯优化）相结合的基于深度学习的类似方法的发展$^{35–38}$。这些方法生成在描述符空间（如扩展连通性指纹（ECFPs））中与给定输入化合物相似的分子，并且还生成由反应模板描述的反应路径。已经报道了强化学习模型，如actor-critic和蒙特卡洛树搜索（MCTS），可以通过训练神经网络来预测最大化奖励函数值的反应模板和构建块，从而生成优化的化合物$^{39–42}$。最近，报道了一种利用GFlowNet的基于模板的方法，用于生成概率与奖励值成正比的化合物$^{43,44}$。然而，这些模型中使用的反应模板过于简化，化学选择性和区域选择性这些在实际化学反应中的重要方面难以考虑（图1a）。尽管通过详细描述目标反应的模式可能会考虑这些选择性，但全面定义所有可能反应的底物范围是不现实的。由于新的化学反应每天都在报道，描述详细的化学转化是一个耗时的过程。

作为一种数据驱动的方法，开发能够明确学习化学反应的结构优化方法有潜力解决上述挑战。Bradshaw等人提出了将分子或反应树嵌入到潜在空间的模型，这是一个从输入数据中学到的低维表示；这里，相似的数据点（即化学结构或反应树）被嵌入以使它们彼此接近$^{45,46}$。Molecule Chef $^{45}$ 从“反应物袋”（反应物嵌入的总和）学习潜在空间，并将这些反应物与Molecular Transformer$^{47}$预测的产物性质联系起来。尽管这个过程能够在考虑真实化学反应的同时生成化合物，但它只能处理单步反应。DoG-Gen$^{46}$通过分层门控图神经网络（GGNN）成功映射了反应树的潜在向量，实现了多步合成路径的生成。然而，在结构优化的情况下，DoG-Gen无法遵循药物发现过程；它无法从特定的先导化合物开始生成化合物。

在本研究中，我们提出了TRACER，这是一个结合了条件Transformer和MCTS的分子生成模型。Transformer模型基于注意力机制构建，允许其选择性地直接关注输入序列数据中包含的重要组件$^{48}$。因此，在化学反应（如简化分子输入行条目系统（SMILES）$^{49}$中包含的结构转化）上训练的Transformer模型有望从整个反应物中识别影响反应的子结构，从而在正向合成预测任务中实现高精度$^{47}$。先前开发的数据驱动方法$^{45,46}$在潜在空间中生成优化结构，因此它们可以在平滑填充连续空间的结构中搜索具有所需性质的化合物（图1b）。与以前使用端到端架构的工作相比，我们提出的框架将作为产物预测模型的Transformer与用作结构优化算法的MCTS解耦。因此，TRACER能够基于虚拟化学反应进行直接优化；因此，它有望提出远离训练数据集的结构和合成路径（图1c）。我们的模型学习了1000种与真实化学转化相关的反应类型，使模型能够比使用大约100个反应模板的基于规则的方法$^{35,36,38–42}$学习更细粒度的信息。随着化学反应数据库的不断增长，模型需要应用于日益多样化的反应。然而，在自然语言处理任务中，Transformer模型已被验证遵循缩放定律$^{50}$，这表明增加模型参数的数量可以有效地应对这一挑战。

（

### TRACER 的核心思路

TRACER 是一个新的分子生成框架，目标是 **同时优化分子性质和合成路径**。

它的关键设计是：

1. **条件 Transformer**
   - 用 Transformer 学习反应规律。
   - 给定一组反应物和反应类型，它可以预测可能的产物。
   - 因为 Transformer 能捕捉复杂的全局依赖，所以它能学到比模板更细致的化学反应规律。
2. **蒙特卡洛树搜索（MCTS）**
   - 用于在化学空间里探索更优的分子。
   - 它像搜索算法一样，不断尝试不同的组合，寻找最优解。
3. **反应类型约束**
   - TRACER 学习了 **1000 种真实的反应类型**（远多于传统模板方法的 ~100 个）。
   - 这样模型能在生成新分子的同时，自动考虑“能不能合成出来”。

）

## 结果与讨论  
### 条件标记对化学反应的影响

在本研究中，Transformer模型在通过SMILES序列将反应物和产物分别作为源分子和目标分子创建的分子对上进行训练。遵循Yoshikai等人$^{51}$提出的评估方法，计算了验证数据的部分准确率（每个SMILES标记的准确率）和完美准确率（每个分子的准确率），以评估模型识别部分和整个产物结构的能力。研究了包含或排除反应模板信息对模型准确率的影响（图2）。部分准确率迅速达到约0.9，无论是否存在反应模板信息；这一结果表明模型能够从反应物中学习部分产物结构，这与化学知识一致，即反应物的骨架通常保留在产物中。然而，完美准确率在有条件模型和无条件模型中的增长都比部分准确率慢；因此，学习整个产物结构需要更多时间。完美准确率在有条件模型中稳定在约0.6，在无条件模型中稳定在约0.2。这些结果表明，在没有反应模板的情况下，模型难以重现训练数据中的化学反应，而在提供此信息时准确率有所提高。这些结果表明了一个事实：在没有额外约束的情况下，单一底物可以经历无数化学反应并产生各种产物。

计算了模型在测试数据上达到的top-n准确率（表1）。通过添加反应模板信息，top-1、top-5和top-10准确率得到了提高。这些结果表明，关于化学反应的知识缩小了预测分子的化学空间，并提高了模型从学习的化学反应中生成适当产物的能力。

在某些情况下，观察到无条件Transformer从反应物生成的产物存在偏差。图3a显示了一个例子，其中在训练好的无条件Transformer中，使用束宽（beam width）为100的束搜索生成了top-10产物，其中描绘的反应物的SMILES序列用作输入。生成的产物偏向于涉及吲哚型氮原子上的Csp(3)键的反应转化的化合物；这些结果归因于USPTO数据集中涉及类似转化的化学反应的偏差。相反，通过使用条件Transformer和GCN预测的反应模板，可以通过多样的化学反应生成化合物（图3b）。GCN不仅选择了训练数据中包含的反应物和反应模板的组合，如环氧环开环反应和与乙烯基卤化物的Buchwald-Hartwig交叉偶联，还选择了训练数据中不存在的化合物和反应模板的组合，如酰胺化和磺酰胺化，并成功生成了合理的候选产物。这些结果表明，条件Transformer可以从条件标记中提取化学反应知识，即使对于未知的反应物和化学反应组合，也能提出合理的产物。在反应模板不匹配（未找到匹配的子结构）的条件下生成的产物结果示例见补充图1。

（

## 通俗解释版 · 条件标记对化学反应的影响

### 模型怎么训练的？

研究里用 **Transformer** 来预测化学反应的产物：

- 输入：反应物的 **SMILES 表示**（分子的字符串形式）。
- 输出：预测的产物（也是 SMILES）。
- 训练目标：让模型学会“从反应物推断产物”。

评估指标有两种：

1. **部分准确率**：产物字符串里的每个片段对不对。
2. **完美准确率**：整个产物的 SMILES 是否完全正确。

）

### 通过QSAR模型针对特定蛋白质生成合成路线的化合物优化

从USPTO 1k TPL数据集中为DRD2、AKT1和CXCR4选择了五个起始材料。详细过程在第4.5节中描述，所选分子如图4所示。利用这些反应物作为根节点，MCTS计算从选择到回溯执行了200步。扩展步骤中预测的反应模板数量设置为10。这些超参数可以根据可用的计算资源进行调整。

通过MCTS生成的化合物根据Transformer推理过程中的束宽、生成的分子总数（Total）、与USPTO数据集中所含分子相比唯一分子的比例、分子数量（Num）以及QSAR模型预测活性概率超过阈值0.5的分子比例（Proportion）进行评估（表2、3和4）。除了这些指标之外，为了确认MCTS的深度，还研究了生成化合物的反应步骤比例（图5）。唯一性分析显示，TRACER在所有靶蛋白上均有效地生成了与USPTO数据库中不同的化合物。生成QSAR值超过0.5的化合物的效率受起始化合物的显著影响。对于DRD2和AKT1，当起始材料具有高QSAR值时，结构探索过程从与QSAR模型学习的活性化合物相似的骨架开始，导致发现具有高奖励值的分子比例更高。然而，即使从具有低QSAR值的化合物开始，通过调整Transformer的束宽，TRACER也成功地生成了具有高QSAR值的化合物。对于CXCR4，从具有低QSAR值的化合物（起始材料11和12）开始很难发现优化的化合物。但是，当从化合物13-15开始时，我们的模型成功地以约15%的速率生成了QSAR值超过0.5的化合物。从化合物11、12和15开始，提高QSAR值具有挑战性。这可能是因为已知的CXCR4配体数量相对较少，并且与QSAR模型学习的活性化合物相似的、可以从这些起始材料访问的骨架很少，从而限制了进一步开发的机会。

总体而言，发现条件Transformer模型的最佳束宽因起始化合物而异。如图5所示，增加束宽导致深度探索过程减少。相反，较窄的束宽增加了深度，导致更多的反应步骤。这些结果表明，在某些情况下，通过缩小树宽来促进深度探索比通过扩大树宽来分散要探索的节点更有效地发现具有高奖励值的分子（起始材料2-3、6、9和11）。相反，在其他情况下，增加束宽通过促进广度探索促进了高奖励分子的生成（起始材料4、10和13）。这些发现表明，TRACER能够通过优化束宽，从各种初始化合物开始进行结构优化。

为了研究增加束宽对生成分子多样性的潜在影响，计算了超过阈值的化合物集的内部多样性。在某些情况下，当束宽增加时，观察到多样性下降的趋势（起始材料1、3、7、9和12），而对其他材料没有观察到显著影响。

在我们的模型中，由于某些反应物是隐式处理的，因此还研究了新添加到输入化合物中的反应物。对于表2、3和4中生成的所有路径，总共识别出20,765种不同的反应物，使该模型能够生成与USPTO数据集相比具有独特结构的分子。

（

**起始材料选择**

- 从一个大型化学反应数据库（USPTO）中选择了15个起始化合物作为"原料"
- 就像厨师选择食材来制作不同的菜品一样

**MCTS算法**

- 使用蒙特卡洛树搜索算法来探索可能的化学反应路径
- 就像下棋时AI会提前计算很多步可能的走法一样
- 设置了200步的搜索深度，每次可以预测10种可能的反应模板

**评估标准** 研究者用几个指标来评估生成的化合物质量：

- **唯一性**：生成的分子是否与已知数据库中的不同
- **活性预测**：QSAR模型预测这些分子对目标蛋白质的活性概率
- **阈值**：活性概率超过0.5被认为是有希望的候选药物

##### 主要发现

**成功案例**

- 对于DRD2和AKT1蛋白质，当从高活性的起始材料开始时，更容易生成高活性的新化合物
- 即使从低活性材料开始，通过调整搜索策略也能找到好的候选分子

**挑战案例**

- CXCR4蛋白质比较困难，因为已知的有效药物较少
- 从某些起始材料很难生成高活性化合物，成功率只有约15%

**搜索策略的影响**

- 束宽

  就像搜索的"广度"：

  - 窄束宽 = 深度探索，走更多反应步骤
  - 宽束宽 = 广度探索，尝试更多不同方向

- 最佳策略因起始材料而异，需要针对性调整

）

### 模型生成的反应路径

图6显示了每个起始材料具有最高奖励值的化合物及其合成路线。在这些合成路径中，并非所有反应物都被明确处理。因此，通过Reaxys$^{52}$研究了这些图中显示的构建块的可用性，并且发现所有起始材料都是可商购的（补充图2-16）。这一结果表明，USPTO中的广泛反应物是可购买或可合成的，并且条件Transformer捕捉到了这一特征，即使模型隐式地学习了部分反应物。

从1开始，选择了一系列反应，包括醇对芳基氯的亲核攻击和还原胺化；这产生了QSAR值为0.962的1c。在合成1a期间，1的芳基氟骨架和新添加的2-氯吡嗪（补充图2）可能在芳香亲核取代反应中竞争。然而，USPTO数据集中包含几个类似的反应，其中脂肪醇与氯吡嗪反应，并且存在实际的专利$^{53}$。这一结果表明，TRACER通过学习真实的化学反应正确预测了芳香亲核取代反应的选择性。预测这种选择性对于基于反应模板的方法来说是极其困难的，这些方法在不考虑真实化学反应的情况下确定适用的反应模板。对于化合物2，利用胺的磺酰胺化来合成2a，然后进行亲核取代和对烷基卤的亲核攻击以产生2c。在3的情况下，通过与新戊酰氯进行酰胺化生成3a。3a通过Friedel-Crafts酰基化转化为3b。通常，当sp3碳与芳香环键合时，Friedel-Crafts反应选择性地发生在对位$^{54}$，条件Transformer模型成功预测了这一点。从4的衍生涉及酰胺化以及随后杂原子对烷基卤化物的两次亲核攻击以产生4c。当使用5作为起始材料时，使用邻苯二甲酰亚胺衍体在哌嗪氮上引入脂肪胺部分得到5b，随后的酰胺化得到QSAR值为0.994的5c。

使用AKT1和CXCR4的QSAR模型进行的实验结果分别显示在图7和图8中。在这些反应路径中发现了几个例子，其中化学反应的选择性被正确预测。在从8a到8b的转化中，已经报道了一种在保留氰基的同时还原硝基的方法$^{55}$。在合成8d期间，正确预测了吡啶3位上的取代和苯胺的邻对位导向效应。在从9a到9b的合成中，已经报道了类似的反应，其中使用金属催化剂对底物中酰胺基团的邻位进行卤化$^{56,57}$。对于11a，已经报道了完全相同的反应，其中还原胺化发生的同时保留了醇$^{58}$。在从11a到11b的转化中，芳基氟 undergo 取代反应，同时保留了新添加构建块的醛；确实，存在几个这样的例子$^{59,60}$。如这些例子所示，TRACER可以通过明确学习实际化学反应来预测化学反应的选择性。

尽管探索了使用无条件Transformer的结构优化过程，但生成的合成路线由于使用了几个不合理的转化而被证明不切实际，例如在单次推理中执行多步反应、将碳原子插入烷基链以及忽视选择性。因此，研究了通过以DRD2的QSAR模型作为奖励函数进行结构优化生成的路径中反应物Murcko骨架$^{61}$的保留率。条件Transformer的保留率为94.5%，而无条件Transformer的保留率为43.3%。将这些结果与第5.3节进行的完美准确率研究结合起来，可以得出结论，对化学反应进行条件化是至关重要的。

（

主要发现：AI生成的合成路径是可行的

）

### 与结合合成路线生成的分子生成模型的比较分析

为了评估TRACER，使用基准模型（Molecule Chef $^{45}$、DoG-Gen$^{46}$、CasVAE$^{37}$和SynFlowNet$^{43}$）在DRD2、AKT1和CXCR4的QSAR模型上进行了结构优化实验。这些模型之间的比较如表5所示。指定起始化合物是药物发现中的关键功能，只有我们的框架支持此功能。此外，能够明确学习化学反应的模型仅限于TRACER、Molecule Chef和DoG-Gen。值得注意的是，Molecule Chef仅适用于单步反应。尽管CasVAE和SynFlowNet是使用反应模板的基于规则的方法，能够执行伴随反应路径生成的结构优化，但它们无法考虑实际的化学反应。每个模型提出的优化方法如下所述。

**Molecule Chef**  
在Molecule Chef中，参与化学反应的 reaction物通过图神经网络（GNNs）进行向量化。通过利用“反应物袋”（通过求和这些向量化表示获得）来学习映射到潜在空间的网络。在原论文中，将属性预测器连接到潜在空间，并通过计算潜在变量的梯度来更新坐标以生成优化分子。为了进行比较实验，通过预先计算原论文中使用的训练数据集（源自USPTO数据集）的QSAR值来训练属性预测器。

**DoG-Gen**  
DoG-Gen是本比较研究中唯一能够明确学习化学反应并处理多步转化的基准模型。通过使用两阶段门控图神经网络（GGNN）对从USPTO数据集构建的反应树进行编码来学习到潜在空间的映射。在基准测试中，遵循原论文的方法，执行了30轮操作。每轮涉及从DoG-Gen采样7000个反应树，并对按QSAR值排名前1500的分子进行两次微调。本实验中使用了与原论文相同的训练数据集，该数据集源自USPTO。

**CasVAE**  
在CasVAE中，为化合物和反应树构建了独立的潜在空间，利用通过基于反应模板的逆合成模型$^{62}$分解USPTO数据集中的分子获得的路径。通过对从每个空间采样的潜在变量进行解码。根据原始实现，通过对分子潜在空间应用贝叶斯优化，生成了五批3000个分子。

**SynFlowNet**  
SynFlowNet是一个基于GFlowNet的模型，旨在以与奖励值成比例的概率采样数据点。通过神经网络执行预测化合物适用的反应模板和双分子反应模板的反应伙伴的任务，并将基于化学反应的结构扩展过程建模为马尔可夫过程。原论文使用了源自Enamine构建块$^{63}$的构建块。在本比较实验中，为了标准化条件，使用与原论文相同的预处理程序将训练数据替换为USPTO数据集。实验条件设置为与原始实现相同；生成步骤的批次大小为64，温度参数为32，训练步数为1000。

**实验结果**  
对于所有基准测试模型，通过DRD2、AKT1和CXCR4的QSAR模型研究了它们的化合物生成效果（表6）。对于TRACER，使用了表2-4中为每个起始材料优化的束宽组合。除了第2.2节使用的指标之外，还计算了生成化合物中唯一化合物的比例（唯一性）和Fréchet ChemNet距离（FCD）$^{64}$，以评估生成化合物集与训练数据集之间的相似性。排除了唯一化合物数量少于5000的情况，因为原论文表明当化合物数量较少时，特别是低于此阈值时，FCD不可靠。

当以DRD2为目标时，TRACER呈现出最高的化合物唯一性和最大的USPTO唯一分子比例。SynFlowNet生成了大量重复化合物，导致唯一化合物仅占10.9%。DoG-Gen的USPTO唯一化合物比例为41.3%，表明由于其学习了从反应树到潜在空间的映射，生成了大量与训练数据重叠的化合物。此外，Molecule Chef和DoG-Gen的FCD值都很低，这表明通过反应树构建潜在空间的方法倾向于探索接近训练数据的区域。SynFlowNet表现出最高的FCD，这归因于其基于反应模板的方法，其中各种反应模板可以应用于起始材料以增加FCD。TRACER也观察到了14.3的高FCD，表明其在明确学习化学反应的同时，成功完成了生成与训练数据不相似的化合物这一具有挑战性的任务。CasVAE在QSAR值大于0.5的化合物比例和总体多样性方面显示出最佳结果。这可能是因为CasVAE仅对化合物结构的潜在空间执行贝叶斯优化，因此执行了一个更简单的任务，没有考虑反应树的潜在表示。因此，如第2.4.3节所示，CasVAE尝试生成15,000个反应树，但仅产生615个有效树。这主要是由于在反应树生成过程中选择了错误的反应模板以及未能构建完整的反应路径，导致缺乏高通量能力。

当以AKT1和CXCR4为目标时，TRACER呈现出USPTO唯一化合物比例和FCD的最高值。这一结果表明，TRACER在考虑真实化学反应的同时，针对奖励函数稳健地探索了广阔的化学空间。关于超过QSAR阈值的化合物比例，CasVAE对AKT1呈现出最高值，而DoG-Gen对CXCR4呈现出最高比例。如第5.2节所述，DRD2、AKT1和CXCR4的活性化合物数量分别为9,913、4,035和925。这表明CasVAE的性能显著受已知配体数量的影响，使其难以发现CXCR4的优化分子。DoG-Gen表现出最高程度的多样性，这可能是由于没有初始化合物选择阶段，允许在探索化学空间时具有更大的自由度。TRACER在生成具有高QSAR值的化合物比例方面取得了中等性能（DRD2为17.0%，AKT1为25.5%，CXCR4为13.5%），与DoG-Gen和CasVAE相比。值得注意的是，这种能力显著取决于最初选择的化合物。为了评估我们的框架在奖励函数方面的优化能力，有意将具有低QSAR值的化合物设置为起始化合物。然而，其他方法没有如此强的约束，允许在搜索过程中具有更大的灵活性。

每个模型生成的化合物的SA分数和分子量的平均值和标准偏差显示在表7中。关于SA分数，除针对CXCR4的SynFlowNet外，所有模型都表现出相当的性能。尽管SA分数包含一个尺寸惩罚项$^{17}$，旨在惩罚较大的分子，但TRACER生成的化合物具有比Molecule Chef和DoG-Gen更高的分子量，同时保持了可比的SA分数。

（

#### 实际应用意义

**药物合成难度（SA分数）**：

- 所有模型生成的分子都具有相当的合成难度
- TRACER虽然生成的分子稍大，但合成难度保持合理

**真实世界的价值**：

1. TRACER的独特优势：
   - 能从现有药物出发进行改进（起始化合物功能）
   - 生成的反应路径在现实中可行
   - 平衡创新与可行性
2. 行业应用前景：
   - 制药公司可以指定特定的起始材料
   - 生成的分子既新颖又可以实际合成
   - 为药物开发提供更实用的解决方案

这个对比研究证明了TRACER在药物设计AI领域的独特价值：它不是在所有指标上都最好，但它提供了其他模型无法提供的关键功能——**在保证化学可行性的前提下，从指定起点出发进行创新药物设计**。

）

### 从未见分子开始涉及QSAR模型的结构优化实验

进行了旨在模拟实际药物发现过程的结构优化实验。如第5.3节所述，从ZINC数据库获得了一个由非专利样化合物组成的数据集。然后为每个蛋白质选择具有最高配体效率（LE）的化合物，LE定义为绝对对接分数除以重原子数归一化。结果，选择16用于DRD2（对接分数= -5.1，LE = 0.729）和AKT1（对接分数= -4.5，LE = 0.643），而选择17用于CXCR4（对接分数= -4.7，LE = 0.671）（图9）。

使用化合物16和17作为起始材料进行的实验结果呈现在表8中。尽管这些初始化合物表现出高LE，但它们的QSAR值对于DRD2、AKT1和CXCR4分别为0.00、0.01和0.01。然而，TRACER通过优化这些化合物的结构增强了奖励值。通过束宽调查，以11.2%（束宽=10）的速率生成QSAR值超过阈值的化合物用于DRD2，24.5%（束宽=10）用于AKT1，以及9.81%（束宽=50）用于CXCR4。与第2.2节呈现的结果一致，大多数生成的化合物未包含在训练数据集中。

对于每个靶蛋白，将TRACER生成的化合物集与已知配体进行了比较。计算了已知配体与TRACER生成的QSAR值大于0.5的化合物之间的成对Tanimoto系数。具有最高相似性的化合物对、生成化合物的合成路线以及对接姿势显示在图10中。尽管这些已知配体未包含在训练数据中，但为DRD2、AKT1和CXCR4生成了Tanimoto相似性分别为0.838、0.588和0.552的化合物。对于这些生成的化合物，TRACER生成了由2、3和4步组成的合成路线。尽管QSAR模型没有明确考虑对接分数，但这些生成的化合物表现出与活性配体相当的结合潜力。这些结果表明，即使从训练数据集中未包含的分子开始，我们的框架也可以有效地为奖励函数生成优化化合物。此外，可以通过QSAR模型生成与未包含在训练数据集中的活性配体结构相似的化合物，同时生成反应路径。

（

这个实验证明了TRACER具有真正的"药物发现直觉"——它能够：

1. 识别分子结构中的潜在优化点
2. 设计合理的化学修饰策略
3. 生成与已知有效药物相似但全新的化合物
4. 提供实际可行的合成路径

）

## 结论

在本研究中，开发了一个用于预测多样化化学反应的条件Transformer模型。与无条件模型相比，所提出的模型显著提高了输出产物预测的准确性。通过整合反应模板信息，该模型能够通过各种化学反应生成分子。

研究了TRACER（条件Transformer、GCN和MCTS的组合）通过生成合成路径来优化针对特定蛋白质的化合物的能力。使用QSAR模型预测DRD2、AKT1和CXCR4活性的基于MCTS的分子生成实验证明了条件Transformer在药物发现方面的适用性。具有各种搜索宽度的MCTS表明，在某些情况下，加宽搜索宽度能够有效生成具有高奖励值的化合物。相反，在其他情况下，缩小搜索宽度有助于更深入的搜索过程，从而发现具有高奖励值的分子。此外，该模型通过附加可商购的构建块执行结构转化，尽管它隐式地学习了数据集中的部分反应物。在基准测试中，TRACER探索了比其他模型距离训练数据更远的化学空间。此外，即使从训练数据中不存在且QSAR值接近零的化合物开始，TRACER也成功地生成了具有高奖励值的化合物。这些结果表明我们的框架对于起始化合物和靶蛋白的变化具有鲁棒性。整合其他模型，例如反应条件推荐模型$^{65,66}$或用于先导化合物优化的多目标优化模型$^{67}$，有望进一步加速药物发现任务中优化新化合物的过程。未来的工作需要通过使用更大的数据集优化模型并探索其涉及多样化奖励函数的潜在应用来提高所提出方法的鲁棒性。

（

#### 主要技术突破

**条件Transformer模型的成功**

- 开发出了能够预测复杂化学反应的AI模型
- 比起"盲目"的无条件模型，这个"有指导"的模型准确性大幅提升
- 就像从一个随意涂鸦的画家升级为能按要求精确作画的专业画师

#### TRACER系统的核心能力

**三大组件协同工作**：

1. **条件Transformer**：理解化学反应规律
2. **GCN（图卷积网络）**：理解分子结构
3. **MCTS（蒙特卡洛树搜索）**：智能探索可能的合成路径

**就像一个完整的药物设计团队**：

- 化学家（Transformer）：懂反应机理
- 结构分析师（GCN）：理解分子形状和性质
- 策略规划师（MCTS）：制定最优搜索策略

）

## 方法

图11a显示了所提出模型用于从给定反应物生成可行产物并基于虚拟化学反应优化结构的流程。首先，使用图卷积网络（GCN）预测反应物适用的反应模板。为了过滤不匹配的模板，此时会移除那些与反应物没有子结构匹配的模板。其次，Transformer在针对反应物预测的反应类型条件下执行分子生成。此操作使模型能够通过学习与条件模板对应的真实化学反应来意识到反应物应如何转化为产物。每个反应类型的输出数量在第2节中进行了检查。对于仅有一个合适产物（例如亲核取代基的脱保护）的反应模板，手动用反应模板的索引进行注释。假定这些模板仅生成单个分子。图11b显示了使用MCTS的优化框架概述。在本研究中，MCTS使用模拟选择性地增长搜索树并与Transformer和GCN结合确定最有希望的行动。它通过四个步骤搜索最优化合物：选择（Selection）、扩展（Expansion）、模拟（Simulation）和回溯（Backpropagation）。在选择步骤中，通过第4.3节描述的上置信界（UCB）分数确定最有希望的分子。在扩展步骤中，扩展分子适用的反应模板，并且Transformer对每个反应模板执行条件正向合成预测。在模拟步骤中，通过进一步探索树来评估新生成分子的潜力。最后，估计值传播回根节点，确保在下一个周期中更可能选择具有更高值的分子。所有实验的计算环境是Ubuntu 22.04操作系统、Intel(R) Xeon(R) Gold 5318Y CPU、128 GB RAM和NVIDIA GeForce RTX 4090 GPU（具有24 GB内存）。

（

**TRACER就像一个智能的化学实验室**，包含两个主要工作站：

### 第一工作站：反应预测系统

**图卷积网络（GCN） - "化学反应顾问"**

- 分析给定的起始分子结构
- 预测这个分子可以进行哪些类型的化学反应
- 就像一个经验丰富的化学家看到分子后说："这个分子可以做A反应、B反应、C反应"

**模板过滤器 - "可行性检查员"**

- 剔除那些与分子结构不匹配的反应模板
- 就像检查钥匙是否与锁孔匹配，去除明显不合适的选项

### 第二工作站：产物生成系统

**条件Transformer - "智能反应器"**

- 在确定了可能的反应类型后，具体预测反应产物
- "条件化"意味着它知道当前要进行什么类型的反应
- 就像一个懂得根据不同烹饪方式来处理食材的智能厨师

**特殊处理机制**

- 对于只有一种可能产物的反应（如脱保护反应），系统会直接给出结果
- 就像某些固定搭配的菜谱，不需要太多选择

#### 优化框架：MCTS搜索系统（图11b）

**MCTS就像一个智能的路径规划师**，通过四个步骤来寻找最优的药物设计方案：

#### 1. 选择（Selection） - "策略制定"

- 使用UCB（上置信界）分数来评估哪个分子最有希望
- 就像投资顾问根据风险和收益来选择最值得投资的项目
- 平衡"已知的好选择"和"未知但可能更好的选择"

#### 2. 扩展（Expansion） - "探索新可能"

- 对选中的分子，列出所有可能进行的化学反应
- Transformer对每种反应类型生成具体的产物
- 就像从一个路口探索所有可能的道路方向

#### 3. 模拟（Simulation） - "前景评估"

- 对新生成的分子进行进一步探索，评估它们的潜力
- 就像试想如果走这条路，最终能到达什么目的地

#### 4. 回溯（Backpropagation） - "经验积累"

- 将评估结果反馈给整个搜索树
- 更新每个节点的价值估计
- 就像把探索的结果记录在地图上，为下次决策提供参考

#### 系统工作流程

**完整的药物设计过程**：

1. **输入起始分子** → GCN分析可能的反应类型
2. **过滤不可行反应** → 保留化学上合理的选项
3. **Transformer生成产物** → 针对每种反应类型预测具体产物
4. **MCTS智能搜索** → 通过四步循环找到最优路径
5. **输出优化分子** → 得到改进的药物候选分子和合成路线

）

### 条件Transformer模型

近年来，基于Transformer架构$^{48}$的化合物生成模型的发展显著推进了$^{68,69}$。这些模型利用注意力机制来学习序列数据（如SMILES字符串）中的长程依赖关系。值得注意的是，Molecular Transformer$^{47}$在正向化学反应预测任务中实现了高精度。它在USPTO数据集上训练，以从给定反应物预测产物。与先前开发的在提供所有起始材料时生成产物的正向合成预测模型不同，我们的模型被开发用于预测可以通过各种化学反应从单一起始材料合成的化合物。

采用条件Transformer来控制虚拟化学反应的多样性。先前关于Transformer模型的研究已经报道，通过附加与所需性质对应的条件标记可以成功控制生成分子的性质$^{70–72}$。在本研究中，将从USPTO 1k TPL数据集$^{73}$提取的1000个反应模板的索引添加到输入SMILES序列的开头，并训练Transformer预测与该反应对应的产物（图12）。进行这些操作是为了通过将各种反应模板条件化于反应物上来控制化学反应的多样性。如第5.1节所述，反应数据集仅将单个反应物作为显式输入考虑。然而，在大多数存在伙伴化合物的情况下，可以从反应模板以及反应物和产物之间的结构差异来确定它们。这种训练方法使模型能够学习描述化学反应所需的所有信息，同时避免从数万种可能存在的分子中选择伙伴化合物。因此，预期该模型能够参考真实的化学反应生成合理的结构转化。

为了阐明条件标记的效果，通过PyTorch$^{74}$和torchtext库实现了条件Transformer和无条件Transformer模型。超参数根据原始Transformer模型$^{48}$确定如下：批次大小为128，模型维度为512，dropout比例为0.1，编码器和解码器的层数为6，前馈网络维度为2048，激活函数为整流线性单元（ReLU），注意力头数为8。

（

#### 本研究的创新之处

**单一起始材料的挑战**

- 与以往不同，这个模型设计用于从**单一起始材料**预测可能的产物
- 这更符合实际药物开发的需求
- 就像给厨师一种主要食材，让他想出各种可能的菜谱

#### 条件化的核心思想

**什么是"条件化"？**

- 在输入的SMILES字符串开头添加一个特殊的"条件标记"
- 这个标记来自1000个反应模板索引中的一个
- 就像给厨师不仅提供食材，还告诉他要用什么烹饪方法（煎、炒、炸、煮等）

**工作原理示例（图12）**：

```
输入：[反应模板651] + [起始分子SMILES]
输出：[预测的产物SMILES]
```

）

### 图卷积网络 (GCN)

GCN模型$^{75}$是基于先前一项从化合物的SMILES表示预测反应模板的研究$^{76}$实现的。具有n个节点的分子表示为$M = (A, E, F)$，其中$A \in \{0, 1\}^{n \times n}$是邻接矩阵，$F \in \mathbb{R}^{n \times d}$是节点特征矩阵，$E \in \{0, 1\}^{n \times n \times T}$是当存在T种键类型时的边张量。使用这些表示和$A = \{A^{(t)} \mid t \in T\}$，分子可以表示为$M' = (A, F)$，其中如果节点i和j之间存在类型t的键，则$(A^{(t)})_{i,j} = 1$。GCN的架构通常由卷积层、密集层和聚合层组成。

**图卷积层**  
图卷积计算如下：
$$ X^{(l+1)} = \sigma \left( \sum_{t} \tilde{A}^{(t)} X^{(l)} W^{(l)}_{t} \right) $$
其中$\tilde{A}^{(t)}$是键类型t的归一化邻接矩阵，$X^{(l)}$是第l层的输入矩阵，$W^{(l)}_{t}$是第l层键类型t的参数矩阵。

**密集层**  
在应用图卷积之后，输出$X^{(l+1)}$通常通过一个密集层（也称为全连接层）。密集层独立地对每个节点的特征应用线性变换。输出$X^{(l+1)}$可以表示如下：
$$ X^{(l+1)} = X^{(l)} W^{(l)} $$
其中$X^{(l)}$是输入矩阵，$W^{(l)}$是密集层的权重矩阵。

**聚合层**  
在图卷积和密集层之后，需要对得到的节点表示进行聚合以获得图级表示。遵循先前的工作$^{76}$，采用求和聚合。输出计算如下：
$$ X^{(l+1)}_{j} = \sum_{j} X^{(l)}_{ij} $$
其中$X^{(l+1)}_{j}$是向量表示的第j个元素。通过求和聚合获得的图级表示$X^{(l+1)}$可以进一步处理以产生GCN模型的最终输出。应用额外的密集层来预测反应模板。GCN模型的整体架构如图13所示。

对于训练数据中包含的反应物，GCN模型被训练从1000个可用模板中预测正确的反应模板。这意味着随机选择反应模板的准确率将为0.001。GCN模型的超参数通过Optuna$^{77}$确定（表9）。最终，在测试数据上评估训练好的GCN模型，得到top-1准确率为0.483，top-5准确率为0.771，top-10准确率为0.862。

（

#### GCN的核心作用

**GCN就像一个"化学反应顾问"**

- 输入：一个分子的结构
- 输出：这个分子可能进行的化学反应类型
- 就像一个经验丰富的化学家看到分子后，立即知道它能进行哪些反应

#### 分子的数字化表示

**将分子转换为计算机能理解的形式**

分子用三个数学对象来表示：`M = (A, E, F)`

1. 邻接矩阵 (A)：

   - 记录哪些原子之间有化学键连接
   - 就像一张"朋友关系图"，标记谁和谁是邻居

2. 节点特征矩阵 (F)

   ：

   - 记录每个原子的属性（如碳、氮、氧等）
   - 就像给每个人标记姓名、年龄等个人信息

3. 边张量 (E)：

   - 记录化学键的类型（单键、双键、三键等）
   - 就像标记朋友关系的亲密程度（普通朋友、好朋友、最好的朋友）

#### GCN的三层架构

##### 1. 图卷积层 - "信息传递员"

**工作原理**：

```
X^(l+1) = σ(∑_t Ã^(t) X^(l) W^(l)_t)
```

**通俗解释**：

- 每个原子收集来自邻居原子的信息
- 不同类型的化学键传递不同的信息
- 就像在社交网络中，每个人从朋友那里收集消息，不同关系的朋友提供不同类型的信息

**例子**：

- 碳原子会从与它相连的氧原子、氮原子那里收集信息
- 单键和双键传递的信息权重不同

##### 2. 密集层 - "信息处理器"

**工作原理**：

```
X^(l+1) = X^(l) W^(l)
```

**通俗解释**：

- 对每个原子收集到的信息进行处理和变换
- 就像大脑对接收到的信息进行分析和整理

##### 3. 聚合层 - "总结专家"

**工作原理**：

```
X^(l+1)_j = ∑_j X^(l)_ij
```

**通俗解释**：

- 将所有原子的信息汇总成整个分子的特征
- 就像把所有部门的报告汇总成公司的总体报告

##### 模型训练和性能

**训练目标**：

- 从1000个可能的反应模板中选择正确的
- 随机猜测的准确率只有0.001（千分之一）

**模型性能**：

- **Top-1准确率**: 48.3%（第一次猜中的概率）
- **Top-5准确率**: 77.1%（前5个猜测中有正确答案的概率）
- **Top-10准确率**: 86.2%（前10个猜测中有正确答案的概率）

**性能解读**：

- 相比随机猜测的0.1%，48.3%的准确率是巨大的提升
- 在前10个推荐中，有86.2%的概率包含正确答案

##### 超参数优化

**使用Optuna自动调优**：

- Optuna是一个自动寻找最佳参数组合的工具
- 就像自动调试，找到模型的最佳设置
- 避免了人工试错的繁琐过程

##### 实际工作流程

**完整过程示例**：

1. **输入**：阿司匹林分子

2. **图卷积层**：分析分子中每个原子的环境

3. **密集层**：处理收集到的结构信息

4. **聚合层**：生成整个分子的特征向量

5. 输出预测

   ：推荐可能的反应类型

   - 第1推荐：酯化反应（概率40%）
   - 第2推荐：氧化反应（概率25%）
   - 第3推荐：还原反应（概率15%）
   - ...

）

### 蒙特卡洛树搜索 (MCTS)

利用MCTS算法$^{78}$生成针对奖励函数的优化化合物。与RNN、遗传算法或匹配分子对$^{5–7,79–81}$集成的MCTS已被证明在分子优化任务中是有效的。在我们的框架中，节点代表分子，涉及任何节点的路径代表合成路线。MCTS算法包括四个步骤，每个步骤详细说明如下。

1.  **选择（Selection）**：从根节点开始，根据其估计值和访问次数迭代地选择子节点。树策略是选择子节点的得分函数，在MCTS的性能中起着重要作用。一个常用的度量是UCB分数，它最初是为多臂老虎机问题提出的。这种方法平衡了对访问次数较少的节点的探索和对有希望节点的利用。节点i的UCB分数通过以下公式计算：
    $$ \text{UCB} = \arg \max_{i} \left\{ Q(s_i) + 2C_p \sqrt{\frac{\ln N(s_p)}{N(s_i)}} \right\} $$
    其中$Q(s)$表示状态s对其子节点的平均估计值，$N(s)$表示状态s的访问次数。每个节点i及其父节点的状态分别表示为$s_i$和$s_p$。$C_p$平衡了利用和探索之间的权衡。UCB分数中的第一项对应于利用，鼓励选择具有较高估计值的节点。第二项对应于探索，促进选择访问次数较少的节点。这种选择策略将搜索引导到搜索空间中最有价值的区域，从而能够发现可能具有所需性质的分子。

2.  **扩展（Expansion）**：通过使用Transformer进行虚拟化学反应生成的产物作为新的子节点添加到所选节点。在条件Transformer中，从GCN采样后通过子结构匹配过滤的反应模板用作条件标记。在无条件Transformer中，模型仅预测产物而不考虑反应模板。采用束搜索（一种通过扩展最有希望的节点来探索图的启发式搜索算法）来获得可行的产物，并研究了束宽对分子优化过程的影响。

3.  **模拟（Simulation）**：估计在扩展步骤中生成的子节点的值。通过Transformer重复执行虚拟化学反应预定次数。模拟步骤中采样的反应模板数量设置为5，并为每个反应模板生成单个分子。探索的反应步骤数设置为2，并将生成的化合物（25个或更少）中的最大奖励值设置为每个节点的值。

4.  **回溯（Backpropagation）**：通过模拟步骤计算的值沿着从生成的节点到根节点的路径传播给所有节点，并计算累积分数。通过此步骤高度评估搜索树中包含有希望化合物的方向上的节点；这一步使得能够高效生成针对评估函数优化的化合物。

在本研究中，使用定量构效关系（QSAR）模型作为奖励函数，该模型根据化学结构预测靶蛋白的活性概率，范围从0到1。生成化合物的多样性通过以下方程进行评估，如先前分子生成模型的基准研究$^{82}$中所述：
$$ \text{IntDiv}_p(G) = 1 - \sqrt{ \frac{1}{|G|^2} \sum_{m_1, m_2 \in G} T(m_1, m_2)^p } $$
其中$G$是生成的有效化合物集合，$T$是通过ECFP4（2048位）计算的化合物$m_1$和$m_2$之间的Tanimoto相似性。

（

这段文字详细介绍了TRACER系统中蒙特卡洛树搜索（MCTS）算法的工作原理。让我用通俗的语言来解释这个复杂的搜索策略：

#### MCTS的核心作用

**MCTS就像一个智能的药物发现探险家**

- 在巨大的化学空间中寻找最优的药物分子
- 每个节点代表一个分子，路径代表合成路线
- 目标是找到活性最高的化合物

#### MCTS的四个核心步骤

##### 1. 选择（Selection）- "战略决策"

**UCB分数公式**：

```
UCB = arg max{Q(si) + 2Cp√[ln N(sp)/N(si)]}
```

**通俗解释**： 这就像一个投资顾问的决策公式：

- 第一部分 Q(si)

  ：利用已知信息

  - 代表这个分子的"已知价值"
  - 就像选择历史表现好的股票

- 第二部分 2Cp√[ln N(sp)/N(si)]

  ：探索未知可能

  - 鼓励尝试还没有充分探索的分子
  - 就像投资一些还没被充分研究但可能有潜力的新兴股票

**平衡策略**：

- 既要选择已知效果好的路径（利用）
- 也要尝试未知但可能更好的路径（探索）
- 就像平衡"稳健投资"和"风险投资"

#### 2. 扩展（Expansion）- "生成新可能"

**工作流程**：

1. 选定一个分子后，GCN预测它能进行哪些反应
2. 条件Transformer根据每种反应类型生成具体产物
3. 使用束搜索算法获得最有前景的候选分子

**束搜索**：

- 不是尝试所有可能，而是只保留最有希望的几个
- 就像海选时不是让所有人都进下一轮，而是只选最优秀的几位

#### 3. 模拟（Simulation）- "前景评估"

**模拟参数设置**：

- 每次模拟采样5个反应模板
- 每个模板生成1个分子
- 进行2步反应探索
- 最多产生25个候选分子

**评估方式**：

- 在生成的所有分子中取最高奖励值
- 就像考察一个区域的发展潜力，看最好的那个项目能达到什么水平

#### 4. 回溯（Backpropagation）- "经验反馈"

**信息传播**：

- 将模拟得到的评估结果向上传播到所有相关节点
- 更新整条路径上每个节点的价值估计
- 就像把市场调研结果反馈给决策链上的每个环节

#### 奖励函数：QSAR模型

**QSAR模型作为评分标准**：

- 根据分子结构预测对目标蛋白质的活性概率（0-1分）
- 就像一个专业评委，给每个药物候选分子打分
- 分数越高，说明药物效果越可能好

#### 多样性评估

**多样性公式**：

```
IntDiv_p(G) = 1 - √[1/|G|² ∑ T(m1,m2)^p]
```

**通俗解释**：

- 评估生成的分子集合是否足够多样化
- 避免生成很多相似的分子
- 就像确保投资组合的多样性，不要把鸡蛋放在一个篮子里

**Tanimoto相似性**：

- 用ECFP4指纹（2048位）计算分子相似性
- 就像分子的"指纹识别"，判断两个分子有多相似

#### 完整的MCTS工作流程示例

**实际操作过程**：

1. **开始**：从起始分子（如阿司匹林）开始
2. 选择阶段：
   - 计算UCB分数：阿司匹林节点 = 0.6 + 0.3 = 0.9
   - 选择分数最高的节点继续探索
3. 扩展阶段：
   - GCN预测：可进行酯化、氧化、还原反应
   - Transformer生成：3个新的分子候选
   - 束搜索保留最优的2个
4. 模拟阶段：
   - 对每个候选分子继续2步反应
   - 生成25个终端分子
   - QSAR评分：最高分为0.85
5. 回溯阶段：
   - 将0.85的评分反馈给路径上所有节点
   - 更新每个节点的价值估计

#### 算法优势

**智能搜索策略**：

1. **效率高**：不需要穷尽搜索，重点关注有希望的区域
2. **平衡性**：在利用已知信息和探索未知可能间找到平衡
3. **自适应**：根据反馈自动调整搜索方向
4. **可扩展**：能处理大规模的化学空间搜索

）

### QSAR模型

针对多巴胺受体D2（DRD2）、AKT丝氨酸/苏氨酸激酶1（AKT1）和C-X-C基序趋化因子受体4（CXCR4）开发的活性预测模型被用作MCTS的奖励函数。训练了一个随机森林分类器，基于2048位ECFP6，根据第5.2节中描述的每个蛋白质的活性/非活性标记数据集来预测活性化合物的概率。训练数据集包括DRD2的9006个活性化合物和306,457个非活性化合物，AKT1的3623个活性化合物和14,814个非活性化合物，以及CXCR4的853个活性化合物和3051个非活性化合物。这些模型的超参数通过Optuna$^{77}$进行了优化（表10）。训练模型的性能通过曲线下面积（AUC）进行评估，AUC是一种评估二元分类模型在分类阈值上识别类别能力的性能指标，测试数据集包括DRD2的907个活性化合物和33,943个非活性化合物，AKT1的412个活性化合物和1,636个非活性化合物，以及CXCR4的72个活性化合物和355个非活性化合物。模型对于DRD2达到了0.984的准确率和0.703的AUC，对于AKT1达到了0.958的准确率和0.897的AUC，对于CXCR4达到了0.974的准确率和0.924的AUC。

### MCTS起始化合物的选择

第2.2节中用于化合物生成的MCTS的起始材料是从DRD2、AKT1和CXCR4的QSAR值分布中随机抽样的。从USPTO 1k TPL数据集中移除了具有超过8元环且分子量≥300的分子。由于反应物需要容易发生化学反应的官能团，因此还移除了不包含卤素、羰基、不饱和键或亲核取代基的那些；这将化合物数量从890,230减少到374,675。对于每个蛋白质靶标，选择五个初始化合物如下：计算过滤后化合物的QSAR值，并从每个QSAR值区间（0-0.1、0.1-0.2、0.2-0.3、0.3-0.4和0.4-0.5）中随机抽取一个分子。这些选定的化合物在第2.2节中描述。

## 数据
### USPTO数据集

在本研究中，使用了Schwaller等人在其反应分类工作$^{73}$中报告的USPTO 1k TPL数据集作为化学反应数据集。USPTO 1k TPL源自Lowe$^{83}$创建的USPTO数据库，包含445,000个反应和相应的反应模板。通过选择通过RxnMapper$^{84}$进行原子映射和模板提取获得的1000个最频繁的模板哈希来策划反应模板。为了进一步预处理数据，从数据集中移除了某些试剂和溶剂（例如盐酸、乙酸乙酯和二氯甲烷）。随后形成每个反应的反应物和产物之间具有最高Tanimoto相似性的分子对，表明从反应物到产物的结构转化。每个数据点的标记反应模板的索引被附加到反应物SMILES字符串的开头。TRACER的源代码、活性预测模型和策划的数据集可在我们的公共存储库https://github.com/sekijima-lab/TRACER 中找到。

### QSAR建模数据集

针对DRD2的QSAR建模数据集从ExCAPE$^{85}$和ChEMBL$^{86}$中提取。除了源自ExCAPE的活性/非活性标记数据集外，还添加了来自ChEMBL的一组化合物作为活性化合物。根据文献$^{87}$，从ChEMBL检索的活性化合物经过以下参数过滤：标准关系等于“=”，pChEMBL >= 6.0，分子量 < 750。从DUD-E$^{88}$获取的AKT1和CXCR4的活性/非活性标记数据集与来自ChEMBL的活性化合物合并。从ChEMBL获得的活性化合物通过与应用于DRD2相同的过滤标准进行预处理。最终的DRD2数据集包括9913个活性化合物和340,400个非活性化合物，AKT1数据集包括4035个活性化合物和16,450个非活性化合物，CXCR4数据集包括925个活性化合物和3406个非活性化合物。

### 从未见分子开始进行结构优化的数据集和对接模拟

为了证明我们的框架在从未见分子开始生成优化化合物的能力，创建了一个未包含在训练集中的分子数据集。该数据集是通过从ZINC数据库$^{89,90}$中可用的构建块中移除用于训练Transformer和QSAR模型的化合物构建的。应用第4.5节描述的相同过滤过程后，使用Vina-GPU 2.0$^{91}$对DRD2（PDB ID：6CM4）、AKT1（PDB ID：3CQW）和CXCR4（PDB ID：3ODU）进行对接模拟。然后为每个蛋白质选择具有最高配体效率（绝对对接分数除以重原子数）的化合物作为起始材料。

## 数据可用性

本研究中使用的活性预测模型和策划的数据集可在我们的公共存储库https://github.com/sekijima-lab/TRACER 中找到。

## 代码可用性

TRACER的源代码和训练模型已在MIT许可证下发布，可在我们的公共存储库https://github.com/sekijima-lab/TRACER 中找到。

## 收到日期：2024年8月26日；接受日期：2025年1月28日；
## 参考文献
1.  DiMasi, J. A., Grabowski, H. G. & Hansen, R. W. Innovation in the pharmaceutical industry: new estimates of r&d costs. J. Health Econ. 47, 20–33 (2016).
2.  Wouters, O. J., McKee, M. & Luyten, J. Estimated research and development investment needed to bring a new medicine to market, 2009-2018. JAMA 323, 844 (2020).
3.  Bleicher, K. H., Böhm, H.-J., Müller, K. & Alanine, A. I. Hit and lead generation: beyond high-throughput screening. Nat. Rev. Drug Discov. 2, 369–378 (2003).
4.  Jiménez-Luna, J., Grisoni, F. & Schneider, G. Drug discovery with explainable artificial intelligence. Nat. Mach. Intell. 2, 573–584 (2020).
5.  Yang, X., Zhang, J., Yoshizoe, K., Terayama, K. & Tsuda, K. Chemts: an efficient python library for de novo molecular generation. Sci. Technol. Adv. Mater. 18, 972–976 (2017).
6.  Erikawa, D., Yasuo, N. & Sekijima, M. Mermaid: an open source automated hit-to-lead method based on deep reinforcement learning. J. Cheminform. https://doi.org/10.1186/s13321-021-00572-6 (2021).
7.  Erikawa, D., Yasuo, N., Suzuki, T., Nakamura, S. & Sekijima, M. Gargoyles: An open source graph-based molecular optimization method based on deep reinforcement learning. ACS Omega 8, 37431–37441 (2023).
8.  Loeffler, H. H. et al. Reinvent 4: Modern ai–driven generative molecule design. J. Cheminformatics 16, 20 (2024).
9.  Gómez-Bombarelli, R. et al. Automatic chemical design using a data-driven continuous representation of molecules. ACS Cent. Sci. 4, 268–276 (2018).
10. Lim, J., Ryu, S., Kim, J. W. & Kim, W. Y. Molecular generative model based on conditional variational autoencoder for de novo molecular design. J. Cheminform. https://doi.org/10.1186/s13321-018-0286-7 (2018).
11. Sanchez-Lengeling, B. & Aspuru-Guzik, A. Inverse molecular design using machine learning: Generative models for matter engineering. Science 361, 360–365 (2018).
12. Prykhodko, O. et al. A de novo molecular generation method using latent vector based generative adversarial network. J. Cheminform. https://doi.org/10.1186/s13321-019-0397-9 (2019).
13. Ozawa, M., Nakamura, S., Yasuo, N. & Sekijima, M. Iev2mol: Molecular generative model considering protein–ligand interaction energy vectors. J. Chem. Inf. Modeling 64, 6969–6978 (2024).
14. Chiba, S. et al. A prospective compound screening contest identified broader inhibitors for sirtuin 1. Sci. Rep. https://doi.org/10.1038/s41598-019-55069-y (2019).
15. Yoshino, R. et al. Discovery of a hidden trypanosoma cruzi spermidine synthase binding site and inhibitors through in silico, in vitro, and X-ray crystallography. ACS Omega (2023).
16. Schneider, G. & Clark, D. E. Automated de novo drug design: are we nearly there yet? Angew. Chem. Int. Ed. 58, 10792–10803 (2019).
17. Ertl, P. & Schuffenhauer, A. Estimation of synthetic accessibility score of drug-like molecules based on molecular complexity and fragment contributions. J. Cheminformatics 1, 1–11 (2009).
18. Chen, S. & Jung, Y. Estimating the synthetic accessibility of molecules with building block and reaction-aware sascore. J. Cheminformatics 16, 83 (2024).
19. Stanley, M. & Segler, M. Fake it until you make it? generative de novo design and virtual screening of synthesizable molecules. Curr. Opin. Struct. Biol. 82, 102658 (2023).
20. de Almeida, A. F., Moreira, R. & Rodrigues, T. Synthetic organic chemistry driven by artificial intelligence. Nat. Rev. Chem. 3, 589–604 (2019).
21. Strieth-Kalthoff, F. et al. Artificial intelligence for retrosynthetic planning needs both data and expert knowledge. J. Am. Chem. Soc. https://doi.org/10.1021/jacs.4c00338 (2024).
22. Coley, C. W., Rogers, L., Green, W. H. & Jensen, K. F. Scscore: synthetic complexity learned from a reaction corpus. J. Chem. Inf. Model. 58, 252–261 (2018).
23. Voršilák, M., Kolár^, M., Čmelo, I. & Svozil, D. Syba: Bayesian estimation of synthetic accessibility of organic compounds. J. Cheminformatics 12, 1–13 (2020).
24. Thakkar, A., Chadimová, V., Bjerrum, E. J., Engkvist, O. & Reymond, J.-L. Retrosynthetic accessibility score (rascore)–rapid machine learned synthesizability classification from ai driven retrosynthetic planning. Chem. Sci. 12, 3339–3349 (2021).
25. Yu, J. et al. Organic compound synthetic accessibility prediction based on the graph attention mechanism. J. Chem. Inf. Model. 62, 2973–2986 (2022).
26. Wang, S., Wang, L., Li, F. & Bai, F. Deepsa: a deep-learning driven predictor of compound synthesis accessibility. J. Cheminformatics 15, 103 (2023).
27. Skoraczyński, G., Kitlas, M., Miasojedow, B. & Gambin, A. Critical assessment of synthetic accessibility scores in computer-assisted synthesis planning. J. Cheminformatics 15, 6 (2023).
28. Klingler, F.-M. et al. SAR by space: Enriching hit sets from the chemical space. Molecules 24, 3096 (2019).
29. Grygorenko, O. O. et al. Generating multibillion chemical space of readily accessible screening compounds. iScience 23, 101681 (2020).
30. Nicolaou, C. A., Watson, I. A., Hu, H. & Wang, J. The proximal lilly collection: Mapping, exploring and exploiting feasible chemical space. J. Chem. Inf. Model. 56, 1253–1266 (2016).
31. Hu, Q. et al. Pfizer global virtual library (pgvl): a chemistry design tool powered by experimentally validated parallel synthesis information. ACS Combinat. Sci. 14, 579–589 (2012).
32. Vinkers, H. M. et al. Synopsis: synthesize and optimize system in silico. J. Med. Chem. 46, 2765–2773 (2003).
33. Hartenfeller, M. et al. Dogs: reaction-driven de novo design of bioactive compounds. PLoS Comput. Biol. 8, e1002380 (2012).
34. Spiegel, J. O. & Durrant, J. D. Autogrow4: an open-source genetic algorithm for de novo drug design and lead optimization. J. Cheminformatics 12, 1–16 (2020).
35. Button, A., Merk, D., Hiss, J. A. & Schneider, G. Automated de novo molecular design by hybrid machine intelligence and rule-driven chemical synthesis. Nat. Mach. Intell. 1, 307–315 (2019).
36. Gao, W., Mercado, R. & Coley, C. W. Amortized tree generation for bottom-up synthesis planning and synthesizable molecular design. In International Conference on Learning Representations https://openreview.net/forum?id=FRxhHdnxt1 (2022).
37. Nguyen, D. H. & Tsuda, K. Generating reaction trees with cascaded variational autoencoders. J. Chem. Phys. 156, 044117 (2022).
38. Noh, J. et al. Path-Aware and Structure-Preserving generation of synthetically accessible molecules. In Chaudhuri, K. et al. (eds.) Proceedings of the 39th International Conference on Machine Learning, vol. 162 of Proceedings of Machine Learning Research, 16952-16968 (PMLR, 2022).
39. Gottipati, S. K. et al. Learning to navigate the synthetically accessible chemical space using reinforcement learning. In International conference on machine learning, 3668-3679 (PMLR, 2020).
40. Horwood, J. & Noutahi, E.Molecular design in synthetically accessible chemical space via deep reinforcement learning. ACS Omega 5, 32984–32994 (2020).
41. Fialková, V. et al. Libinvent: Reaction-based generative scaffold decoration for in silico library design. J. Chem. Inf. Modeling 62, 2046–2063 (2021).
42. Swanson, K. et al. Generative ai for designing and validating easily synthesizable and structurally novel antibiotics. Nat. Mach. Intell. 6, 338–353 (2024).
43. Cretu, M., Harris, C., Roy, J., Bengio, E. & Liò, P. Synflownet: Towards molecule design with guaranteed synthesis pathways. In ICLR 2024 Workshop on Generative and Experimental Perspectives for Biomolecular Design (2024).
44. Wang, Y., Zhang, H., Zhu, J., Li, Y. & Feng, L. Rgfn: Recurrent graph feature network for clickbait detection. In 2021 International Conference on High Performance Big Data and Intelligent Systems (HPBD&IS), 151-156 (IEEE, 2021).
45. Bradshaw, J., Paige, B., Kusner, M. J., Segler, M. & Hernández-Lobato, J. M. A model to search for synthesizable molecules. Adv. Neural Inf. Process. Syst. 32, 12000–12010 (2019).
46. Bradshaw, J., Paige, B., Kusner, M. J., Segler, M. & Hernández-Lobato, J. M. Barking up the right tree: an approach to search over molecule synthesis dags. Adv. neural Inf. Process. Syst. 33, 6852–6866 (2020).
47. Schwaller, P. et al. Molecular transformer: A model for uncertainty-calibrated chemical reaction prediction. ACS Cent. Sci. 5, 1572–1583 (2019).
48. Vaswani, A. et al. Attention is all you need. Adv. Neural Inf. Process. Syst. 30, 6000–6010 (2017).
49. Weininger, D. Smiles, a chemical language and information system. 1. introduction to methodology and encoding rules. J. Chem. Inf. Computer Sci. 28, 31–36 (1988).
50. Brown, T. et al. Language models are few-shot learners. Adv. Neural Inf. Process. Syst. 33, 1877–1901 (2020).
51. Yoshikai, Y., Mizuno, T., Nemoto, S. & Kusuhara, H. Difficulty in chirality recognition for transformer architectures learning chemical structures from string representations. Nat. Commun. 15, 1197 (2024).
52. Reaxys. https://www.reaxys.com.
53. Yi, X. et al. Substituted Oxaborole Antibacterial Agents. WO Patent 2011/17125 A1 (2011).
54. Sartori, G. & Maggi, R. Use of solid catalysts in friedel- crafts acylation reactions. Chem. Rev. 106, 1077–1104 (2006).
55. Quinn, J. F., Bryant, C. E., Golden, K. C. & Gregg, B. T. Rapid reduction of heteroaromatic nitro groups using catalytic transfer hydrogenation with microwave heating. Tetrahedron Lett. 51, 786–789 (2010).
56. Kianmehr, E. & Afaridoun, H. Nickel (ii)-and silver (i)-catalyzed c–h bond halogenation of anilides and carbamates. synthesis 53, 1513–1523 (2021).
57. Singh, H., Sen, C., Sahoo, T. & Ghosh, S. C. A visible light-mediated regioselective halogenation of anilides and quinolines by using a heterogeneous cu-mno catalyst. Eur. J. Org. Chem. 2018, 4748–4753 (2018).
58. Legnani, L. & Morandi, B. Direct catalytic synthesis of unprotected 2-amino-1-phenylethanols from alkenes by using iron (ii) phthalocyanine. Angew. Chem. Int. Ed. 55, 2248–2251 (2016).
59. Cantello, B. C. et al. [[. omega.-(heterocyclylamino) alkoxy] benzyl]-2, 4-thiazolidinediones as potent antihyperglycemic agents. J. Med. Chem. 37, 3977–3985 (1994).
60. Fan, Y.-H. et al. Structure–activity requirements for the antiproliferative effect of troglitazone derivatives mediated by depletion of intracellular calcium. Bioorg. Med. Chem. Lett. 14, 2547–2550 (2004).
61. Bemis, G. W. & Murcko, M. A. The properties of known drugs. 1. molecular frameworks. J. Med. Chem. 39, 2887–2893 (1996).
62. Chen, B., Li, C., Dai, H. & Song, L. Retro*: learning retrosynthetic planning with neural guided a* search. In International conference on machine learning, 1608–1616 (PMLR, 2020).
63. Enamine building blocks. https://enamine.net/building-blocks/building-blocks-catalog. Accessed on 2024/11/05.
64. Preuer, K., Renz, P., Unterthiner, T., Hochreiter, S. & Klambauer, G. Fréchet chemnet distance: a metric for generative models for molecules in drug discovery. J. Chem. Inf. Model. 58, 1736–1741 (2018).
65. Gao, H. et al. Using machine learning to predict suitable conditions for organic reactions. ACS Cent. Sci. 4, 1465–1476 (2018).
66. Shim, E. et al. Predicting reaction conditions from limited data through active transfer learning. Chem. Sci. 13, 6655–6668 (2022).
67. Fu, T., Xiao, C., Li, X., Glass, L. M. & Sun, J. Mimosa: Multi-constraint molecule sampling for molecule optimization. In Proceedings of the AAAI Conference on Artificial Intelligence, 35, 125–133 (2021).
68. Pang, C., Qiao, J., Zeng, X., Zou, Q. & Wei, L. Deep generative models in de novo drug molecule generation. J. Chem. Inf. Modeling 64, 2174–2194 (2023).
69. Yang, X. et al. Gpmo: Gradient perturbation-based contrastive learning for molecule optimization. In IJCAI, 4940-4948 (2023).
70. Yang, Y. et al. SyntaLinker: automatic fragment linking with deep conditional transformer neural networks. Chem. Sci. 11, 8312–8322 (2020).
71. Wang, J. et al. Multi-constraint molecular generation based on conditional transformer, knowledge distillation and reinforcement learning. Nat. Mach. Intell. 3, 914–922 (2021).
72. He, J. et al. Transformer-based molecular optimization beyond matched molecular pairs. J. Cheminformatics. https://doi.org/10.1186/s13321-022-00599-3 (2022).
73. Schwaller, P. et al. Mapping the space of chemical reactions using attention-based neural networks. Nat. Mach. Intell. 3, 144–152 (2021).
74. Paszke, A. et al. Pytorch: An imperative style, high-performance deep learning library. Adv. Neural Infor. Process. Syst. 32, 8024–8035 (2019).
75. Kipf, T. N. & Welling, M. Semi-Supervised Classification with Graph Convolutional Networks. In Proceedings of the 5th International Conference on Learning Representations, ICLR ’17 https://openreview.net/forum?id=SJU4ayYgl (2017).
76. Ishida, S., Terayama, K., Kojima, R., Takasu, K. & Okuno, Y. Prediction and interpretable visualization of retrosynthetic reactions using graph convolutional networks. J. Chem. Inf. modeling 59, 5026–5033 (2019).
77. Akiba, T., Sano, S., Yanase, T., Ohta, T. & Koyama, M. Optuna: A next-generation hyperparameter optimization framework. In Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining, 2623-2631 (2019).
78. Browne, C. B. et al. A survey of monte carlo tree search methods. IEEE Trans. Comput. Intell. AI games 4, 1–43 (2012).
79. Jensen, J. H. A graph-based genetic algorithm and generative model/monte carlo tree search for the exploration of chemical space. Chem. Sci. 10, 3567–3572 (2019).
80. Sun, M. et al. Molsearch: search-based multi-objective molecular generation and property optimization. InProceedings of the 28th ACM SIGKDD conference on knowledge discovery and data mining, 4724-4732 (2022).
81. Suzuki, T., Ma, D., Yasuo, N. & Sekijima, M. Mothra: Multiobjective de novo molecular generation using monte carlo tree search. J. Chem. Inf. Model. 64, 7291–7302 (2024).
82. Polykovskiy, D. et al. Molecular sets (MOSES): A benchmarking platform for molecular generation models. Front. Pharmacol. 11, 565644 (2020).
83. Lowe, D. Chemical reactions from u.s. patents (1976-sep2016), 2017.https://figshare.com/articles/dataset/Chemical_reactions_from_US_patents_1976-Sep2016_5104873.
84. Schwaller, P., Hoover, B., Reymond, J.-L., Strobelt, H. & Laino, T. Extraction of organic chemistry grammar from unsupervised learning of chemical reactions. Sci. Adv. 7, eabe4166 (2021).
85. Sun, J. et al. Excape-db: an integrated large scale dataset facilitating big data analysis in chemogenomics. J. Cheminformatics 9, 1–9 (2017).
86. Zdrazil, B. et al. The chembl database in 2023: a drug discovery platform spanning multiple bioactivity data types and time periods. Nucleic acids Res. 52, D1180–D1192 (2024).
87. Papadopoulos, K., Giblin, K. A., Janet, J. P., Patronov, A. & Engkvist, O. De novo design with deep generative models based on 3d similarity scoring. Bioorg. Medicinal Chem. 44, 116308 (2021).
88. Mysinger, M. M., Carchia, M., Irwin, J. J. & Shoichet, B. K. Directory of useful decoys, enhanced (dud-e): better ligands and decoys for better benchmarking. J. Medicinal Chem. 55, 6582–6594 (2012).
89. Irwin, J. J. et al. Zinc20-a free ultralarge-scale chemical database for ligand discovery. J. Chem. Inf. Model. 60, 6065–6073 (2020).
90. Zinc database. https://files.docking.org/bb/.
91. Ding, J. et al. Vina-gpu 2.0: further accelerating autodock vina and its derivatives with graphics processing units. J. Chem. Inf. Model. 63, 1982–1998 (2023).

## 致谢

这项工作部分得到了日本文部科学省 transformative Research Areas (A) "Latent Chemical Space" JP24H01760 对 M.S. 的资助；来自 AMED 的生命科学和药物发现研究支持项目（支持创新药物发现和生命科学研究的基础（BINDS）） under Grant Number JP24ama121026 对 M.S. 的资助；日本学术振兴会（JSPS） under KAKENHI Grant Number JP20H00620 对 M.S. 的资助；以及 JST SPRING Grant Number JPMJSP2106 对 S.N. 的资助。我们感谢 Springer Nature Author Services 提供的专业英语语言编辑服务（验证码：9754-750B-ED41-C1BD-E29P）。

## 作者贡献

S.N. 设计了研究，M.S. 监督了计算研究。S.N. 开发了软件，N.Y. 指导了实验。S.N. 撰写了手稿。所有作者都批判性地审阅和修订了手稿草案，并批准了最终提交版本。

## 利益竞争

作者声明没有竞争利益。

## 附加信息

**补充信息** 在线版本包含补充材料，可在 https://doi.org/10.1038/s42004-025-01437-x 获取。

**通讯和材料请求** 应 addressed to Masakazu Sekijima。

**同行评议信息** Communications Chemistry感谢Fang Bai和其他匿名审稿人对本工作的同行评议所做的贡献。

**再版和许可信息** 可在 http://www.nature.com/reprints 获取。

**出版商注** Springer Nature对出版地图中的管辖权主张保持中立。

© The Author(s) 2025

本文根据 Creative Commons Attribution 4.0 International License 许可，该许可允许在任何媒介或格式中使用、共享、改编、分发和复制，只要您适当地注明原作者和来源，提供指向 Creative Commons 许可证的链接，并指明是否进行了更改。本文中的图像或其他第三方材料包含在文章的 Creative Commons 许可证中，除非在材料的信用额度中另有说明。如果材料未包含在文章的 Creative Commons 许可证中，并且您的预期用途不受法律法规的允许或超出了允许的用途，您将需要直接获得版权持有人的许可。要查看此许可证的副本，请访问 http://creativecommons.org/licenses/by/4.0/。