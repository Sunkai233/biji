# transformer代码实例1

### {

### 自定义 Transformer Block 示例

```python
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        # 多头注意力
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        # 前馈层
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        # 层归一化
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        # dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [seq_len, batch_size, embed_dim]
        attn_output, _ = self.attention(x, x, x)  
        x = self.norm1(x + self.dropout(attn_output))  # 残差连接 + norm
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))   # 残差连接 + norm
        return x
```

👉 这样你就有了一个基本的 **Transformer Encoder Block**。
 如果要堆叠多个，可以用 `nn.ModuleList` 包一层。

### }

### {

## 文本 Token 示例

```python
import torch
import torch.nn as nn

# 假设词表大小为 10000，每个 token 映射到 512 维向量空间
vocab_size = 10000
embed_dim = 512

# nn.Embedding 就是一个查表层，把索引转成向量
embedding = nn.Embedding(vocab_size, embed_dim)

# 模拟一个句子：假设分词后对应的 token id 是 [1, 5, 9, 7]
# 这里加了 batch 维度，所以 shape = [batch=1, seq_len=4]
input_ids = torch.tensor([[1, 5, 9, 7]])  

# 输入 embedding，得到 token embeddings
# shape: [batch, seq_len, embed_dim] = [1, 4, 512]
token_embeddings = embedding(input_ids)
print(token_embeddings.shape)  # torch.Size([1, 4, 512])
```

------

## 图像 Token 示例（ViT）

```python
# 假设输入图像 shape: [B, C, H, W]
B, C, H, W = 1, 3, 32, 32
image = torch.randn(B, C, H, W)  # 随机一张 32x32 彩色图

patch_size = 16
embed_dim = 512

# unfold(2, 16, 16): 在 H 维度上滑窗切 patch
# unfold(3, 16, 16): 在 W 维度上滑窗切 patch
# 输出 shape: [B, C, H/16, W/16, 16, 16]
patches = image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)

# 把 patch 展平成向量
# flatten(2)：从第 2 维开始展平，得到每个 patch 的像素向量
# shape: [B, C, num_patches_h, num_patches_w, patch_size*patch_size]
patch_tokens = patches.flatten(2)

# 一个 patch 的维度是 C * patch_size^2
patch_dim = C * patch_size * patch_size

# 用线性层把 patch 向量映射到 embedding 空间
linear_proj = nn.Linear(patch_dim, embed_dim)

# 注意要 reshape 成 [B, num_patches, patch_dim] 再投影
tokens = linear_proj(patch_tokens.view(B, -1, patch_dim))
print(tokens.shape)  # torch.Size([1, 4, 512]) (因为 32/16=2, 2*2=4 个 patch)
```

------

## 时间序列 Token 示例

```python
# 假设时间序列：batch=2, 长度=100, 特征维度=1
time_series = torch.randn(2, 100, 1)

# 我们把序列切成窗口，每个窗口长度=10
window_size = 10
num_windows = time_series.shape[1] // window_size  # = 10

# reshape: [batch, num_windows, window_size*feature_dim]
tokens = time_series.view(2, num_windows, -1)  # 每个窗口是一个 token
print(tokens.shape)  # torch.Size([2, 10, 10])

# 投影到 embedding 空间
embed_dim = 64
proj = nn.Linear(window_size, embed_dim)
tokens = proj(tokens)
print(tokens.shape)  # torch.Size([2, 10, 64])
```

------

## 位置编码 (Positional Encoding)

```python
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        
        # 初始化一个位置编码矩阵 [max_len, embed_dim]
        pe = torch.zeros(max_len, embed_dim)
        
        # position: 每个位置的索引，从 0 到 max_len-1
        # shape: [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # div_term: 控制正弦/余弦函数的频率
        # 用公式 10000^(2i/d_model)，这里取 log 再取 exp
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * 
            (-torch.log(torch.tensor(10000.0)) / embed_dim)
        )
        
        # 偶数位置用 sin
        pe[:, 0::2] = torch.sin(position * div_term)
        # 奇数位置用 cos
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 增加 batch 维度 [max_len, 1, embed_dim]
        self.pe = pe.unsqueeze(1)
        
    def forward(self, x):
        # x shape: [seq_len, batch, embed_dim]
        # 返回：输入加上位置编码
        return x + self.pe[:x.size(0)]

# 测试
embed_dim = 512
pos_encoding = PositionalEncoding(embed_dim)
x = torch.zeros(10, 2, embed_dim)  # seq_len=10, batch=2
out = pos_encoding(x)
print(out.shape)  # torch.Size([10, 2, 512])
```

------

✅ 总结：

1. **文本 token**：用 `nn.Embedding`。
2. **图像 token**：切 patch → flatten → `nn.Linear`。
3. **时间序列 token**：切窗口 → flatten → `nn.Linear`。
4. **位置编码**：用 `sin/cos` 给序列加上顺序信息。

### }

### {

### PhysicsSolver 中的 **P-Attention** 改造了**Attention**机制，使其更适合 PDE 场景： 

1. **输入不同**    - 普通 Attention 输入是自然语言序列或时间序列。   - P-Attention 输入的是 **物理输入（$t,x,v$）+ 数据输入** 的混合嵌入。

2. **归一化方式不同**    - 标准 Attention 的分母是 $\sqrt{d_k}$。   - P-Attention 用的是 $||QK^T||_{l_2}$（L2 范数归一化）。   - 这样可以更稳定地处理 **高维连续时空数据**，避免数值过大或过小。     

   在python使用torch框架下，是怎么实现的



#### 核心思路

1. **输入处理**

   - 假设我们已经有了 `phys_embed`（物理输入 embedding，例如 $[t, x, v]$ 经过 MLP）
   - 和 `data_embed`（观测/状态数据 embedding）。
   - 将二者拼接后得到 `input_embed`。

2. **Q, K, V 生成**

   - 用 `nn.Linear` 生成 Q, K, V 矩阵。

3. **P-Attention 核心**

   - 普通 Attention:

     ```python
     attn_scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)
     attn_weights = softmax(attn_scores, dim=-1)
     ```

   - P-Attention 改造:

     ```python
     raw_scores = Q @ K.transpose(-2, -1)
     norm = torch.norm(raw_scores, p=2, dim=-1, keepdim=True) + 1e-8
     attn_scores = raw_scores / norm
     attn_weights = F.softmax(attn_scores, dim=-1)
     ```

------

#### PyTorch 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, "embed_dim 必须能整除 num_heads"
        self.head_dim = embed_dim // num_heads
        
        # Q, K, V projection
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, phys_embed, data_embed):
        """
        phys_embed: [B, N, Dp]  物理输入 (t,x,v)
        data_embed: [B, N, Dd]  数据输入
        """
        # 拼接输入
        x = torch.cat([phys_embed, data_embed], dim=-1)  # [B, N, Dp+Dd]
        
        B, N, _ = x.shape
        
        # 线性变换
        Q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, d]
        K = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, d]
        V = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, d]
        
        # 计算注意力分数 (未归一化)
        raw_scores = torch.matmul(Q, K.transpose(-2, -1))  # [B, H, N, N]
        
        # 使用 L2 范数归一化 (代替 sqrt(d_k))
        norm = torch.norm(raw_scores, p=2, dim=-1, keepdim=True) + 1e-8
        attn_scores = raw_scores / norm
        
        # softmax 得到权重
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, H, N, N]
        
        # 加权求和
        context = torch.matmul(attn_weights, V)  # [B, H, N, d]
        
        # 拼回
        context = context.transpose(1, 2).contiguous().view(B, N, self.embed_dim)
        
        return self.out_proj(context)
```

------

#### 使用示例

```python
B, N, Dp, Dd = 4, 16, 3, 8  # batch=4, 序列长=16, 物理维度=3, 数据维度=8
phys_embed = torch.randn(B, N, Dp)  # (t, x, v)
data_embed = torch.randn(B, N, Dd)

p_attn = PAttention(embed_dim=Dp+Dd, num_heads=2)
output = p_attn(phys_embed, data_embed)

print(output.shape)  # [4, 16, 11] (Dp+Dd)
```

### }

### {

### 怎么在 PyTorch 里把 Transformer 融入 iLQR 框架

#### 实现思路

1. **iLQR 原始流程**
   - Forward pass：给定输入序列 $u_{0:T-1}$，模拟系统动力学，得到 $x_{0:T}$。
   - Backward pass：递推计算反馈/前馈增益 $K_t, k_t$。这是最耗时的部分。
2. **改造后的流程**
   - **部分反向过程**：只计算后半段的 $k_{i:T-1}, K_{i:T-1}$。
   - **Transformer 补全**：输入 $(X, k_{i:T-1}, K_{i:T-1})$，预测 $\hat{k}*{0:i-1}, \hat{K}*{0:i-1}$。
   - 拼接得到完整的 $k_{0:T-1}, K_{0:T-1}$，用于 forward control。
3. **Transformer 模块设计**
   - 仅解码器（causal mask），保持因果性。
   - 输入：
     - 状态 $X = (x_0, \ldots, x_T)$
     - 已知增益 $k_{i:T-1}, K_{i:T-1}$
   - 嵌入后加位置编码，输入多头注意力。
   - 输出 reshape 回 $(\hat{k}, \hat{K})$。

### 🚀 PyTorch 实现：iLQR-Transformer 增益预测模块

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

###############################################
# 位置编码模块：让 Transformer 知道时间顺序
###############################################
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        d_model: 特征维度
        max_len: 序列最大长度
        """
        super().__init__()
        
        # 创建 [max_len, d_model] 的位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * 
                             (-torch.log(torch.tensor(10000.0)) / d_model))
        
        # 偶数位置用 sin，奇数位置用 cos
        pe[:, 0::2] = torch.sin(position * div_term)  
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 扩展 batch 维度，便于直接加到输入上
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        # 注册为 buffer，模型保存时一并保存，但不会作为参数更新
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [B, T, d_model] 输入序列
        返回: 加上位置编码后的输入
        """
        return x + self.pe[:, :x.size(1)]


###############################################
# iLQR Transformer：预测缺失的增益矩阵 (k, K)
###############################################
class iLQRTransformer(nn.Module):
    def __init__(self, state_dim, k_dim, K_dim, d_model=128, nhead=4, num_layers=3):
        """
        state_dim: 状态维度 (x_t 大小)
        k_dim: 前馈增益向量的维度
        K_dim: 反馈增益矩阵的展平维度
        d_model: Transformer 特征维度
        nhead: 注意力头数
        num_layers: Transformer 解码器层数
        """
        super().__init__()
        
        # 状态嵌入层，把原始状态映射到 d_model 维度
        self.state_embed = nn.Linear(state_dim, d_model)
        
        # 增益嵌入层，把 k, K 映射到 d_model 维度
        self.k_embed = nn.Linear(k_dim, d_model)
        self.K_embed = nn.Linear(K_dim, d_model)
        
        # 用于拼接后的增益映射到 d_model
        self.gain_proj = nn.Linear(2 * d_model, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer 解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,    # 特征维度
            nhead=nhead,        # 多头注意力头数
            batch_first=True    # [B, T, d_model] 格式
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # 输出层：把 Transformer 输出转回增益矩阵
        self.out_k = nn.Linear(d_model, k_dim)
        self.out_K = nn.Linear(d_model, K_dim)

    def forward(self, X, k_partial, K_partial):
        """
        X: [B, T, state_dim]    系统状态序列 (全时域)
        k_partial: [B, T-i, k_dim]   已知一部分的前馈增益
        K_partial: [B, T-i, K_dim]   已知一部分的反馈增益
        返回: 预测的 (k, K)，补全未计算部分
        """
        B, T, _ = X.shape
        
        # === Step1. 状态嵌入 ===
        x_embed = self.state_embed(X)   # [B, T, d_model]
        
        # === Step2. 部分增益嵌入 ===
        k_embed = self.k_embed(k_partial)   # [B, T-i, d_model]
        K_embed = self.K_embed(K_partial)   # [B, T-i, d_model]
        
        # 拼接 (k, K)，并投影回 d_model
        gain_embed = torch.cat([k_embed, K_embed], dim=-1)  # [B, T-i, 2*d_model]
        gain_embed = self.gain_proj(gain_embed)             # [B, T-i, d_model]
        
        # === Step3. 位置编码 ===
        memory = self.pos_encoder(x_embed)   # 状态作为 memory
        tgt = self.pos_encoder(gain_embed)   # 增益作为目标
        
        # === Step4. 因果 mask，保证解码器只能看过去 ===
        tgt_len = tgt.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(tgt.device)
        
        # === Step5. Transformer 解码器 ===
        out = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)  # [B, T-i, d_model]
        
        # === Step6. 输出预测的增益 ===
        pred_k = self.out_k(out)  # [B, T-i, k_dim]
        pred_K = self.out_K(out)  # [B, T-i, K_dim]
        
        return pred_k, pred_K


###############################################
# 🔍 使用示例
###############################################
if __name__ == "__main__":
    B, T, state_dim = 8, 20, 6   # batch=8, 时间长度=20, 状态维度=6
    k_dim, K_dim = 2, 4          # 前馈增益=2维，反馈增益=4维 (展平)
    
    # 模拟输入
    X = torch.randn(B, T, state_dim)       # 系统状态序列
    k_partial = torch.randn(B, T//2, k_dim) # 已知一半增益
    K_partial = torch.randn(B, T//2, K_dim)
    
    # 定义模型
    model = iLQRTransformer(state_dim, k_dim, K_dim)
    
    # 前向预测
    pred_k, pred_K = model(X, k_partial, K_partial)
    
    print("预测前馈增益 pred_k 形状:", pred_k.shape)  # [8, 10, 2]
    print("预测反馈增益 pred_K 形状:", pred_K.shape)  # [8, 10, 4]
```

------

#### 🔑 代码要点

- **部分反向过程**：`k_partial, K_partial` 就是你实际算出来的后半段增益。
- **Transformer 预测**：`pred_k, pred_K` 是模型预测的前半段增益。
- **最终拼接**：把 `k_partial + pred_k` 拼接起来，就得到了完整的 $k_{0:T-1}, K_{0:T-1}$。

### }

### {

###  **Rel-CNN (关系卷积神经网络)** 的 **PyTorch 实现**

### 🔑 解释

1. **RFC** → 直接用差值/绝对值/二值化生成关系特征。
2. **LRFC** → 学习型关系特征，用 Q/K 投影后计算相似度。
3. **层次卷积** → 先降维，再多尺度卷积（避免参数爆炸）。
4. **LPGR Block** → 每个块提取 “局部模式 + 全局关系”，带残差。
5. **Rel-CNN** → 多个 LPGR Block 堆叠，最后用全连接分类。

### 🚀 PyTorch 实现：Rel-CNN

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

######################################################
# 1. 关系特征计算模块
######################################################
class RFC(nn.Module):
    """ 
    Relationship Feature based Convolution Filtering (固定规则关系特征)
    输入: [B, T, D]，B=批大小，T=时间长度，D=特征维度
    输出: [B, T, T, D]，关系矩阵
    """
    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.size()
        
        # 扩展维度便于做 pairwise 操作
        x1 = x.unsqueeze(2)  # [B, T, 1, D]
        x2 = x.unsqueeze(1)  # [B, 1, T, D]
        
        # 差值关系 p - q
        diff = x1 - x2       # [B, T, T, D]
        # 绝对差 |p - q|
        abs_diff = torch.abs(diff)
        # 二值化关系 (差值是否小于阈值，这里用均值近似代替)
        binary = (abs_diff < abs_diff.mean()).float()
        
        # 拼接所有关系特征
        relation = torch.cat([diff, abs_diff, binary], dim=-1)  # [B, T, T, 3D]
        return relation


class LRFC(nn.Module):
    """ 
    Latent Relationship Feature Convolution (学习关系特征)
    输入: [B, T, D]
    输出: [B, T, T] (相似度矩阵, 每个特征维度会做成一个通道)
    """
    def __init__(self, D, d_latent=32):
        super().__init__()
        # 学习两个投影矩阵 Q, K
        self.proj_q = nn.Linear(D, d_latent)
        self.proj_k = nn.Linear(D, d_latent)

    def forward(self, x):
        # x: [B, T, D]
        Q = self.proj_q(x)  # [B, T, d_latent]
        K = self.proj_k(x)  # [B, T, d_latent]

        # 相似度计算 (点积形式)
        sim = torch.matmul(Q, K.transpose(-2, -1))  # [B, T, T]
        # 归一化 (softmax)
        sim = F.softmax(sim, dim=-1)
        
        # 输出相似度关系特征
        return sim.unsqueeze(-1)  # [B, T, T, 1]


######################################################
# 2. 层次卷积模块
######################################################
class HierarchicalConv(nn.Module):
    """
    分层卷积：先共享卷积降维，再多尺度卷积提取模式
    输入: [B, C, T, T] (C=通道数, 包含原始特征+关系特征)
    输出: [B, C_out, T, T]
    """
    def __init__(self, in_channels, hidden_channels=32, out_channels=64, kernel_sizes=[2,4,8]):
        super().__init__()
        
        # Step1: 降维卷积 (共享权重卷积)
        self.reduce_conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        
        # Step2: 多尺度卷积 (不同 kernel size 提取不同尺度特征)
        self.multi_scale_convs = nn.ModuleList([
            nn.Conv2d(hidden_channels, out_channels, kernel_size=(k, k), padding="same")
            for k in kernel_sizes
        ])
    
    def forward(self, x):
        # 先降维
        x = self.reduce_conv(x)  # [B, hidden, T, T]
        
        # 多尺度卷积
        features = [conv(x) for conv in self.multi_scale_convs]  # list of [B, out, T, T]
        
        # 拼接所有尺度特征
        out = torch.cat(features, dim=1)  # [B, out_channels*len(kernel_sizes), T, T]
        return out


######################################################
# 3. Rel-CNN 核心模块: LPGR Block
######################################################
class LPGRBlock(nn.Module):
    """
    Local Pattern + Global Relationship Block
    """
    def __init__(self, D, use_lrfc=True):
        super().__init__()
        
        # 关系特征
        self.rfc = RFC()
        self.lrfc = LRFC(D) if use_lrfc else None
        
        # 层次卷积
        # 输入通道数 = 原始特征(D) + RFC关系特征(3D) + LRFC关系特征(1)
        in_channels = D + 3*D + (1 if use_lrfc else 0)
        self.hier_conv = HierarchicalConv(in_channels=in_channels)
        
        # 残差连接 (保证稳健)
        self.residual = nn.Conv2d(D, self.hier_conv.multi_scale_convs[0].out_channels * len(self.hier_conv.multi_scale_convs), kernel_size=1)

    def forward(self, x):
        """
        x: [B, T, D] 时间序列输入
        返回: [B, C, T, T] 特征图
        """
        B, T, D = x.size()
        
        # === Step1. 关系特征 ===
        rfc_feat = self.rfc(x)      # [B, T, T, 3D]
        if self.lrfc:
            lrfc_feat = self.lrfc(x)   # [B, T, T, 1]
            relation_feat = torch.cat([rfc_feat, lrfc_feat], dim=-1)  # [B, T, T, 3D+1]
        else:
            relation_feat = rfc_feat   # [B, T, T, 3D]
        
        # === Step2. 拼接原始输入 (广播到矩阵维度)
        x_matrix = x.unsqueeze(2).repeat(1, 1, T, 1)  # [B, T, T, D]
        feat = torch.cat([x_matrix, relation_feat], dim=-1)  # [B, T, T, D+3D(+1)]
        
        # === Step3. 转换为卷积输入格式 [B, C, T, T]
        feat = feat.permute(0, 3, 1, 2)  # [B, C, T, T]
        
        # === Step4. 层次卷积提取特征 ===
        out = self.hier_conv(feat)   # [B, C_out, T, T]
        
        # === Step5. 残差连接 ===
        res = self.residual(x_matrix.permute(0, 3, 1, 2))  # [B, C_out, T, T]
        out = out + res
        
        return out


######################################################
# 4. Rel-CNN 整体架构
######################################################
class RelCNN(nn.Module):
    def __init__(self, D, num_classes=5, num_blocks=3, use_lrfc=True):
        """
        D: 输入特征维度
        num_classes: 分类类别数
        num_blocks: LPGR 堆叠层数
        """
        super().__init__()
        
        # 堆叠多个 LPGR Block
        self.blocks = nn.ModuleList([LPGRBlock(D, use_lrfc=use_lrfc) for _ in range(num_blocks)])
        
        # 分类头
        self.fc = nn.Linear(64*3, num_classes)  # 假设 HierarchicalConv 输出通道数=64*3
        
    def forward(self, x):
        """
        x: [B, T, D] 输入时间序列
        返回: 分类结果 [B, num_classes]
        """
        out = None
        for block in self.blocks:
            out = block(x)   # [B, C, T, T]
        
        # 全局池化，把 [B, C, T, T] → [B, C]
        out = F.adaptive_avg_pool2d(out, (1,1)).squeeze(-1).squeeze(-1)
        
        # 分类层
        logits = self.fc(out)
        return logits


######################################################
# 🔍 使用示例
######################################################
if __name__ == "__main__":
    B, T, D = 8, 20, 3  # batch=8，序列长度=20，特征维度=3
    x = torch.randn(B, T, D)
    
    model = RelCNN(D, num_classes=5, num_blocks=2, use_lrfc=True)
    out = model(x)
    
    print("输入 x 形状:", x.shape)      # [8, 20, 3]
    print("输出 logits 形状:", out.shape)  # [8, 5]
```

### }



