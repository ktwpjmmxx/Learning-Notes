# GPT-2 Multi-Head Self-Attentionの詳細解説とコード実装

GPT-2の核心となるMulti-Head Self-Attention（MHSA）メカニズムについて、実装コードと共に詳細に解説します。

## 1. Multi-Head Self-Attentionの基本概念

### 1.1 Self-Attentionとは

Self-Attentionは、入力シーケンス内の各位置が他の位置とどの程度関連しているかを学習するメカニズムです。

```
"The cat sat on the mat"
 ↓ Self-Attention
各単語が他の単語との関連度を計算
- "cat" → "The"(0.1), "cat"(0.9), "sat"(0.7), "on"(0.2), "the"(0.1), "mat"(0.4)
```

### 1.2 Multi-Headの利点

- **複数の表現部分空間**: 異なる種類の関係性を並列して学習
- **情報の多様性**: 語順、意味、構文など異なる観点で attention を計算
- **表現力の向上**: 単一のattentionヘッドでは捉えきれない複雑な関係を学習

## 2. MHSAの構造とデータフロー

### 2.1 全体的なアーキテクチャ

```
Input: [batch_size, seq_len, hidden_size]
    ↓
Linear Projections (Q, K, V)
    ↓
Reshape & Split into Multiple Heads
    ↓
Scaled Dot-Product Attention (parallel)
    ↓
Concatenate Heads  
    ↓
Output Projection
    ↓
Output: [batch_size, seq_len, hidden_size]
```

### 2.2 PyTorchによる完全実装

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size=768, num_heads=12, dropout_prob=0.1):
        super().__init__()
        
        # 設定値の保存
        self.hidden_size = hidden_size  # 768 (GPT-2 small)
        self.num_heads = num_heads      # 12
        self.head_size = hidden_size // num_heads  # 64
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        # Linear projections for Q, K, V (combined for efficiency)
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_prob)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """重みの初期化"""
        nn.init.normal_(self.qkv_proj.weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.zeros_(self.qkv_proj.bias)
        nn.init.zeros_(self.out_proj.bias)
    
    def forward(self, x, causal_mask=True):
        """
        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]
            causal_mask: Whether to apply causal masking for language modeling
            
        Returns:
            output: Output tensor [batch_size, seq_len, hidden_size]
            attention_weights: Attention weights [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, hidden_size = x.size()
        
        # Step 1: Generate Q, K, V through linear projections
        qkv = self.qkv_proj(x)  # Shape: [batch_size, seq_len, 3 * hidden_size]
        
        # Split into Q, K, V
        q, k, v = qkv.chunk(3, dim=-1)  # Each: [batch_size, seq_len, hidden_size]
        
        print(f"After QKV projection:")
        print(f"  Q shape: {q.shape}")
        print(f"  K shape: {k.shape}")
        print(f"  V shape: {v.shape}")
        
        # Step 2: Reshape and split into multiple heads
        q = q.view(batch_size, seq_len, self.num_heads, self.head_size)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_size)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_size)
        
        # Transpose for attention computation: [batch_size, num_heads, seq_len, head_size]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)  
        v = v.transpose(1, 2)
        
        print(f"After head splitting:")
        print(f"  Q shape: {q.shape}")
        print(f"  K shape: {k.shape}")
        print(f"  V shape: {v.shape}")
        
        # Step 3: Scaled Dot-Product Attention
        attention_output, attention_weights = self.scaled_dot_product_attention(
            q, k, v, causal_mask
        )
        
        print(f"After attention:")
        print(f"  Attention output shape: {attention_output.shape}")
        print(f"  Attention weights shape: {attention_weights.shape}")
        
        # Step 4: Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous()
        # Shape: [batch_size, seq_len, num_heads, head_size]
        
        attention_output = attention_output.view(batch_size, seq_len, hidden_size)
        # Shape: [batch_size, seq_len, hidden_size]
        
        print(f"After concatenation: {attention_output.shape}")
        
        # Step 5: Output projection
        output = self.out_proj(attention_output)
        print(f"Final output shape: {output.shape}")
        
        return output, attention_weights
    
    def scaled_dot_product_attention(self, q, k, v, causal_mask=True):
        """
        Scaled Dot-Product Attention computation
        
        Args:
            q: Query [batch_size, num_heads, seq_len, head_size]
            k: Key [batch_size, num_heads, seq_len, head_size]  
            v: Value [batch_size, num_heads, seq_len, head_size]
            causal_mask: Whether to apply causal masking
            
        Returns:
            output: [batch_size, num_heads, seq_len, head_size]
            attention_weights: [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, num_heads, seq_len, head_size = q.size()
        
        # Step 1: Compute attention scores
        # Q × K^T
        scores = torch.matmul(q, k.transpose(-2, -1))
        # Shape: [batch_size, num_heads, seq_len, seq_len]
        
        print(f"  Raw attention scores shape: {scores.shape}")
        print(f"  Sample scores (first head, first sequence):")
        if batch_size > 0:
            print(f"    {scores[0, 0, :min(4, seq_len), :min(4, seq_len)]}")
        
        # Step 2: Scale by sqrt(head_size)
        scale_factor = math.sqrt(head_size)
        scaled_scores = scores / scale_factor
        
        print(f"  Scale factor: {scale_factor}")
        print(f"  Scaled scores range: [{scaled_scores.min():.3f}, {scaled_scores.max():.3f}]")
        
        # Step 3: Apply causal mask (for language modeling)
        if causal_mask:
            # Create lower triangular mask
            mask = torch.tril(torch.ones(seq_len, seq_len, device=q.device))
            # Apply mask: set upper triangular part to -inf
            scaled_scores = scaled_scores.masked_fill(mask == 0, float('-inf'))
            
            print(f"  Applied causal mask")
        
        # Step 4: Apply softmax to get attention weights
        attention_weights = F.softmax(scaled_scores, dim=-1)
        # Shape: [batch_size, num_heads, seq_len, seq_len]
        
        print(f"  Attention weights shape: {attention_weights.shape}")
        print(f"  Attention weights sum (should be ~1.0): {attention_weights[0, 0, 0].sum():.3f}")
        
        # Step 5: Apply dropout
        attention_weights = self.dropout(attention_weights)
        
        # Step 6: Apply attention weights to values
        output = torch.matmul(attention_weights, v)
        # Shape: [batch_size, num_heads, seq_len, head_size]
        
        return output, attention_weights


# 使用例とテスト
def test_multihead_attention():
    """MHSAの動作テスト"""
    
    # モデルパラメータ (GPT-2 small)
    batch_size = 2
    seq_len = 8
    hidden_size = 768
    num_heads = 12
    
    # ダミー入力データの生成
    torch.manual_seed(42)
    input_tensor = torch.randn(batch_size, seq_len, hidden_size)
    
    print("=== Multi-Head Self-Attention Test ===")
    print(f"Input shape: {input_tensor.shape}")
    print(f"Hidden size: {hidden_size}, Num heads: {num_heads}, Head size: {hidden_size//num_heads}")
    print()
    
    # MHSAモジュールの初期化
    mhsa = MultiHeadSelfAttention(hidden_size, num_heads)
    
    # 順伝播の実行
    with torch.no_grad():
        output, attention_weights = mhsa(input_tensor)
    
    print(f"\n=== Results ===")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    # 統計情報の出力
    print(f"\nOutput statistics:")
    print(f"  Mean: {output.mean():.6f}")
    print(f"  Std: {output.std():.6f}")
    print(f"  Min: {output.min():.6f}")  
    print(f"  Max: {output.max():.6f}")

if __name__ == "__main__":
    test_multihead_attention()
```

## 3. 各ステップの詳細解説

### 3.1 Step 1: Query, Key, Valueの生成

```python
def generate_qkv(x, weight_q, weight_k, weight_v):
    """
    Linear projections to generate Q, K, V
    
    数学的表現:
    Q = X × W_Q + b_Q
    K = X × W_K + b_K  
    V = X × W_V + b_V
    
    Args:
        x: [batch_size, seq_len, hidden_size]
        weight_q, weight_k, weight_v: [hidden_size, hidden_size]
    
    Returns:
        q, k, v: Each [batch_size, seq_len, hidden_size]
    """
    q = torch.matmul(x, weight_q)  # Query: "何を探しているか"
    k = torch.matmul(x, weight_k)  # Key: "各位置の識別子"
    v = torch.matmul(x, weight_v)  # Value: "実際の値/情報"
    
    return q, k, v
```

**直感的理解:**
- **Query**: 「この位置から他の位置の何を知りたいか」
- **Key**: 「各位置がどのような特徴を持つか」  
- **Value**: 「各位置が持つ実際の情報」

### 3.2 Step 2: マルチヘッド分割

```python
def split_into_heads(tensor, num_heads):
    """
    テンソルをマルチヘッドに分割
    
    [batch_size, seq_len, hidden_size] 
    ↓
    [batch_size, seq_len, num_heads, head_size]
    ↓  
    [batch_size, num_heads, seq_len, head_size]
    """
    batch_size, seq_len, hidden_size = tensor.size()
    head_size = hidden_size // num_heads
    
    # Reshape and transpose
    tensor = tensor.view(batch_size, seq_len, num_heads, head_size)
    tensor = tensor.transpose(1, 2)
    
    return tensor

# 例: GPT-2 smallの場合
# Input: [2, 1024, 768]
# After split: [2, 12, 1024, 64]
# 12個のヘッド、各ヘッドは64次元
```

### 3.3 Step 3: Scaled Dot-Product Attention

これがSelf-Attentionの核心部分です。

#### 3.3.1 Attention Score計算

```python
def compute_attention_scores(q, k):
    """
    Attention score = Q × K^T
    
    直感的意味:
    - 各Query（位置i）と各Key（位置j）の内積
    - 内積が大きい = 位置iが位置jを重視すべき
    - 内積が小さい = 位置iにとって位置jは重要でない
    """
    scores = torch.matmul(q, k.transpose(-2, -1))
    
    # Example output shape: [batch_size, num_heads, seq_len, seq_len]
    # scores[b, h, i, j] = Query_i と Key_j の類似度
    
    return scores
```

#### 3.3.2 スケーリングの意味

```python
def scale_scores(scores, head_size):
    """
    スケーリング: scores / sqrt(head_size)
    
    理由:
    1. 次元が高くなると内積の値が大きくなりがち
    2. 大きな値はsoftmax後に極端な分布を作る（勾配消失）
    3. sqrt(head_size)で正規化することで適切な分散を維持
    """
    scale_factor = math.sqrt(head_size)
    scaled_scores = scores / scale_factor
    
    print(f"Scale factor: {scale_factor}")
    print(f"Before scaling - mean: {scores.mean():.3f}, std: {scores.std():.3f}")
    print(f"After scaling - mean: {scaled_scores.mean():.3f}, std: {scaled_scores.std():.3f}")
    
    return scaled_scores
```

#### 3.3.3 Causal Mask（因果マスク）

```python
def apply_causal_mask(scores):
    """
    GPT-2のような言語モデルでは未来のトークンを見てはいけない
    
    マスク前:
    [[s11, s12, s13, s14],
     [s21, s22, s23, s24], 
     [s31, s32, s33, s34],
     [s41, s42, s43, s44]]
    
    マスク後:
    [[s11, -∞,  -∞,  -∞ ],
     [s21, s22, -∞,  -∞ ],
     [s31, s32, s33, -∞ ],
     [s41, s42, s43, s44]]
    """
    seq_len = scores.size(-1)
    
    # 下三角行列のマスクを作成
    mask = torch.tril(torch.ones(seq_len, seq_len))
    
    # 上三角部分（未来の位置）を-infで埋める
    masked_scores = scores.masked_fill(mask == 0, float('-inf'))
    
    return masked_scores
```

#### 3.3.4 Softmax適用

```python
def apply_softmax(scores):
    """
    Softmax: スコアを確率分布に変換
    
    数学的表現:
    attention_weight[i,j] = exp(score[i,j]) / Σ_k exp(score[i,k])
    
    特性:
    - 各行の和が1になる
    - 大きなスコアほど大きな重み
    - -infは0になる（マスクされた位置は無視される）
    """
    attention_weights = F.softmax(scores, dim=-1)
    
    # 検証: 各行の合計は1
    row_sums = attention_weights.sum(dim=-1)
    print(f"Row sums (should be ~1.0): {row_sums[0, 0, :4]}")
    
    return attention_weights
```

#### 3.3.5 Valueへの適用

```python
def apply_attention_to_values(attention_weights, values):
    """
    Attention重みをValueに適用して最終出力を得る
    
    数学的表現:
    output[i] = Σ_j attention_weight[i,j] × value[j]
    
    直感的意味:
    - 位置iの出力は、全位置のValueの重み付き平均
    - 重みはAttention重みで決まる
    - 重要な位置のValueがより大きく貢献
    """
    output = torch.matmul(attention_weights, values)
    
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"Values shape: {values.shape}") 
    print(f"Output shape: {output.shape}")
    
    return output
```

### 3.4 Step 4: ヘッドの結合

```python
def concatenate_heads(multi_head_output):
    """
    複数ヘッドの出力を結合
    
    [batch_size, num_heads, seq_len, head_size]
    ↓
    [batch_size, seq_len, num_heads, head_size]  
    ↓
    [batch_size, seq_len, hidden_size]
    """
    batch_size, num_heads, seq_len, head_size = multi_head_output.size()
    
    # Transpose back
    output = multi_head_output.transpose(1, 2)
    
    # Concatenate heads
    output = output.contiguous().view(batch_size, seq_len, num_heads * head_size)
    
    return output
```

### 3.5 Step 5: 出力射影

```python
def output_projection(concatenated_output, weight_o):
    """
    最終的な線形変換
    
    目的:
    1. 異なるヘッドの情報を統合
    2. モデルが学習しやすい表現に変換
    3. 残差接続との互換性を保つ
    """
    output = torch.matmul(concatenated_output, weight_o)
    return output
```

## 4. 実用的な実装例

### 4.1 簡潔版の実装

```python
class SimpleMultiHeadAttention(nn.Module):
    """簡潔版のMHSA実装"""
    
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        
        # 1. Generate Q, K, V
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 3. Causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len)).to(x.device)
        scores.masked_fill_(mask == 0, float('-inf'))
        
        # 4. Softmax and apply to V
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        
        # 5. Concatenate and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        output = self.W_o(attn_output)
        
        return output
```

### 4.2 効率化版の実装

```python
class EfficientMultiHeadAttention(nn.Module):
    """効率化されたMHSA実装（実際のGPT-2に近い）"""
    
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # QKVを一度に計算（効率化）
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Flash Attentionのための設定（省略可能）
        self.scale = self.d_k ** -0.5
        
    def forward(self, x):
        B, T, C = x.size()  # batch, time, channels
        
        # QKV projection in one go
        qkv = self.qkv_proj(x)  # (B, T, 3*C)
        q, k, v = qkv.split(self.d_model, dim=2)
        
        # Reshape for multi-head
        q = q.view(B, T, self.num_heads, self.d_k).transpose(1, 2)  # (B, nh, T, dk)
        k = k.view(B, T, self.num_heads, self.d_k).transpose(1, 2)  # (B, nh, T, dk)
        v = v.view(B, T, self.num_heads, self.d_k).transpose(1, 2)  # (B, nh, T, dk)
        
        # Attention
        att = (q @ k.transpose(-2, -1)) * self.scale  # (B, nh, T, T)
        
        # Causal mask
        att = att.masked_fill(torch.tril(torch.ones(T, T)) == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        y = att @ v  # (B, nh, T, dk)
        
        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        
        # Output projection
        y = self.out_proj(y)
        
        return y
```

## 5. 数学的理解とVisualizations

### 5.1 Attention重みの可視化

```python
def visualize_attention_weights(attention_weights, tokens=None):
    """
    Attention重みの可視化
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # 最初のヘッドの重みを取得
    weights = attention_weights[0, 0].cpu().numpy()  # [seq_len, seq_len]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(weights, 
                xticklabels=tokens if tokens else False,
                yticklabels=tokens if tokens else False,
                cmap='Blues', 
                cbar=True)
    plt.title('Attention Weights Visualization')
    plt.xlabel('Key positions')
    plt.ylabel('Query positions')
    plt.show()

# 使用例
tokens = ["The", "cat", "sat", "on", "the", "mat", "<EOS>", "<PAD>"]
# visualize_attention_weights(attention_weights, tokens)
```

### 5.2 数式まとめ

**Multi-Head Attention の完全な数式:**

```
MultiHead(Q,K,V) = Concat(head₁, head₂, ..., headₕ)W^O

where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)

Attention(Q,K,V) = softmax(QK^T/√d_k)V

Parameters:
- W_i^Q ∈ ℝ^{d_model × d_k}
- W_i^K ∈ ℝ^{d_model × d_k}  
- W_i^V ∈ ℝ^{d_model × d_v}
- W^O ∈ ℝ^{hd_v × d_model}

Where:
- h = number of heads
- d_k = d_v = d_model / h
```

### 5.3 計算複雑度

```python
def compute_complexity(seq_len, d_model, num_heads):
    """
    MHSAの計算複雑度分析
    """
    d_k = d_model // num_heads
    
    # QKV projections: O(seq_len × d_model²)
    qkv_ops = 3 * seq_len * d_model * d_model
    
    # Attention computation: O(seq_len² × d_model)
    attention_ops = num_heads * seq_len * seq_len * d_k
    
    # Value application: O(seq_len² × d_model)  
    value_ops = num_heads * seq_len * seq_len * d_k
    
    # Output projection: O(seq_len × d_model²)
    output_ops = seq_len * d_model * d_model
    
    total_ops = qkv_ops + attention_ops + value_ops + output_ops
    
    print(f"Complexity Analysis for seq_len={seq_len}, d_model={d_model}")
    print(f"  QKV projections: {qkv_ops:,} ops")
    print(f"  Attention computation: {attention_ops:,} ops") 
    print(f"  Value application: {value_ops:,} ops")
    print(f"  Output projection: {output_ops:,} ops")
    print(f"  Total: {total_ops:,} ops")
    print(f"  Dominant term: O(L²D + LD²) where L={seq_len}, D={d_model}")

# Example: GPT-2 small
compute_complexity(1024, 768, 12)
```

## 6. 実践的なTips

### 6.1 メモリ効率化

```python
class MemoryEfficientMHSA(nn.Module):
    """メモリ効率を考慮したMHSA"""
    
    def __init__(self, d_model, num_heads, max_seq_len=1024):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.max_seq_len = max_seq_len
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Pre-computed causal mask
        self.register_buffer('causal_mask', 
                           torch.tril(torch.ones(max_seq_len, max_seq_len)))
    
    def forward(self, x):
        B, T, C = x.size()
        
        # Ensure we don't exceed max sequence length
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds maximum {self.max_seq_len}"
        
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.d_model, dim=2)
        
        # Reshape with efficient memory layout
        q = q.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        
        # Use pre-computed mask
        att = (q @ k.transpose(-2, -1)) * (self.d_k ** -0.5)
        att = att.masked_fill(self.causal_mask[:T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.out_proj(y)
```

### 6.2 勾配チェックポイント

```python
import torch.utils.checkpoint as checkpoint

class CheckpointedMHSA(nn.Module):
    """勾配チェックポイントを使用したMHSA"""
    
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model, num_heads)
    
    def forward(self, x):
        # メモリを節約するため中間結果を保存せず、必要時に再計算
        return checkpoint.checkpoint(self.attention, x)
```

## 7. トラブルシューティング

### 7.1 よくある問題と解決法

```python
def debug_attention_issues():
    """
    MHSAでよく発生する問題とデバッグ方法
    """
    
    print("=== Common MHSA Issues and Solutions ===")
    
    # 1. NaN値の発生
    print("\n1. NaN Values:")
    print("   Problem: Softmax後にNaN値が出現")
    print("   Causes: ")
    print("     - Attention scoresが全て-infになる")
    print("     - 数値が不安定になる（スケールが大きすぎる）")
    print("   Solutions:")
    print("     - Proper weight initialization")
    print("     - Gradient clipping")  
    print("     - Lower learning rate")
    
    # 2. メモリ不足
    print("\n2. Out of Memory:")
    print("   Problem: GPU memory exhaustion")
    print("   Causes:")
    print("     - Attention matrix O(seq_len²) grows quickly")
    print("     - Long sequences (>2048 tokens)")
    print("   Solutions:")
    print("     - Gradient checkpointing")
    print("     - Mixed precision training")
    print("     - Sequence length reduction")
    print("     - Flash Attention")
    
    # 3. 学習が進まない
    print("\n3. Training Doesn't Progress:")
    print("   Problem: Loss doesn't decrease")
    print("   Causes:")
    print("     - Improper masking")
    print("     - Wrong attention weights")
    print("     - Vanishing/exploding gradients")
    print("   Solutions:")
    print("     - Check causal mask implementation")
    print("     - Verify attention weight normalization")
    print("     - Add residual connections")
    print("     - Use proper initialization")

debug_attention_issues()
```

### 7.2 デバッグ用ユーティリティ

```python
class AttentionDebugger:
    """MHSA の詳細な診断ツール"""
    
    def __init__(self, model):
        self.model = model
        self.hooks = []
        
    def register_hooks(self):
        """中間値を記録するフックを登録"""
        def hook_fn(name):
            def hook(module, input, output):
                print(f"  {name}: {output.shape}")
                if torch.isnan(output).any():
                    print(f"    WARNING: NaN detected in {name}")
                if torch.isinf(output).any():
                    print(f"    WARNING: Inf detected in {name}")
            return hook
        
        # 各層にフックを追加
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                handle = module.register_forward_hook(hook_fn(name))
                self.hooks.append(handle)
    
    def remove_hooks(self):
        """フックを削除"""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []
    
    def diagnose_attention_weights(self, attention_weights):
        """Attention重みの診断"""
        print("=== Attention Weights Diagnosis ===")
        
        # 基本統計
        print(f"Shape: {attention_weights.shape}")
        print(f"Min: {attention_weights.min():.6f}")
        print(f"Max: {attention_weights.max():.6f}")
        print(f"Mean: {attention_weights.mean():.6f}")
        print(f"Std: {attention_weights.std():.6f}")
        
        # 正規化チェック
        row_sums = attention_weights.sum(dim=-1)
        print(f"Row sums (should be ~1.0):")
        print(f"  Min: {row_sums.min():.6f}")
        print(f"  Max: {row_sums.max():.6f}")
        print(f"  Mean: {row_sums.mean():.6f}")
        
        # 異常値チェック
        if torch.isnan(attention_weights).any():
            print("ERROR: NaN values found in attention weights")
        if torch.isinf(attention_weights).any():
            print("ERROR: Infinite values found in attention weights")
        
        # 分布の均一性チェック
        entropy = -(attention_weights * torch.log(attention_weights + 1e-9)).sum(dim=-1)
        print(f"Entropy (higher = more uniform):")
        print(f"  Min: {entropy.min():.3f}")
        print(f"  Max: {entropy.max():.3f}")
        print(f"  Mean: {entropy.mean():.3f}")
```

## 8. Performance Optimization

### 8.1 Flash Attention (理論)

```python
# Flash Attentionの概念実装（実際の実装は複雑）
def flash_attention_concept(q, k, v, block_size=64):
    """
    Flash Attentionのコンセプト実装
    
    主なアイデア:
    1. Attention matrixを全て計算せず、ブロック単位で処理
    2. メモリ使用量をO(seq_len²) → O(seq_len)に削減
    3. 計算速度も向上（メモリアクセス最適化）
    """
    batch_size, num_heads, seq_len, head_dim = q.shape
    
    # 出力バッファ
    output = torch.zeros_like(v)
    
    # ブロック単位で処理
    for i in range(0, seq_len, block_size):
        for j in range(0, seq_len, block_size):
            # 小さなブロックのattention計算
            q_block = q[:, :, i:i+block_size, :]
            k_block = k[:, :, j:j+block_size, :]
            v_block = v[:, :, j:j+block_size, :]
            
            # 通常のattention計算（小さなブロックで）
            scores = torch.matmul(q_block, k_block.transpose(-2, -1))
            scores = scores / math.sqrt(head_dim)
            
            # Causal maskingなどの処理...
            
            attn_weights = F.softmax(scores, dim=-1)
            block_output = torch.matmul(attn_weights, v_block)
            
            # 結果を蓄積
            output[:, :, i:i+block_size, :] += block_output
    
    return output
```

### 8.2 Multi-Query Attention (MQA)

```python
class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention: Key/Valueを全ヘッドで共有
    メモリとキャッシュサイズを削減
    """
    
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Queryは複数ヘッド、Key/Valueは単一
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, self.d_k)  # 単一ヘッドサイズ
        self.W_v = nn.Linear(d_model, self.d_k)  # 単一ヘッドサイズ
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        
        # Query: [B, T, H, D_k]
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        Q = Q.transpose(1, 2)  # [B, H, T, D_k]
        
        # Key, Value: [B, T, D_k] (単一ヘッド)
        K = self.W_k(x).unsqueeze(1)  # [B, 1, T, D_k] 
        V = self.W_v(x).unsqueeze(1)  # [B, 1, T, D_k]
        
        # 全ヘッドでKey/Valueを共有
        K = K.expand(-1, self.num_heads, -1, -1)  # [B, H, T, D_k]
        V = V.expand(-1, self.num_heads, -1, -1)  # [B, H, T, D_k]
        
        # 通常のattention計算
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Causal mask
        seq_len = scores.size(-1)
        mask = torch.tril(torch.ones(seq_len, seq_len)).to(x.device)
        scores.masked_fill_(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        
        # Concatenate and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.W_o(output)
```

## 9. 実際のGPT-2との対応

### 9.1 HuggingFace Transformersとの比較

```python
# 我々の実装
our_mhsa = MultiHeadSelfAttention(hidden_size=768, num_heads=12)

# HuggingFace GPT-2の対応部分
from transformers import GPT2Model
gpt2 = GPT2Model.from_pretrained('gpt2')
hf_attention = gpt2.h[0].attn  # 最初の層のattention

print("=== Parameter Comparison ===")
print(f"Our QKV weight shape: {our_mhsa.qkv_proj.weight.shape}")
print(f"HF c_attn weight shape: {hf_attention.c_attn.weight.shape}")
print(f"Our output weight shape: {our_mhsa.out_proj.weight.shape}")  
print(f"HF c_proj weight shape: {hf_attention.c_proj.weight.shape}")
```

### 9.2 重みの初期化方法

```python
def gpt2_style_init(module):
    """GPT-2スタイルの重み初期化"""
    
    if isinstance(module, nn.Linear):
        # 正規分布初期化（std=0.02）
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

# 適用
model = MultiHeadSelfAttention(768, 12)
model.apply(gpt2_style_init)
```

## 10. まとめと今後の発展

### 10.1 MHSAの重要性

Multi-Head Self-Attentionは現代の言語モデルの根幹をなす技術です：

1. **並列処理**: 各ヘッドが独立して異なる種類の関係性を学習
2. **長距離依存**: シーケンス内の離れた位置間の関係を直接モデル化
3. **解釈可能性**: Attention重みにより、モデルがどこに注目しているかを可視化可能
4. **スケーラビリティ**: 効率化手法により大規模モデルでの実用が可能

### 10.2 発展的な技術

```python
# 将来のAttention技術の方向性
future_attention_trends = {
    "efficiency": [
        "Flash Attention",
        "Linformer", 
        "Performer",
        "BigBird"
    ],
    "architectural": [
        "Multi-Query Attention", 
        "Grouped-Query Attention",
        "RoPE (Rotary Position Embedding)",
        "ALiBi (Attention with Linear Biases)"
    ],
    "optimization": [
        "Mixed Precision",
        "Gradient Checkpointing", 
        "Model Parallelism",
        "Sequence Parallelism"
    ]
}

for category, techniques in future_attention_trends.items():
    print(f"{category.upper()}:")
    for technique in techniques:
        print(f"  - {technique}")
```

### 10.3 実用的な実装のベストプラクティス

```python
def production_ready_mhsa(d_model, num_heads, dropout=0.1, max_seq_len=2048):
    """本番環境で使用可能なMHSA実装のテンプレート"""
    
    class ProductionMHSA(nn.Module):
        def __init__(self):
            super().__init__()
            
            # 基本パラメータ
            self.d_model = d_model
            self.num_heads = num_heads
            self.d_k = d_model // num_heads
            self.scale = 1.0 / math.sqrt(self.d_k)
            
            # レイヤー
            self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=True)
            self.out_proj = nn.Linear(d_model, d_model, bias=True)
            self.dropout = nn.Dropout(dropout)
            
            # 最適化のための事前計算
            self.register_buffer(
                'causal_mask',
                torch.tril(torch.ones(max_seq_len, max_seq_len))
            )
            
            # 重み初期化
            self._init_weights()
        
        def _init_weights(self):
            nn.init.normal_(self.qkv_proj.weight, std=0.02)
            nn.init.normal_(self.out_proj.weight, std=0.02)
            nn.init.zeros_(self.qkv_proj.bias)
            nn.init.zeros_(self.out_proj.bias)
        
        @torch.cuda.amp.autocast()  # Mixed precision support
        def forward(self, x):
            B, T, C = x.size()
            
            # Efficient QKV computation
            qkv = self.qkv_proj(x)
            q, k, v = qkv.split(self.d_model, dim=2)
            
            # Multi-head reshape
            q = q.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
            k = k.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
            v = v.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
            
            # Scaled dot-product attention
            att = (q @ k.transpose(-2, -1)) * self.scale
            att = att.masked_fill(self.causal_mask[:T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.dropout(att)
            
            y = att @ v
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            
            return self.out_proj(y)
    
    return ProductionMHSA()

# 使用例
production_model = production_ready_mhsa(768, 12)
```

Multi-Head Self-Attentionは、GPT-2をはじめとする現代の言語モデルの性能を支える核心技術です。この詳細な理解により、より効率的で強力な言語モデルの開発や、既存モデルの最適化が可能になります。