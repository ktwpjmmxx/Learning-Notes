# GPT-2学習フェーズの詳細解説

GPT-2の学習フェーズでは、大規模テキストデータから言語モデルを訓練します。以下、各段階を詳細に解説します。

## 1. データセットと前処理

### 1.1 データセット構成
GPT-2は以下のデータセットで訓練されました：

- **WebText**: 約40GBのテキストデータ
- **出典**: Reddit上で3カルマ以上の外部リンク
- **品質**: 人間によってフィルタリング済み
- **多様性**: ニュース、ブログ、フォーラムなど多様なジャンル

### 1.2 データ前処理パイプライン

```
Raw Text Data (40GB)
    ↓
文字エンコーディング正規化 (UTF-8)
    ↓
重複除去・品質フィルタリング
    ↓
BPE (Byte Pair Encoding) トークン化
    ↓
シーケンス分割 (1024トークン単位)
    ↓
Training Dataset
```

#### 1.2.1 BPE トークン化詳細
```python
# 疑似コード例
raw_text = "Hello world! How are you today?"

# BPE処理
tokens = bpe_encoder.encode(raw_text)
# Output: [15496, 995, 0, 1374, 389, 345, 1909, 30]

# パディング（必要に応じて）
sequence_length = 1024
if len(tokens) < sequence_length:
    tokens.extend([pad_token_id] * (sequence_length - len(tokens)))
```

#### 1.2.2 データ整形
```python
# バッチ形成
batch_size = 8
sequence_length = 1024
input_ids = torch.tensor(tokens).reshape(batch_size, sequence_length)
# Shape: [8, 1024]

# ラベル生成（1トークンシフト）
labels = input_ids[:, 1:].contiguous()  # Shape: [8, 1023]
input_ids = input_ids[:, :-1].contiguous()  # Shape: [8, 1023]
```

## 2. フォワードパス（順伝播）

### 2.1 入力処理とEmbedding

#### 2.1.1 Token Embedding
```python
# 語彙サイズ: 50,257, 隠れ次元: 768
token_embeddings = embedding_table[input_ids]
# Input shape: [8, 1023]
# Output shape: [8, 1023, 768]
```

#### 2.1.2 Positional Embedding
```python
# 学習可能な位置埋め込み
position_ids = torch.arange(1023).expand(8, -1)  # Shape: [8, 1023]
position_embeddings = position_embedding_table[position_ids]
# Output shape: [8, 1023, 768]
```

#### 2.1.3 最終入力表現
```python
# 要素ごとの加算
hidden_states = token_embeddings + position_embeddings
# Shape: [8, 1023, 768]
```

### 2.2 Transformerブロックの処理

GPT-2（small）では12層のTransformerブロックを通過します。

```
Layer 0: Input [8, 1023, 768]
    ↓
Layer Norm 1
    ↓
Multi-Head Attention (12 heads)
    ↓
Residual Connection
    ↓
Layer Norm 2  
    ↓
Feed Forward Network (768→3072→768)
    ↓
Residual Connection
    ↓
Layer 1: Output [8, 1023, 768]
...
Layer 11: Output [8, 1023, 768]
```

#### 2.2.1 Multi-Head Attention詳細計算

```python
# 各層での処理
def multi_head_attention(x, layer_idx):
    batch_size, seq_len, hidden_size = x.shape  # [8, 1023, 768]
    head_size = hidden_size // num_heads  # 768 // 12 = 64
    
    # Layer Normalization
    x_norm = layer_norm_1(x)  # Shape: [8, 1023, 768]
    
    # Linear transformations for Q, K, V
    q = x_norm @ W_q[layer_idx]  # Shape: [8, 1023, 768]
    k = x_norm @ W_k[layer_idx]  # Shape: [8, 1023, 768]  
    v = x_norm @ W_v[layer_idx]  # Shape: [8, 1023, 768]
    
    # Reshape for multi-head
    q = q.reshape(batch_size, seq_len, num_heads, head_size)
    k = k.reshape(batch_size, seq_len, num_heads, head_size)
    v = v.reshape(batch_size, seq_len, num_heads, head_size)
    # Shape: [8, 1023, 12, 64]
    
    # Transpose for attention computation
    q = q.transpose(1, 2)  # Shape: [8, 12, 1023, 64]
    k = k.transpose(1, 2)  # Shape: [8, 12, 1023, 64]
    v = v.transpose(1, 2)  # Shape: [8, 12, 1023, 64]
    
    # Attention scores
    scores = q @ k.transpose(-2, -1)  # Shape: [8, 12, 1023, 1023]
    scores = scores / math.sqrt(head_size)  # Scale
    
    # Causal mask (下三角行列)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len))
    scores = scores.masked_fill(causal_mask == 0, float('-inf'))
    
    # Attention weights
    attn_weights = F.softmax(scores, dim=-1)  # Shape: [8, 12, 1023, 1023]
    
    # Apply attention
    attn_output = attn_weights @ v  # Shape: [8, 12, 1023, 64]
    
    # Concatenate heads
    attn_output = attn_output.transpose(1, 2)  # Shape: [8, 1023, 12, 64]
    attn_output = attn_output.reshape(batch_size, seq_len, hidden_size)
    # Shape: [8, 1023, 768]
    
    # Output projection
    attn_output = attn_output @ W_o[layer_idx]  # Shape: [8, 1023, 768]
    
    return attn_output
```

#### 2.2.2 Feed Forward Network
```python
def feed_forward(x, layer_idx):
    # Layer Normalization
    x_norm = layer_norm_2(x)  # Shape: [8, 1023, 768]
    
    # First linear layer (expansion)
    hidden = F.gelu(x_norm @ W_ffn1[layer_idx] + b_ffn1[layer_idx])
    # Shape: [8, 1023, 3072]
    
    # Second linear layer (projection)
    output = hidden @ W_ffn2[layer_idx] + b_ffn2[layer_idx]
    # Shape: [8, 1023, 768]
    
    return output
```

#### 2.2.3 完全なTransformerブロック
```python
def transformer_block(x, layer_idx):
    # Multi-head attention with residual connection
    attn_output = multi_head_attention(x, layer_idx)
    x = x + attn_output  # Residual connection
    
    # Feed forward network with residual connection  
    ffn_output = feed_forward(x, layer_idx)
    x = x + ffn_output  # Residual connection
    
    return x
```

### 2.3 出力層
```python
# 全てのTransformerブロック通過後
hidden_states = final_layer_norm(hidden_states)  # Shape: [8, 1023, 768]

# Language modeling head
logits = hidden_states @ lm_head.weight.T  # Shape: [8, 1023, 50257]
```

## 3. 損失関数の計算

### 3.1 Cross Entropy Loss

GPT-2は次のトークン予測タスクとして訓練されます。

```python
# logits: [batch_size, sequence_length, vocab_size] = [8, 1023, 50257]
# labels: [batch_size, sequence_length] = [8, 1023]

def compute_loss(logits, labels):
    # Flatten for cross entropy computation
    flat_logits = logits.reshape(-1, vocab_size)  # Shape: [8184, 50257]
    flat_labels = labels.reshape(-1)  # Shape: [8184]
    
    # Cross entropy loss
    loss = F.cross_entropy(flat_logits, flat_labels, ignore_index=pad_token_id)
    
    return loss
```

### 3.2 損失の数学的定義

各位置 `i` での損失：
```
L_i = -log(P(y_i | x_1, x_2, ..., x_i))
```

バッチ全体の平均損失：
```
L = (1/N) * Σ L_i
```

ここで、`N` は有効トークン数（パディングを除く）

### 3.3 Perplexity計算
```python
# Perplexity = exp(loss)
perplexity = torch.exp(loss)
```

## 4. 逆伝播（Backpropagation）

### 4.1 勾配計算の開始
```python
# 損失から逆方向に勾配を計算
loss.backward()
```

### 4.2 各層の勾配計算

#### 4.2.1 出力層の勾配
```python
# Cross entropy lossの勾配
# dL/d(logits) = softmax(logits) - one_hot(labels)
grad_logits = F.softmax(logits, dim=-1)  # Shape: [8, 1023, 50257]
grad_logits[range(batch_size * seq_len), flat_labels] -= 1
grad_logits = grad_logits / (batch_size * seq_len)

# Language modeling headの重みの勾配
grad_lm_head = hidden_states.transpose(1, 2) @ grad_logits
# Shape: [768, 50257]
```

#### 4.2.2 Transformerブロックの勾配

**Feed Forward Networkの勾配：**
```python
def ffn_backward(grad_output, x, layer_idx):
    # 第2線形層の勾配
    grad_W_ffn2 = hidden_ffn.transpose(1, 2) @ grad_output
    grad_b_ffn2 = grad_output.sum(dim=(0, 1))
    grad_hidden_ffn = grad_output @ W_ffn2[layer_idx].T
    
    # GELU活性化関数の勾配
    grad_gelu = grad_hidden_ffn * gelu_derivative(x_norm @ W_ffn1[layer_idx])
    
    # 第1線形層の勾配  
    grad_W_ffn1 = x_norm.transpose(1, 2) @ grad_gelu
    grad_b_ffn1 = grad_gelu.sum(dim=(0, 1))
    grad_x_norm = grad_gelu @ W_ffn1[layer_idx].T
    
    # Layer Normの勾配
    grad_x = layer_norm_backward(grad_x_norm, x)
    
    return grad_x
```

**Multi-Head Attentionの勾配：**
```python
def attention_backward(grad_output, x, layer_idx):
    # 出力射影の勾配
    grad_W_o = attn_concat.transpose(1, 2) @ grad_output
    grad_attn_concat = grad_output @ W_o[layer_idx].T
    
    # マルチヘッド分割の逆
    grad_attn_output = grad_attn_concat.reshape(batch_size, seq_len, num_heads, head_size)
    grad_attn_output = grad_attn_output.transpose(1, 2)
    
    # Attention適用の勾配
    grad_attn_weights = grad_attn_output @ v.transpose(-2, -1)
    grad_v = attn_weights.transpose(-2, -1) @ grad_attn_output
    
    # Softmaxの勾配
    grad_scores = softmax_backward(grad_attn_weights, attn_weights)
    
    # Attention scoresの勾配
    grad_q = grad_scores @ k
    grad_k = grad_scores.transpose(-2, -1) @ q
    
    # Q, K, V線形変換の勾配
    grad_W_q = x_norm.transpose(1, 2) @ grad_q.reshape(batch_size, seq_len, hidden_size)
    grad_W_k = x_norm.transpose(1, 2) @ grad_k.reshape(batch_size, seq_len, hidden_size)  
    grad_W_v = x_norm.transpose(1, 2) @ grad_v.reshape(batch_size, seq_len, hidden_size)
    
    return grad_W_q, grad_W_k, grad_W_v
```

#### 4.2.3 Embeddingの勾配
```python
# Position embeddingの勾配
grad_pos_embed = grad_hidden_states.sum(dim=0)  # Shape: [1023, 768]

# Token embeddingの勾配（スパース更新）
grad_token_embed = torch.zeros_like(embedding_table)
for batch_idx in range(batch_size):
    for pos_idx in range(seq_len):
        token_id = input_ids[batch_idx, pos_idx]
        grad_token_embed[token_id] += grad_hidden_states[batch_idx, pos_idx]
```

### 4.3 勾配の流れ図
```
Loss (scalar)
    ↓
∂L/∂logits [8, 1023, 50257]
    ↓  
∂L/∂hidden_states [8, 1023, 768]
    ↓
∂L/∂layer_11_output [8, 1023, 768]
    ↓
∂L/∂layer_10_output [8, 1023, 768]
    ↓
...
    ↓
∂L/∂layer_0_output [8, 1023, 768]
    ↓
∂L/∂embeddings [8, 1023, 768]
```

## 5. 勾配クリッピング

大きな勾配による学習の不安定性を防ぐため：

```python
# グローバル勾配ノームの計算
total_norm = 0
for param in model.parameters():
    if param.grad is not None:
        param_norm = param.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
total_norm = total_norm ** (1. / 2)

# クリッピング（max_norm = 1.0）
clip_coef = max_norm / (total_norm + 1e-6)
if clip_coef < 1:
    for param in model.parameters():
        if param.grad is not None:
            param.grad.data.mul_(clip_coef)
```

## 6. オプティマイザによるパラメータ更新

### 6.1 Adam Optimizer

GPT-2の学習ではAdamオプティマイザが使用されます。

#### 6.1.1 Adam のハイパーパラメータ
```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=2.5e-4,           # 学習率
    betas=(0.9, 0.999),  # モーメンタム係数
    eps=1e-8,            # 数値安定性のための小さな値
    weight_decay=0.01    # L2正則化
)
```

#### 6.1.2 Adam更新式

各パラメータ `θ` に対して：

```python
def adam_step(param, grad, m, v, t, lr=2.5e-4, beta1=0.9, beta2=0.999, eps=1e-8):
    # モーメンタムの更新
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad**2
    
    # バイアス補正
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    
    # パラメータ更新
    param = param - lr * m_hat / (sqrt(v_hat) + eps)
    
    return param, m, v
```

数学的表現：
```
m_t = β₁ × m_{t-1} + (1-β₁) × g_t
v_t = β₂ × v_{t-1} + (1-β₂) × g_t²

m̂_t = m_t / (1-β₁ᵗ)
v̂_t = v_t / (1-β₂ᵗ)

θ_t = θ_{t-1} - α × m̂_t / (√v̂_t + ε)
```

### 6.2 学習率スケジューリング

```python
# Cosine Annealing with Warm Restarts
def cosine_schedule(step, warmup_steps=4000, max_steps=100000):
    if step < warmup_steps:
        return step / warmup_steps
    else:
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        return 0.5 * (1 + cos(pi * progress))

# 学習率の更新
current_lr = base_lr * cosine_schedule(step)
for param_group in optimizer.param_groups:
    param_group['lr'] = current_lr
```

## 7. 学習ループ全体

### 7.1 1エポックの処理
```python
def training_step(model, batch, optimizer):
    # Forward pass
    input_ids, labels = batch
    
    # 順伝播
    logits = model(input_ids)  # Shape: [batch_size, seq_len, vocab_size]
    
    # 損失計算
    loss = F.cross_entropy(
        logits.view(-1, vocab_size), 
        labels.view(-1),
        ignore_index=pad_token_id
    )
    
    # 逆伝播
    optimizer.zero_grad()
    loss.backward()
    
    # 勾配クリッピング
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # パラメータ更新
    optimizer.step()
    
    return loss.item()
```

### 7.2 完全な学習ループ
```python
# 学習設定
num_epochs = 3
batch_size = 8
accumulation_steps = 16  # 勾配蓄積
log_interval = 100

model.train()
total_steps = 0

for epoch in range(num_epochs):
    epoch_loss = 0
    accumulated_loss = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # Forward + Backward
        loss = training_step(model, batch, optimizer)
        accumulated_loss += loss
        
        # 勾配蓄積
        if (batch_idx + 1) % accumulation_steps == 0:
            total_steps += 1
            avg_loss = accumulated_loss / accumulation_steps
            
            # ロギング
            if total_steps % log_interval == 0:
                perplexity = math.exp(avg_loss)
                print(f"Step {total_steps}, Loss: {avg_loss:.4f}, PPL: {perplexity:.2f}")
            
            accumulated_loss = 0
        
        epoch_loss += loss
    
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {epoch_loss/len(dataloader):.4f}")
```

## 8. メモリ最適化技法

### 8.1 Gradient Checkpointing
```python
# 中間アクティベーションを保存せず、必要時に再計算
class CheckpointedTransformerBlock(nn.Module):
    def forward(self, x):
        return torch.utils.checkpoint.checkpoint(self._forward, x)
    
    def _forward(self, x):
        # 通常のTransformerブロック処理
        return transformer_block(x)
```

### 8.2 Mixed Precision Training
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# Forward pass in half precision
with autocast():
    logits = model(input_ids)
    loss = compute_loss(logits, labels)

# Backward pass with gradient scaling
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## 9. 学習のモニタリング指標

### 9.1 主要メトリクス
```python
# 損失とPerplexity
train_loss = total_loss / num_batches
train_perplexity = math.exp(train_loss)

# 勾配ノーム
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))

# 学習率
current_lr = optimizer.param_groups[0]['lr']

# GPU使用率とメモリ使用量
gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
```

### 9.2 検証
```python
def validate(model, val_loader):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids, labels = batch
            logits = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1))
            total_loss += loss.item()
    
    model.train()
    return total_loss / len(val_loader)
```

## 10. 計算複雑度とスケーリング

### 10.1 時間計算量
- **Forward pass**: O(L × N × d² + L × N² × d)
- **Backward pass**: 約3倍のForward pass
- **Total**: O(L × N × d² + L × N² × d) per step

ここで：
- L: レイヤー数
- N: シーケンス長
- d: 隠れ次元

### 10.2 メモリ使用量
- **パラメータ**: O(L × d²)
- **アクティベーション**: O(B × L × N × d)
- **勾配**: パラメータと同サイズ
- **Optimizer状態**: 約2倍のパラメータサイズ（Adam）

### 10.3 GPT-2モデル別仕様

| モデル | パラメータ数 | レイヤー数 | 隠れ次元 | アテンションヘッド |
|--------|-------------|------------|----------|-------------------|
| Small  | 117M        | 12         | 768      | 12                |
| Medium | 345M        | 24         | 1024     | 16                |
| Large  | 762M        | 36         | 1280     | 20                |
| XL     | 1.5B        | 48         | 1600     | 25                |

## まとめ

GPT-2の学習フェーズは以下の主要ステップで構成されます：

1. **データ準備**: 大規模テキストのBPEトークン化とバッチ化
2. **順伝播**: Embedding → 12-48層のTransformer → 出力層
3. **損失計算**: Cross Entropy Lossによる次トークン予測精度評価
4. **逆伝播**: 全パラメータの勾配計算（約1.17億〜15億パラメータ）
5. **最適化**: Adam OptimizerによるパラメータPULLOW
6. **反復**: 大規模データセット全体での反復学習

特に、Multi-Head Self-Attentionの勾配計算とメモリ効率化が学習成功の鍵となります。