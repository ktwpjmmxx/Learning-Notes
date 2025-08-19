# GPT-2推論フェーズの詳細解説

GPT-2の推論フェーズでは、入力プロンプトから出力テキストを生成するまでに複数の処理段階を経ます。以下、各段階を詳細に解説します。

## 1. 入力前処理とトークン化

### 1.1 テキストの前処理
```
入力テキスト: "Hello, world!"
↓
正規化・エスケープ処理
↓  
処理済みテキスト: "Hello, world!"
```

### 1.2 トークン化（Byte Pair Encoding）
GPT-2はBPE（Byte Pair Encoding）を使用してテキストをトークンに分割します。

```
"Hello, world!" 
↓ BPE処理
↓
トークンID: [15496, 11, 995, 0]
```

**データ形状**: `[1, sequence_length]` (バッチサイズ=1の場合)

## 2. Embedding層の処理

### 2.1 Token Embedding
各トークンIDを高次元ベクトル（GPT-2では768次元）に変換します。

```python
# 疑似コード
token_ids = [15496, 11, 995, 0]  # shape: [4]
token_embeddings = embedding_table[token_ids]  # shape: [4, 768]
```

### 2.2 Positional Embedding
各位置に対応する位置埋め込みを追加します。

```python
# 位置0, 1, 2, 3に対する位置埋め込み
position_embeddings = position_embedding_table[:4]  # shape: [4, 768]
```

### 2.3 最終的な入力表現
```python
input_embeddings = token_embeddings + position_embeddings  # shape: [4, 768]
```

**データ形状**: `[batch_size, sequence_length, hidden_size]` = `[1, 4, 768]`

## 3. Transformerブロックの処理

GPT-2は12層（small）から48層（large）のTransformerブロックを持ちます。各ブロックは以下の構造です：

```
Input (768次元)
    ↓
Layer Norm 1
    ↓
Multi-Head Self-Attention
    ↓
Residual Connection
    ↓
Layer Norm 2
    ↓
Feed Forward Network
    ↓
Residual Connection
    ↓
Output (768次元)
```

### 3.1 Layer Normalization
```python
# 各層の入力を正規化
x_norm = layer_norm(x)  # shape: [1, 4, 768]
```

### 3.2 Multi-Head Self-Attention

#### 3.2.1 Query, Key, Value の計算
```python
# 線形変換でQ, K, Vを生成
Q = x_norm @ W_q  # shape: [1, 4, 768]
K = x_norm @ W_k  # shape: [1, 4, 768] 
V = x_norm @ W_v  # shape: [1, 4, 768]
```

#### 3.2.2 マルチヘッド分割
```python
# 12個のヘッドに分割（768 ÷ 12 = 64次元/ヘッド）
Q = Q.reshape(1, 4, 12, 64)  # shape: [1, 4, 12, 64]
K = K.reshape(1, 4, 12, 64)  # shape: [1, 4, 12, 64]
V = V.reshape(1, 4, 12, 64)  # shape: [1, 4, 12, 64]
```

#### 3.2.3 Scaled Dot-Product Attention
```python
# Attention Score計算
scores = Q @ K.transpose(-2, -1)  # shape: [1, 12, 4, 4]
scores = scores / sqrt(64)  # スケーリング

# Causal Mask適用（未来のトークンを見えなくする）
mask = torch.tril(torch.ones(4, 4))  # 下三角行列
scores = scores.masked_fill(mask == 0, -inf)

# Softmax適用
attention_weights = softmax(scores, dim=-1)  # shape: [1, 12, 4, 4]

# Valueに適用
attention_output = attention_weights @ V  # shape: [1, 12, 4, 64]
```

#### 3.2.4 マルチヘッド結合
```python
# ヘッドを結合
attention_output = attention_output.reshape(1, 4, 768)
# 線形変換
attention_output = attention_output @ W_o  # shape: [1, 4, 768]
```

### 3.3 残差接続とLayer Norm
```python
# 残差接続
x = x + attention_output  # shape: [1, 4, 768]
# Layer Norm
x_norm2 = layer_norm2(x)  # shape: [1, 4, 768]
```

### 3.4 Feed Forward Network (FFN)
```python
# 第1層（拡張）4倍に拡張（768 → 3072）
ffn_hidden = relu(x_norm2 @ W_ffn1 + b_ffn1)  # shape: [1, 4, 3072]

# 第2層（縮約）元の次元に戻す（3072 → 768）
ffn_output = ffn_hidden @ W_ffn2 + b_ffn2  # shape: [1, 4, 768]
```

### 3.5 最終的な残差接続
```python
# 残差接続
x = x + ffn_output  # shape: [1, 4, 768]
```

## 4. 出力層の処理

### 4.1 最終Layer Norm
```python
# 全てのTransformerブロック処理後
final_hidden_states = final_layer_norm(x)  # shape: [1, 4, 768]
```

### 4.2 言語モデルヘッド
```python
# 最後のトークン位置のみを使用（次のトークン予測）
last_hidden_state = final_hidden_states[:, -1, :]  # shape: [1, 768]

# 語彙サイズ（50,257）への線形変換
logits = last_hidden_state @ lm_head_weights  # shape: [1, 50257]
```

### 4.3 確率分布の計算
```python
# Softmaxで確率分布に変換
probabilities = softmax(logits / temperature)  # shape: [1, 50257]
```

## 5. トークン生成（サンプリング）

### 5.1 サンプリング手法

#### Greedy Decoding
```python
# 最も確率の高いトークンを選択
next_token_id = torch.argmax(probabilities, dim=-1)
```

#### Top-k Sampling
```python
# 上位k個のトークンから選択
top_k_probs, top_k_indices = torch.topk(probabilities, k=50)
top_k_probs = top_k_probs / torch.sum(top_k_probs)  # 正規化
next_token_id = torch.multinomial(top_k_probs, 1)
```

#### Top-p (Nucleus) Sampling
```python
# 累積確率がp以下のトークン群から選択
sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
nucleus_mask = cumulative_probs <= p
nucleus_probs = sorted_probs * nucleus_mask
nucleus_probs = nucleus_probs / torch.sum(nucleus_probs)
next_token_id = torch.multinomial(nucleus_probs, 1)
```

## 6. 自己回帰生成プロセス

GPT-2は自己回帰的にテキストを生成します：

```
Step 1: "Hello" → 次のトークン予測 → ","
Step 2: "Hello," → 次のトークン予測 → " world"  
Step 3: "Hello, world" → 次のトークン予測 → "!"
Step 4: "Hello, world!" → 次のトークン予測 → EOS
```

### データフロー図
```
Input Text
    ↓
Tokenization [batch_size, seq_len]
    ↓
Token Embedding [batch_size, seq_len, 768]
    ↓
+ Position Embedding [batch_size, seq_len, 768]
    ↓
Transformer Block 1 [batch_size, seq_len, 768]
    ↓
Transformer Block 2 [batch_size, seq_len, 768]
    ↓
...
    ↓
Transformer Block N [batch_size, seq_len, 768]
    ↓
Final Layer Norm [batch_size, seq_len, 768]
    ↓
LM Head (last token) [batch_size, vocab_size]
    ↓
Softmax [batch_size, vocab_size]
    ↓
Sampling
    ↓
Next Token
```

## 7. 計算複雑度

### 時間計算量
- Self-Attention: O(n² × d) (nは系列長、dは隠れ次元)
- FFN: O(n × d²)
- 全体（L層）: O(L × n² × d + L × n × d²)

### 空間計算量
- アクティベーション: O(L × n × d)
- Attention重み: O(L × n²)

## 8. 最適化技法

### 8.1 KV-Cache
推論時の計算量削減のため、過去のKey/Value行列をキャッシュ：

```python
# t-1ステップまでのKV-Cacheを保存
cached_keys = [K_0, K_1, ..., K_{t-1}]    # 各層のKey
cached_values = [V_0, V_1, ..., V_{t-1}]  # 各層のValue

# 新しいトークンのK,Vのみ計算
K_new = new_token_embedding @ W_k  # 1トークン分のみ
V_new = new_token_embedding @ W_v  # 1トークン分のみ

# キャッシュに追加
K_full = concat(cached_keys, K_new)
V_full = concat(cached_values, V_new)
```

### 8.2 Gradient Checkpointing（学習時）
メモリ使用量を削減するため、一部の中間結果を再計算。

## まとめ

GPT-2の推論フェーズは以下の主要ステップで構成されます：

1. **前処理**: テキスト → BPEトークン
2. **埋め込み**: トークンID → ベクトル表現
3. **Transformer処理**: 多層のSelf-AttentionとFFN
4. **出力生成**: 隠れ状態 → 語彙確率分布
5. **サンプリング**: 確率分布 → 次のトークン
6. **反復**: 自己回帰的な生成継続

各段階でのテンソル形状の変化と、特にSelf-Attentionメカニズムの詳細な計算プロセスが、GPT-2の言語生成能力の核心となっています。