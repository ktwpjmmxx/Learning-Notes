# Finetuning Strategies: ファインチューニングの手法と戦略

事前学習済み GPT モデルを特定タスク向けに調整する手法を整理。  
データ準備からハイパーパラメータ調整、評価方法まで、実務で役立つ知見をまとめる。

---

### 目次
1. データ準備とプリプロセシング
   - トークナイズとパディング
   - データ拡張やクリーニング
2. ハイパーパラメータ調整
   - 学習率、バッチサイズ、エポック数
   - LoRA や Freeze Layer などの戦略
3. 評価指標と検証方法
   - Perplexity や BLEU などの定量評価
   - サンプル生成による定性的評価

### 1. データ準備とプリプロセシング

#### 1-1. トークナイズとパディング
- 入力文章をトークナイザーで **トークンIDに変換**
- 文の長さを揃えるために **パディング** を追加
  - Attention Mask でパディング部分を無視させる
- ミニバッチ学習に必須

#### GPT 向けパディング設定の例

- GPT には `<PAD>` トークンがないので、パディングが必要な場合は **`<EOS>` トークンで代用**
- HuggingFace の tokenizer で設定する場合：

```python
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # pad = eos に設定

# 例: バッチの文をエンコード
sentences = ["Hello world", "Hi"]
encoded = tokenizer(
    sentences,
    padding=True,         # 短い文は pad で埋める
    truncation=True,      # 長すぎる文は切る
    return_tensors="pt"
)

print(encoded["input_ids"])
print(encoded["attention_mask"])

```

#### 1-2. データ拡張やクリーニング
- ノイズや不要な文字を除去して学習データを整形
- 類義語置換や文章順序変更などで **データ拡張** 可能
- データ量が少ない場合に過学習を防ぐ助けになる

#### ポイント
- データの質がファインチューニングの精度を大きく左右
- 特定タスク用に必要な形式に変換することが最優先

#### データクリーニング・ノイズ除去の例

- **特殊文字・記号の除去**
```python
import re

text = "Hello world!!! <br> How are you?"
cleaned = re.sub(r"<[^>]+>", "", text)  # HTMLタグ削除
cleaned = re.sub(r"[^a-zA-Z0-9\s]", "", cleaned)  # 記号削除
print(cleaned)  # "Hello world How are you"

空白・改行の正規化

text = "Hello   world\n\nHow are you?"
cleaned = " ".join(text.split())
print(cleaned)  # "Hello world How are you"

極端に長い・短い文の除外

sentences = ["short", "this is a normal sentence", "x"*500]
filtered = [s for s in sentences if 5 <= len(s.split()) <= 50]
print(filtered)  # ['this is a normal sentence']

```


- **クリーニング**
  - HTMLタグや特殊文字の除去
  - 不要な空白や改行の正規化
  - 異常に長い文章や極端に短い文章の除外
- **データ拡張**
  - 類義語置換: "good" → "nice" など
  - 文の順序変更: 文中の副文を前後入れ替え
  - 小規模ノイズ挿入: スペルミスや表記揺れを模倣
- **ポイント**
  - データの多様性を増やすことで、モデルの汎化性能を向上
  - 学習データが少ない場合は特に重要
  - ただし、過度な拡張はノイズになりうるのでバランスが大事

### 2. ハイパーパラメータ調整

#### 2-1. 基本パラメータ
- **学習率（learning rate）**
  - 小さすぎると学習が遅い
  - 大きすぎると発散する
- **バッチサイズ（batch size）**
  - 大きいほど GPU を効率的に使えるがメモリ消費増
  - 小さいほどノイズのある勾配で汎化性能向上
- **エポック数（num_epochs）**
  - データセット全体を何回学習するか
  - 過学習にならない範囲で設定

#### 2-2. 応用戦略
- **LoRA（Low-Rank Adaptation）**
  - 一部の重みだけ微調整して学習効率を向上
  - 大規模モデルでも少量データでファインチューニング可能
- **Freeze Layer**
  - 上位層や embedding を固定して、下位層だけ学習
  - 計算量削減＆過学習防止

#### 2-3. 調整のポイント
- 小規模データの場合は学習率を低めに設定
- バッチサイズと勾配計算のバランスを考慮
- ハイパーパラメータは **Validation データで評価しながら最適化**

#### ハイパーパラメータ調整と応用戦略（Python 例）

```python
from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler

# --- データローダーでバッチサイズ指定 ---
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# --- エポック数指定 ---
num_epochs = 3  # データセット全体を3回学習

# --- 学習率とオプティマイザ設定 ---
learning_rate = 5e-5
optimizer = AdamW(model.parameters(), lr=learning_rate)

# --- LoRA を使った微調整例 ---
for name, param in model.named_parameters():
    if "lora" in name:
        param.requires_grad = True  # 微調整対象
    else:
        param.requires_grad = False # 固定

# --- Freeze Layer の例 ---
for layer in model.transformer.h[:6]:  # 上半分を Freeze
    for param in layer.parameters():
        param.requires_grad = False

# --- Scheduler 設定 ---
num_training_steps = len(train_dataloader) * num_epochs
lr_scheduler = get_scheduler(
    "linear", optimizer=optimizer, num_warmup_steps=100, num_training_steps=num_training_steps
)

# --- 学習ループ（簡易例） ---
for epoch in range(num_epochs):
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
```

param.requires_grad = False で特定の層を固定 → 計算量削減＆過学習防止
LoRA 用の層だけ微調整すると、大規模モデルでも少量データで学習可能
バッチサイズやエポック数は GPU メモリ・学習進行・過学習のバランスを見ながら設定
学習率や scheduler は Validation を見ながら調整する

### 3. 評価指標と検証方法

#### 3-1. 定量評価（Quantitative Evaluation）
- **Perplexity（パープレキシティ）**
  - 言語モデルの予測精度の指標
  - 低いほどモデルが文脈に沿った生成ができている
- **BLEU / ROUGE**
  - 翻訳や要約タスクで使う
  - モデル生成文と正解文の類似度を測定

  - **BLEU（Bilingual Evaluation Understudy）**  
  - 翻訳タスクで生成文と正解文の n-gram 一致度を測る指標  
  - 高いほど正解文に近い生成
  - **ROUGE（Recall-Oriented Understudy for Gisting Evaluation）**  
  - 要約タスクで使われる指標  
  - 生成文と参照文の重複度を計算（Recall, Precision, F1 で評価）


#### 3-2. 定性評価（Qualitative Evaluation）
- 実際にサンプルを生成して確認
  - 文法的に自然か
  - 文脈に沿った内容か
- 特定のユースケースで期待する応答の品質チェック

#### 3-3. クロスバリデーションと検証データ
- Validation データを用いてモデル性能を定期的にチェック
- 過学習を早期に検知し、学習の打ち切りやハイパーパラメータ調整に活用
- 分割例：
  - Training 70%
  - Validation 15%
  - Test 15%

- **Validation データ**  
  - 学習に使わないデータで、モデル性能をチェックするためのセット  
  - 過学習の早期検知やハイパーパラメータ調整に利用

- **Test データ**  
  - 最終的な性能評価用のデータセット  
  - 学習や調整には一切使わない

#### 具体例：モデル評価の流れ

```python
from datasets import load_metric

# --- Perplexity 計算 ---
import torch
from torch.nn import CrossEntropyLoss

loss_fn = CrossEntropyLoss()
model.eval()

total_loss = 0
with torch.no_grad():
    for batch in val_dataloader:
        outputs = model(**batch)
        # 出力の logits と正解ラベルから loss 計算
        total_loss += loss_fn(outputs.logits.view(-1, vocab_size), batch["labels"].view(-1)).item()

avg_loss = total_loss / len(val_dataloader)
perplexity = torch.exp(torch.tensor(avg_loss))
print(f"Perplexity: {perplexity}")

# --- BLEU 計算例 ---
bleu_metric = load_metric("bleu")
predictions = ["Hello world", "How are you?"]
references = [["Hello world"], ["How are you?"]]
bleu_score = bleu_metric.compute(predictions=predictions, references=references)
print(f"BLEU score: {bleu_score}")

# --- 定性評価（サンプル生成） ---
prompt = "Once upon a time"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

```

Perplexity でモデルの全体精度を定量的に確認
BLEU で特定タスク（翻訳など）の精度を確認
実際に生成した文章を確認することで文脈や自然さを定性的に評価
Validation データを用いて学習途中にチェックすると過学習を防ぎやすい