# Google ColabでHugging FaceのGPT-2事前学習モデルを動かす手順

## 1. 環境セットアップ

まずは必要なライブラリをインストールします。

```python
!pip install transformers
!pip install torch
```

---

## 2. モデルとトークナイザーのインポート

Hugging Faceの`transformers`ライブラリを使ってGPT-2のモデルとトークナイザーを読み込みます。

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# トークナイザーのロード
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# GPT-2の事前学習済みモデルをロード（言語モデルヘッド付き）
model = GPT2LMHeadModel.from_pretrained("gpt2")
```

---

## 3. テキストのトークナイズ（前処理）

モデル入力用にテキストをトークンIDに変換します。

```python
input_text = "Hello, how are you?"
input_ids = tokenizer.encode(input_text, return_tensors="pt")  # PyTorchテンソル形式
```

---

## 4. モデルでテキスト生成 or ログ確率計算

### (a) テキスト生成例

```python
# 生成時の設定（最大長など）
outputs = model.generate(input_ids, max_length=50, num_return_sequences=1)

# 生成結果のデコード
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

---

### (b) ログ確率（損失）計算例

次のトークンの予測精度を確認したい場合。

```python
outputs = model(input_ids, labels=input_ids)
loss = outputs.loss
logits = outputs.logits

print(f"Loss: {loss.item()}")
```

---
