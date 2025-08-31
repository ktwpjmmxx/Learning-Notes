# GPT-2 Tokenizerの仕組み：エンコード/デコード とBPE

## 1. BPE (Byte Pair Encoding) とは

BPEは、テキストを効率的にトークン化するアルゴリズムです。

### BPEの基本原理
```
段階的にサブワード単位を学習：
1. 文字レベルから開始
2. 最も頻出するペアを統合
3. 語彙サイズに達するまで繰り返し

例：
"hello" → ['h', 'e', 'l', 'l', 'o']
頻出ペア "ll" を統合 → ['h', 'e', 'll', 'o']
```

### GPT-2でのBPE特徴
- **語彙サイズ**: 50,257トークン
- **未知語対応**: どんな文字列も分割可能
- **効率性**: 頻出単語は1トークン、稀な単語は複数トークンに分割

## 2. エンコード処理

### 基本的なエンコード
```python
text = "Hello, world!"
input_ids = tokenizer.encode(text)
print(input_ids)  # [15496, 11, 995, 0]
```

### 詳細なエンコード処理フロー
1. **正規化**: テキストの前処理
2. **BPE分割**: サブワード単位に分割
3. **ID変換**: 語彙辞書でトークンIDに変換
4. **特殊トークン付与**: 必要に応じて開始/終了トークン追加

### エンコードオプション
```python
# 基本
tokens = tokenizer.encode("Hello world")

# テンソル形式で返す
tokens = tokenizer.encode("Hello world", return_tensors="pt")

# 詳細情報も取得
encoded = tokenizer.encode_plus(
    "Hello world",
    return_tensors="pt",
    return_attention_mask=True,
    padding=True
)
```

## 3. デコード処理

### 基本的なデコード
```python
token_ids = [15496, 11, 995]
text = tokenizer.decode(token_ids)
print(text)  # "Hello, world"
```

### デコード処理フロー
1. **ID→トークン変換**: 語彙辞書を参照
2. **BPE逆変換**: サブワードを結合
3. **特殊トークン処理**: 除去または保持
4. **テキスト復元**: 最終的な文字列生成

### デコードオプション
```python
# 特殊トークンを除去（推奨）
text = tokenizer.decode(token_ids, skip_special_tokens=True)

# 特殊トークンを保持
text = tokenizer.decode(token_ids, skip_special_tokens=False)

# クリーンアップなし
text = tokenizer.decode(token_ids, clean_up_tokenization_spaces=False)
```

## 4. 実際の例で理解する

### 例1: 単純な単語
```python
text = "hello"
tokens = tokenizer.tokenize(text)    # ['hello']
ids = tokenizer.encode(text)         # [31373]
decoded = tokenizer.decode(ids)      # "hello"
```

### 例2: 未知語・複合語
```python
text = "GPT-2は素晴らしい"
tokens = tokenizer.tokenize(text)    # ['G', 'PT', '-', '2', 'は', '素', '晴', 'らしい']
ids = tokenizer.encode(text)         # [38, 11571, 12, 17, ...]
decoded = tokenizer.decode(ids)      # "GPT-2は素晴らしい"
```

### 例3: 特殊文字・記号
```python
text = "Hello 🤖 World!"
tokens = tokenizer.tokenize(text)    # ['Hello', 'Ġ', '🤖', 'Ġ', 'World', '!']
# 注: 'Ġ'はスペースを表すBPE表現
```

## 5. チャットボット実装での活用

### 入力処理
```python
def prepare_input(user_text, conversation_history):
    # 会話履歴と新しい入力を結合
    full_context = conversation_history + user_text
    
    # エンコード
    input_ids = tokenizer.encode(full_context, return_tensors="pt")
    
    # 長さ制限（GPT-2は1024トークンまで）
    if input_ids.size(1) > 1000:
        input_ids = input_ids[:, -1000:]  # 末尾1000トークンのみ保持
    
    return input_ids
```

### 出力処理
```python
def process_output(generated_ids, input_length):
    # 入力部分を除去（新しく生成された部分のみ取得）
    new_tokens = generated_ids[0][input_length:]
    
    # デコード
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    # 後処理（不完全な文の除去など）
    response = response.split('.')[0] + '.'  # 最初の完全な文のみ
    
    return response
```

## 6. BPEの利点

### 1. **語彙外単語（OOV）問題の解決**
- 任意の文字列を分割可能
- 新しい単語や固有名詞にも対応

### 2. **効率的な表現**
- 頻出単語：1トークン
- 稀な単語：複数トークンに分割
- メモリ効率が良い

### 3. **多言語対応**
- 文字レベルで処理するため多言語に対応
- 言語固有の前処理が不要

## 7. 注意点

### トークン数の把握
```python
text = "長いテキストの例..."
token_count = len(tokenizer.encode(text))
print(f"トークン数: {token_count}")  # GPT-2の制限（1024）を確認
```

### エンコード/デコードの可逆性
```python
original = "Hello world"
encoded = tokenizer.encode(original)
decoded = tokenizer.decode(encoded)
assert original == decoded  # 基本的に可逆だが、特殊文字で例外あり
```

この仕組みにより、GPT-2は柔軟で効率的なテキスト処理を実現しています。