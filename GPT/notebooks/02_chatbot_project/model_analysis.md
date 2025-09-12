# GPT-2モデル分析ガイド

## 1. 環境セットアップ

### 1.1 ライブラリのインストール

```python
# !pip install transformers torch -q
```

- `!pip install`: Colab上でPython外のコマンドを実行する書き方
- `-q`: quietモードで、出力を最小限にするオプション

### 1.2 必要なライブラリのインポート

```python
import torch
from transformers import (
    GPT2LMHeadModel,      # GPT-2の言語モデル本体
    GPT2Tokenizer,        # テキストをトークンに変換するツール
    GPT2Config,           # モデルの設定情報
    set_seed              # 再現性のためのシード設定
)

import warnings
warnings.filterwarnings('ignore')

set_seed(42)
```

### 1.3 再現性の固定について

- ニューラルネットワークでは乱数により出力が変わることがある
- シード値を固定すると毎回同じ結果が出る

### 1.4 動作環境の確認

```python
print("ライブラリの読み込みが完了しました！")
print(f"PyTorchバージョン: {torch.__version__}")　# 使用中のPyTorchのバージョン
print(f"CUDA利用可能: {torch.cuda.is_available()}")　# GPUが使えるかどうかをTrue/Falseで返す
```

- `torch.cuda.is_available()`でGPUが使えるかチェック
- `torch.__version__`でPyTorchのバージョンを確認

## 2. GPT-2とトークナイザーの初期化

### 2.1 役割について

主に次のことを行う：

1. **モデルとトークナイザーの初期化**
2. **テキストのトークン化（文字列→数字）**
3. **応答の生成**
4. **対話型チャットを実現**

### 2.2 コンストラクタの実装

```python
def __init__(self, model_name='gpt2', device=None):
    """
    コンストラクタ：モデルとトークナイザーを初期化
    
    Args:
        model_name (str): 使用するGPT-2モデルの名前
                        'gpt2': 最小モデル（124Mパラメータ）
                        'gpt2-medium': 中規模（355Mパラメータ）
                        'gpt2-large': 大規模（774Mパラメータ）
                        'gpt2-xl': 超大規模（1.5Bパラメータ）
        device (str): 実行デバイス（Noneの場合は自動選択）
    """
```

#### 引数について

- **model_name**: GPT-2のサイズを指定（gpt2, gpt2-mediumなど）
  - デフォルト値が'gpt2' → 引数を渡さなければ最小モデル（124Mパラメータ）が使われる
- **device**: GPU/CPU指定（Noneなら自動選択）
  - デフォルトはNone → 自動判定

### 2.3 デバイス設定

```python
if device is None:
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    self.device = torch.device(device)
```

- Noneの場合 → GPUが使えるならGPU (cuda)、なければCPU
- 'cpu'や'cuda:0'などを渡すと明示的にデバイスを指定可能
※elseの部分

### 2.4 主な処理の流れ

#### 2.4.1 デバイス認定
```python
self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```
GPUが使えるならGPU、なければCPU

#### 2.4.2 トークナイザーの読み込み
```python
self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
```
文字列をトークンIDに変換するツール

#### 2.4.3 パディングトークンの設定
```python
self.tokenizer.pad_token = self.tokenizer.eos_token
```

#### 2.4.4 モデルの読み込みとGPU/CPU転送
```python
self.model = GPT2LMHeadModel.from_pretrained(model_name)
self.model.to(self.device)
```
言語生成できるGPT-2をロードし、デバイスに移動

#### 2.4.5 評価モードに設定
```python
self.model.eval()
```
推論用モード（ドロップアウト無効化）

##### Dropoutとは

- **概要**: ニューラルネットワークの過学習（オーバーフィッティング）を防ぐための手法
  - 過学習とは → 学習データにはバッチリ適応したけど、新しいデータには弱い状態

- **仕組み**: 訓練時にランダムにニューロン（ノード）を一部無効化する
  - 例：層に100個のニューロンがある
  - ドロップアウト率0.2の場合、ランダムに20個のニューロンを無効化
  - 無効化されたニューロンはそのステップの計算に寄与しない
  - イメージ：回路の一部の電球をランダムに消しても全体の明かりがつくように学習させる

- **推論時にoffにする理由**:
  - 推論（生成）時にドロップアウトを使うと、毎回出力が変わる
  - 学習時はランダム性が必要だけど、推論時は安定性が欲しい

#### 2.4.6 モデル情報を表示
```python
self._print_model_info()
```
語彙数・隠れ層サイズ・パラメータ数などを出力

### 2.5 モデル情報表示の詳細

```python
def _print_model_info(self):
    """モデルの基本情報を表示"""
    config = self.model.config
    
    print("\n" + "="*60)
    print("【GPT-2モデル情報】")
    print("="*60)
    print(f"実行デバイス: {self.device}")
    print(f"語彙サイズ: {config.vocab_size:,} トークン")
    print(f"最大文脈長: {config.n_positions:,} トークン")
    print(f"隠れ層の次元数: {config.n_embd}")
    print(f"アテンションヘッド数: {config.n_head}")
    print(f"Transformerブロック数: {config.n_layer}")
    
    # パラメータ数の計算
    total_params = sum(p.numel() for p in self.model.parameters())
    print(f"総パラメータ数: {total_params:,}")
    print("="*60 + "\n")
```

#### なぜモデル情報を表示するのか

モデルの規模や構造を一目で把握して、安全かつ効率的に使うため

- **使用環境との適合確認**
  - モデルがGPUで処理できるか、CPUしかないかを確認
  - 大型モデルをCPUで動かすと時間がかかりすぎるので、事前に把握する

- **モデルの能力把握**
  - 語彙サイズや隠れ層の次元数、ブロック数を見れば、どの程度の文章生成力があるか
  - 長文を扱えるか
  - これにより用途に合ったモデル選択ができる

- **デバッグや教育的価値**
  - コードの動作確認や学習用として
  - GPT-2には何層あるのか？
  - アテンションヘッドは何個？
  - こういう情報を可視化することで理解が深まる

- **パラメータ数の把握**
  - 総パラメータ数で計算コストの目安が分かる
  - 小型 → 軽量で高速
  - 大型 → 重くて高精度
  - Colabや自分のPCで動かすときの判断材料になる

「このモデルは自分の環境で使えるのか」「どの程度の能力があるのか」「計算コストはどのくらいか」を確認するために、モデル情報を表示している。

### 2.6 パラメータ数の計算

```python
total_params = sum(p.numel() for p in self.model.parameters())
print(f"総パラメータ数: {total_params:,}")
```

**役割**: GPT-2モデルの総パラメータ数を計算
- `p.numel()`はテンソルの要素数を返すので、それを全部合計
- `:,`はフォーマット指定で「3桁区切り」を意味
- 「モデルがどれくらい大きいか」を確認できる

## 3. 主要メソッドの実装

### 3.1 tokenize_text(): テキストをトークン化

```python
def tokenize_text(self, text, show_details=False):
    tokens = self.tokenizer.encode(text, return_tensors='pt').to(self.device)
```
**メソッドの定義**:
- 引数 text → ユーザーが入力する文字列
- 引数 show_details → 詳細情報（各トークンIDなど）を表示するか

 if show_details:
            print("\n【トークナイゼーション詳細】")
            print(f"入力テキスト: '{text}'")
            print(f"トークン数: {tokens.shape[1]}")

・もしshow_detailsがtrueであれば...
・変数textに代入された文字列がTokenizerによっていくつのトークンに分割されるのか表示する

text = "Hello, world!"
tokens = tokenizer.encode(text)
print(tokens)  # 例: [15496, 11, 995]

**役割**: 入力テキストをトークンIDに変換
- 例：「Hello」→ [15496] のように整数IDに
- `return_tensors='pt'`でPyTorchのテンソル型にする
- self.tokenizer.encode() で文字列を トークンIDに変換
- return_tensors='pt' → PyTorch のテンソル形式に変換
- .to(self.device) → GPU または CPU に配置

            token_ids = tokens[0].cpu().numpy()
            for i, token_id in enumerate(token_ids):
                token_str = self.tokenizer.decode([token_id])
                print(f"  トークン{i}: ID={token_id:5d}, テキスト='{token_str}'")
**役割**
- 前提として→tokens = self.tokenizer.encode(text, return_tensors='pt').to(self.device)
- これで tokens は PyTorchのテンソル になってる。
- **例として "Hello, how are you?" を入力したとする**
- # tensor([[15496,    11,   703,   389,   345,    30]])
- ポイント：
形状は (1, 6)（バッチサイズ1、トークン数6）
中身は「各トークンを数字（ID）に変換したもの」

① tokens[0]
tokens は2次元（バッチ×トークン列）。
[0] を指定することで、最初のバッチだけ取り出す。
結果は1次元のテンソル：tensor([15496, 11, 703, 389, 345, 30])

② .cpu()
tokens は .to(self.device) で GPUに載っているかもしれない。
.cpu() で CPUメモリに移動させる。

③ .numpy()
PyTorch の tensor を NumPy の ndarray に変換する。
array([15496,    11,   703,   389,   345,    30], dtype=int64)

### 3.2 generate_response(): 応答生成

```python
outputs = self.model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_length=max_length + input_length,
    temperature=temperature,
    top_k=top_k,
    top_p=top_p,
    num_return_sequences=num_return_sequences,
    do_sample=True,
    pad_token_id=self.tokenizer.eos_token_id,
    eos_token_id=self.tokenizer.eos_token_id,
    early_stopping=True
)
```

#### 処理の流れ

1. `input_ids`にトークン化した入力を渡す
2. `attention_mask`で全トークンを有効化（ゼロなら無視される）
3. `generate()`で文章を続けて生成
4. サンプリング方法を調整できる：
   - `temperature` → 高いほどランダム性が増す
   - `top_k` → 上位k個の候補から選ぶ
   - `top_p` → 確率の合計がpになるまでの候補から選ぶ

**出力**: 入力＋生成されたトークン列
→ ここから入力部分をカットして「生成文」だけ返すようにしている

### 3.3 interactive_chat(): 対話モード

#### 3.3.1 履歴の初期化

```python
conversation_history = ""
max_history_length = 500  # 履歴の最大トークン数
```

#### 3.3.2 ユーザー入力ループと終了判定

```python
user_input = input("あなた: ").strip()
if user_input.lower() in ['quit', 'exit', '終了']:
    break
if user_input.lower() == 'help':
    self._show_help(); continue
if not user_input: continue
```

- `strip()`で前後空白除去
- `lower()`で大文字小文字を無視

#### 3.3.3 プロンプト組み立て（文脈保持）

```python
if conversation_history:
    prompt = f"{conversation_history}\nHuman: {user_input}\nAI:"
else:
    prompt = f"Human: {user_input}\nAI:"
```

def _print_model_info(self):
        """モデルの基本情報を表示"""
        config = self.model.config
        
        print("\n" + "="*60)
        print("【GPT-2モデル情報】")
        print("="*60)
        print(f"実行デバイス: {self.device}")
        print(f"語彙サイズ: {config.vocab_size:,} トークン")
        print(f"最大文脈長: {config.n_positions:,} トークン")
        print(f"隠れ層の次元数: {config.n_embd}")
        print(f"アテンションヘッド数: {config.n_head}")
        print(f"Transformerブロック数: {config.n_layer}")
        
モデル(事前学習済み)の読み込みと空の設計図を設定↓
config = self.model.config

事前学習モデルのデフォルト数値
・vocab_size = 50,257
・n_positions = 1,024
・n_embd = 768
・n_layer = 12
・n_head = 12
・bos_token_id
・eos_token_id = 50256
・pad_token_id = 50256

### パラメータの確認

 total_params = sum(p.numel() for p in self.model.parameters())
        print(f"総パラメータ数: {total_params:,}")
        print("="*60 + "\n")

 **###self.model.parameters()**
- モデルの 全ての重み（weights）やバイアス（bias） を取得する
- ここには学習済みの行列やベクトルが入っている

**##p.numel()**
- パラメータ p の 要素数 を数える
- 例えば 768 x 768 の行列なら 768*768=589,824 個の値

🔹 tokenize_text の流れ

1. テキストをトークン化

tokens = self.tokenizer.encode(text, return_tensors='pt').to(self.device)

**self.tokenizer.encode**
- 入力テキストを トークンIDの列 に変換
- return_tensors='pt'
**PyTorch の tensor 形式にする（例：tensor([[40, 484, 326, 8415]])）**
- .to(self.device)
- CPU / GPU のどちらを使うかに応じて配置

・self.tokenizer は GPT-2 専用のトークナイザー（Byte Pair Encoding, BPE）
・.encode(text, ...) は入力テキストを トークンIDの列に変換

