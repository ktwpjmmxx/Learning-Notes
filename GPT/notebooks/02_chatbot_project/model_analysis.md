
# !pip install transformers torch -q

!pip install：Colab上でPython外のコマンドを実行する書き方
-q は quietモード で、出力を最小限にするオプション

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

- **再現性の固定**
 ニューラルネットワークでは乱数により出力が変わることがある
 シード値を固定すると毎回同じ結果が出る

# 出力確認

print("ライブラリの読み込みが完了しました！")
print(f"PyTorchバージョン: {torch.__version__}")
print(f"CUDA利用可能: {torch.cuda.is_available()}")

- **動作環境の確認**
 torch.cuda.is_available() でGPUが使えるかチェック
 torch.__version__でPyTorchのバージョンを確認


# 2.GPT2とトークナイザーの初期化

## 役割について
### 主に次のことを行う：
- **1 - モデルとトークナイザーの初期化**
- **2 - テキストのトークン化（文字列→数字）**
- **3 - 応答の生成**
- **4 - 対話型チャットを実現**

def __init__(self, model_name='gpt2', device=None):

## 引数について：
**model_name**  
 GPT-2のサイズを指定(gpt2,gpt2-mediumなど)
**device**  
 GPU/CPU指定(Noneなら自動選択)

## **主な処理**

### **1.デバイス認定**
- **self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')**
 GPUが使えるならGPU、なければCPU

### **2.トークナイザーの読み込み** 
- **self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)**  
 文字列をトークンIDに変換するツール

### **3.パディングトークンの設定**
- **self.tokenizer.pad_token = self.tokenizer.eos_token**  

### **4.モデルの読み込みとGPU/CPU転送**
- **self.model = GPT2LMHeadModel.from_pretrained(model_name)**
- **self.model.to(self.device)**
言語生成できるGPT-2をロードし、デバイスに移動

### **5.評価モードに設定**  
- **self.model.eval()**
 推論用モード（ドロップアウト無効化）

### **6.モデル情報を表示**  
- **self._print_model_info()**
 語彙数・隠れ層サイズ・パラメータ数などを出力


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

# モデルの規模や構造を一目で把握して、安全かつ効率的に使うため

- **使用環境との適合確認**
 モデルが GPUで処理できるか、CPUしかないか を確認
 大型モデルをCPUで動かすと時間がかかりすぎるので、事前に把握する

- **モデルの能力把握**
 語彙サイズや隠れ層の次元数、ブロック数を見れば、どの程度の文章生成力があるか
 長文を扱えるか
 これにより 用途に合ったモデル選択 ができる

- **デバッグや教育的価値**
 コードの動作確認や学習用として
 GPT-2には何層あるのか？
 アテンションヘッドは何個？
 こういう情報を 可視化することで理解が深まる

- **パラメータ数の把握**
 総パラメータ数で 計算コストの目安 が分かる
 小型→軽量で高速
 大型→重くて高精度
 Colabや自分のPCで動かすときの判断材料になる

## このモデルは自分の環境で使えるのか」「どの程度の能力があるのか」「計算コストはどのくらいか」を確認するために、モデル情報を表示している


  