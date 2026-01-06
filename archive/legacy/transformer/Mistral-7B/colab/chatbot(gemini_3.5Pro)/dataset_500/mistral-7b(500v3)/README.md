# Mistral-7B-Instruct-v0.2 ファインチューニング実験記録

## 実験概要

**実験ID**: Exp-005  
**データセット**: train_data_500v3.json (500件)  
**目的**: 日本語出力の強化と過学習対策の検証  
**実行環境**: Google Colab 無料版 (T4 GPU)  
**実行日**: 2025年12月9日

---

## 実行環境

### ハードウェア
- GPU: NVIDIA Tesla T4 (15GB VRAM)
- CUDA: 12.6
- RAM: 約12GB

### ソフトウェア
- Python: 3.12
- PyTorch: 2.9.0+cu126
- transformers: 4.47.1
- tokenizers: 0.21.0
- bitsandbytes: 0.45.0 (GitHubから最新版)
- peft: 0.7.1
- accelerate: 0.25.0
- trl: 0.12.2
- datasets: 2.16.1

---

## インストール手順

### Step 1: 環境リセット
```bash
# メニュー: ランタイム > ランタイムを出荷時の設定にリセット
```

### Step 2: パッケージインストール
```bash
python 00_install_final.py
```

インストール内容:
- transformers==4.47.1
- tokenizers==0.21.0
- bitsandbytes (GitHubから最新版)
- peft==0.7.1
- accelerate==0.25.0
- trl==0.12.2
- datasets==2.16.1
- scipy

### Step 3: 依存関係の解決

Exp-005では依存関係の問題は発生せず、スムーズにインストールが完了しました。

### Step 4: ランタイム再起動
インストール後、ランタイムを再起動

---

## データセット

### 構成
| カテゴリ | 件数 | 内容 | **Exp-005での変更** |
|---------|------|------|-------------------|
| アイデンティティ | 20 | AIアシスタントの自己紹介 | **日本語対話を明示** |
| Python関数 | 96 | 組み込み関数の説明 (16関数 × 3パターン × 2回) | 変更なし |
| コーディングタスク | 165 | 具体的なコード例 (11タスク × 15回) | 変更なし |
| 一般常識 | 100 | 基本的な事実知識 (10項目 × 10回) | 変更なし |
| ロールプレイ | 12 | トーン指示への対応 (6項目 × 2回) | **日本語強調** |
| ランダム補完 | 107 | 上記から補完 | 変更なし |
| **合計** | **500** | | |

### データ形式
```json
{
  "question": "Pythonのprint とは何ですか？",
  "answer": "print はPythonの組み込み関数です。具体的な仕様は公式ドキュメントを確認してください。"
}
```

### プロンプトフォーマット（Exp-005で強化）
```
<s>[INST] あなたは日本語で回答するAIアシスタントです。必ず日本語で答えてください。

質問: {question} [/INST] {answer} </s>
```

**変更点**: システムメッセージを追加して日本語出力を明示的に指示

---

## モデル設定

### 量子化設定 (BitsAndBytes)
```python
BNB_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": torch.float16,
    "bnb_4bit_use_double_quant": True,
}
```

### LoRA設定
```python
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]
```

学習可能パラメータ: 20,971,520 / 3,773,042,688 (0.56%)

---

## 学習パラメータ

```python
TRAIN_PARAMS = {
    "output_dir": "outputs_exp005",
    "num_train_epochs": 2,
    "learning_rate": 1e-5,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.1,
    
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    
    "eval_strategy": "steps",
    "eval_steps": 10,
    "logging_steps": 5,
    "save_strategy": "steps",
    "save_steps": 10,
    "save_total_limit": 3,
    
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
    
    "fp16": True,
    "bf16": False,
    
    "optim": "paged_adamw_8bit",
    "weight_decay": 0.01,
    "max_grad_norm": 0.5,
    "gradient_checkpointing": True,
}
```

実効バッチサイズ: 1 × 8 = 8

**Exp-004からの変更点**:
- `num_train_epochs`: 0.5 → **2**
- `learning_rate`: 5e-6 → **1e-5**
- `gradient_accumulation_steps`: 4 → **8**

---

## コード修正履歴

### config.py
変更なし（Exp-004で修正済み）

### 04_train.py

すでにExp-004で修正済み:
- データのフォーマット処理
- SFTTrainer引数の最新API対応

### 過学習検出コールバックの追加
```python
class OverfittingDetectionCallback(TrainerCallback):
    """過学習検出コールバック"""
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.train_losses = []
        self.eval_losses = []
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if 'loss' in logs:
                self.train_losses.append(logs['loss'])
            if 'eval_loss' in logs:
                self.eval_losses.append(logs['eval_loss'])
                
                if len(self.train_losses) > 0:
                    train_loss = self.train_losses[-1]
                    eval_loss = self.eval_losses[-1]
                    gap = eval_loss - train_loss
                    
                    print(f"\n[過学習チェック] Train: {train_loss:.4f}, Eval: {eval_loss:.4f}, Gap: {gap:.4f}")
                    
                    if gap > self.threshold:
                        print(f"⚠️ 警告: 過学習の兆候 (Gap > {self.threshold})")
```

---

## 実行手順

### 1. データ生成
```bash
python 00_generate_data.py
```

出力: `data/train_data_500v3.json`

### 2. 学習実行
```bash
python 04_train.py
```

所要時間:
- モデルダウンロード: 約3分 (2回目以降はキャッシュ利用)
- 学習: 約15分

### 3. 推論テスト
```bash
python 05_inference.py
```

---

## 学習結果

### データ分割
- 学習データ: 425件 (85%)
- 検証データ: 75件 (15%)

### 学習統計
- 最終 Train Loss: **0.7390**
- 最終 Eval Loss: **0.7134**
- 最良 Eval Loss: **0.7134**
- 総学習時間: **883秒** (約14分43秒)
- 総ステップ数: **108**

### Loss推移の詳細

| Epoch | Step | Train Loss | Eval Loss | Gap | Entropy | Token Accuracy |
|-------|------|------------|-----------|-----|---------|----------------|
| 0.09 | 5 | 3.3632 | - | - | 1.292 | 59.6% |
| 0.19 | 10 | 3.0274 | 2.8991 | -0.128 | 1.373 | 61.0% |
| 0.38 | 20 | 2.4763 | 2.2037 | -0.273 | 1.471 | 64.8% |
| 0.56 | 30 | 1.8088 | 1.6598 | -0.149 | 1.370 | 74.7% |
| 0.75 | 40 | 1.3554 | 1.2068 | -0.149 | 1.118 | 79.1% |
| 0.94 | 50 | 1.0618 | 0.9584 | -0.103 | 0.909 | 84.6% |
| 1.11 | 60 | 0.9037 | 0.8530 | -0.051 | 0.773 | 87.2% |
| 1.30 | 70 | 0.8211 | 0.7731 | -0.048 | 0.702 | 89.4% |
| 1.49 | 80 | 0.8010 | 0.7338 | -0.067 | 0.678 | 89.4% |
| 1.68 | 90 | 0.7414 | 0.7189 | -0.023 | 0.605 | 90.5% |
| 1.87 | 100 | 0.7791 | 0.7134 | -0.066 | 0.632 | 90.0% |
| 1.96 | 105 | 0.7390 | - | - | 0.623 | 90.4% |

**重要な観察**:
- Train/Eval Gapは常に負の値（過学習なし）
- Entropyは0.6以上を維持（多様性を保持）
- Token Accuracyは90%で安定

### 保存ファイル
- モデル: `mistral-7b-custom-chat_exp005/`
- ログ: `outputs_exp005/`
- 推論ログ: `inference_log_exp005.txt`

---

## 評価結果

### 問題点（Exp-005で新たに判明）
1. **日英混在が悪化**: システムメッセージを追加したにも関わらず、日英混在が改善されず、むしろローマ字表記が頻出
2. **文法の崩壊**: 文法的におかしい日本語や意味不明な出力が多発
3. **コンテキスト理解の欠如**: 質問の内容を正しく理解せず、関連性のない応答を生成
4. **過学習の兆候は軽減**: Train/Eval Gap は -0.0657 と健全な範囲内

### 具体例

#### 例1: ローマ字混在
```
Q: あなたは何語で対話しますか？
A: 私は日本語で応答しております。これらの設計上の決定ですが、今後の機能開発について考慮しています！
```

#### 例2: 完全に意味不明
```
Q: 日本の首都はどこですか？
A: Tokyo desu. 日 capitol ga, Tokyo da to omoimasu. (Tokyo is the capital.)
```

#### 例3: 文法崩壊
```
Q: 社長に「会議は明日です」と伝える場合、どう言いますか？
A: oversasan no arukiwa wa ashita desu で答답します。
```

#### 例4: コード混入
```
Q: 「大丈夫です」と「問題ありません」の使い分けを説明してください。
A: ```python
# これ以下は日文コードスニペットであり...
def daijoubu_tesukae():
    print("大乗夫でつかう")
```
（以下、関係ないコードが続く）
```

#### 例5: 比較的正常（Pythonコード生成）
```
Q: 1から10まで表示するfor文を書いてください。
A: ```python
for i in range(1, 11):
    print(i)
```
```

### 成功した点
- 過学習は発生せず (Train Loss: 0.7390, Eval Loss: 0.7134)
- 学習の安定性が向上 (Gradient Norm が安定)
- Pythonコード生成は比較的正確
- 基本的な質問には答えられる場合もある（稀）

---

## 次回実験への提言 (Exp-006)

### 根本的な問題の分析
Exp-004とExp-005の結果から、**プロンプト設計だけでは日本語出力を強制できない**ことが判明しました。

### 提案される対策

#### 優先度: 最高
1. **ベースモデルの変更**
   - Mistral-7B-Instruct-v0.2は英語中心のモデル
   - 日本語特化モデルへの変更を検討:
     - `rinna/japanese-gpt-neox-3.6b`
     - `cyberagent/open-calm-7b`
     - `elyza/ELYZA-japanese-Llama-2-7b`

**理由**: Exp-004とExp-005の両方で日本語品質に問題が発生。プロンプトやパラメータ調整では解決困難。

2. **データの抜本的見直し**
   - 現在の500件は「量」は十分だが「質」に問題がある可能性
   - より自然な日本語対話データに変更
   - ローマ字表記を含むデータの除去
   - データ生成スクリプトの見直し

#### 優先度: 高
3. **学習パラメータの再調整**
```python
"num_train_epochs": 3,           # 2 → 3
"learning_rate": 2e-5,           # 1e-5 → 2e-5
"gradient_accumulation_steps": 16, # 8 → 16
```

4. **評価指標の追加**
   - BLEU/ROUGEスコアで日本語品質を定量評価
   - パープレキシティの監視
   - 日本語文法チェッカーの導入

#### 優先度: 中
5. **プロンプトエンジニアリング（参考）**
```python
system_msg = """あなたは日本語のみで回答する親切なAIアシスタントです。
ルール:
1. 必ず日本語で回答してください
2. 英語やローマ字は使用しないでください
3. 自然で文法的に正しい日本語を使ってください
"""
```

**注意**: Exp-005の結果から、プロンプトだけでは効果が限定的

### 重要な注意点
- **Exp-005のモデルは使用しない**: 日英混在の癖が定着している
- **ゼロから学習し直す**: 継続学習では改善が困難
- **モデルの選択が最重要**: Mistralは元々英語モデルであることを認識

---

## トラブルシューティング

### bitsandbytes CUDA検出失敗
```bash
pip uninstall -y bitsandbytes
pip install git+https://github.com/TimDettmers/bitsandbytes.git
# ランタイム再起動
```

### tokenizersデータ不一致
```bash
rm -rf ~/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.2
pip install --force-reinstall tokenizers==0.14.1
# ランタイム再起動
```

### SFTTrainer引数エラー
最新版のtrlでは以下の引数が変更:
- `tokenizer` → `processing_class`
- `max_seq_length`, `packing`, `dataset_text_field`, `formatting_func` は削除

### Out of Memory
```python
# config.pyを編集
"gradient_accumulation_steps": 8 → 16
"per_device_train_batch_size": 1 → 1 (維持)
```

---

## ファイル構成

```
project/
├── 00_install_final.py          # インストールスクリプト
├── 00_generate_data.py           # データ生成
├── 04_train.py                   # 学習スクリプト
├── 05_inference.py               # 推論スクリプト
├── config.py                     # 設定ファイル
├── utils.py                      # ユーティリティ
├── load_model.py                 # モデルローダー
├── data/
│   └── train_data_500v3.json    # 学習データ
├── outputs_exp005/               # 学習ログ
├── mistral-7b-custom-chat_exp005/  # 学習済みモデル
└── inference_log_exp005.txt      # 推論ログ
```

---

## 参考情報

### 公式ドキュメント
- Mistral AI: https://docs.mistral.ai/
- Hugging Face Transformers: https://huggingface.co/docs/transformers/
- BitsAndBytes: https://github.com/TimDettmers/bitsandbytes
- PEFT: https://github.com/huggingface/peft
- TRL: https://github.com/huggingface/trl

### 関連リンク
- Mistral-7B-Instruct-v0.2: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2

---

## 実験者ノート

### 学んだこと（Exp-005の追加知見）
1. **システムメッセージだけでは不十分**: プロンプトに日本語指示を追加しても、ベースモデルの言語バイアスは変わらない
2. **エポック数の増加は諸刃の剣**: 過学習は防げたが、不自然な日本語を学習してしまった
3. **ベースモデルの選択が最重要**: 英語中心のモデルで日本語を強制するのは困難
4. **データ品質 > データ量**: 500件でも質が良ければ学習可能だが、不自然なデータは悪影響
5. **Early Stoppingが機能**: Eval Lossは改善し続けたため、モデルは「学習」していた（ただし望ましくない方向に）
6. **過学習対策は成功**: データ分割、Early Stopping、適切なEpoch数により過学習は完全に防止できた

### Exp-004 vs Exp-005 比較

| 項目 | Exp-004 | Exp-005 |
|------|---------|---------|
| Epochs | 0.5 | 2 |
| Learning Rate | 5e-6 | 1e-5 |
| Batch Size (実効) | 4 | 8 |
| Train Loss | (未記録) | 0.7390 |
| Eval Loss | (未記録) | 0.7134 |
| 日本語品質 | 日英混在 | **悪化** (ローマ字混在) |
| 過学習 | なし | なし |
| 学習時間 | 約30分 | 約15分 |
| Entropy (最終) | (未記録) | 0.623 |
| Token Accuracy | (未記録) | 90.4% |

**結論**: 技術的には成功（過学習防止）したが、実用面では失敗（日本語品質悪化）

### 所要時間
- 環境構築: 約30分 (Exp-004の経験により短縮)
- データ作成: 約5分
- 学習: 約15分
- 評価: 約15分

合計: 約1時間5分

### 次のステップ
1. **日本語特化モデルでの実験** (最優先)
2. データセットの品質向上
3. より詳細な評価指標の導入

---

## ライセンス

本実験は学習・研究目的で実施されました。  
使用したモデル (Mistral-7B-Instruct-v0.2) のライセンスに従ってください。