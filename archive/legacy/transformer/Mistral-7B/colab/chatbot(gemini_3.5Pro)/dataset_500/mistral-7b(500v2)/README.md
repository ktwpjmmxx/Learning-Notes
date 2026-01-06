# Mistral-7B-Instruct-v0.2 ファインチューニング実験記録

## 実験概要

**実験ID**: Exp-004  
**データセット**: train_data_500v2.json (500件)  
**目的**: 4bit量子化を用いたMistral-7Bのファインチューニングと過学習対策の検証  
**実行環境**: Google Colab 無料版 (T4 GPU)  
**実行日**: 2025年12月8日

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
- transformers==4.36.2
- tokenizers==0.15.0
- bitsandbytes (GitHubから最新版)
- peft==0.7.1
- accelerate==0.25.0
- trl==0.7.10
- datasets==2.16.1
- scipy

### Step 3: 依存関係の解決

発生した問題と対処:

1. **diffusers依存エラー**
```bash
pip uninstall -y diffusers
```

2. **tokenizersデータ不一致エラー**
```bash
pip uninstall -y tokenizers
pip install tokenizers==0.14.1
# キャッシュクリア
rm -rf ~/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.2
```

3. **huggingface_hub バージョン競合**
```bash
pip install --upgrade huggingface_hub
```

4. **transformers互換性エラー**
```bash
pip install --upgrade transformers
```

5. **trl互換性エラー (top_k_top_p_filtering)**
```bash
pip install --upgrade trl
```

### Step 4: ランタイム再起動
各アップグレード後、必ずランタイムを再起動

---

## データセット

### 構成
| カテゴリ | 件数 | 内容 |
|---------|------|------|
| アイデンティティ | 20 | AIアシスタントの自己紹介 |
| Python関数 | 96 | 組み込み関数の説明 (16関数 × 3パターン × 2回) |
| コーディングタスク | 165 | 具体的なコード例 (11タスク × 15回) |
| 一般常識 | 100 | 基本的な事実知識 (10項目 × 10回) |
| ロールプレイ | 12 | トーン指示への対応 (6項目 × 2回) |
| ランダム補完 | 107 | 上記から補完 |
| **合計** | **500** | |

### データ形式
```json
{
  "question": "Pythonのprint とは何ですか？",
  "answer": "print はPythonの組み込み関数です。具体的な仕様は公式ドキュメントを確認してください。"
}
```

### プロンプトフォーマット
```
<s>[INST] {question} [/INST] {answer} </s>
```

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
    "output_dir": "outputs_exp004",
    "num_train_epochs": 0.5,
    "learning_rate": 5e-6,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.1,
    
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    
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

実効バッチサイズ: 1 × 4 = 4

---

## コード修正履歴

### config.py
```python
# 修正前
"evaluation_strategy": "steps"

# 修正後
"eval_strategy": "steps"
```

### 04_train.py

#### 修正1: データのフォーマット
```python
# 修正後
def format_data(example):
    prompt = generate_prompt_for_training(example)
    return {"text": prompt}

dataset = dataset.map(format_data, remove_columns=dataset.column_names)
```

#### 修正2: SFTTrainer引数
```python
# 修正前
trainer = SFTTrainer(
    tokenizer=tokenizer,
    max_seq_length=512,
    packing=False,
    dataset_text_field=None,
    formatting_func=formatting_prompts_func,
)

# 修正後
trainer = SFTTrainer(
    processing_class=tokenizer,
    # max_seq_length, packing, dataset_text_field, formatting_func は削除
)
```

---

## 実行手順

### 1. データ生成
```bash
python 00_generate_data.py
```

出力: `data/train_data_500v2.json`

### 2. 学習実行
```bash
python 04_train.py
```

所要時間:
- モデルダウンロード: 約5-10分
- 学習: 約20-30分

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
- 最終 Train Loss: (記録)
- 最終 Eval Loss: (記録)
- 最良 Eval Loss: (記録)

### 保存ファイル
- モデル: `mistral-7b-custom-chat_exp004/`
- ログ: `outputs_exp004/`
- 推論ログ: `inference_log_exp004.txt`

---

## 評価結果

### 問題点
1. **日英混在**: 日本語の質問に対して英語で回答する傾向
2. **学習不足**: エポック数0.5では学習データの特徴を十分に習得できなかった
3. **プロンプト設計**: 日本語出力を促す指示が不足

### 成功した点
- 4bit量子化での学習が正常に完了
- 過学習は発生せず (Train/Eval Gap < 0.5)
- コード例や事実知識は部分的に学習

---

## 次回実験への提言 (Exp-005)

### データ修正
```python
# プロンプトに日本語指示を明示
def format_prompt(question, answer=None):
    system_msg = "あなたは日本語で回答するAIアシスタントです。"
    prompt = f"<s>[INST] {system_msg}\n\n質問: {question} [/INST]"
    if answer:
        prompt += f" {answer} </s>"
    return prompt
```

### パラメータ調整
```python
"num_train_epochs": 2,           # 0.5 → 2
"learning_rate": 1e-5,           # 5e-6 → 1e-5
"gradient_accumulation_steps": 8, # 4 → 8
```

### 重要な注意点
- 前回のモデル (exp004) は使用せず、ゼロから学習し直す
- 日英混在の癖を持つモデルを継続学習しても改善は困難

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
最新版のtrlでは以下の引数が削除:
- `tokenizer` → `processing_class`
- `max_seq_length`
- `packing`
- `dataset_text_field`
- `formatting_func`

### Out of Memory
```python
# config.pyを編集
"gradient_accumulation_steps": 4 → 8
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
│   └── train_data_500v2.json    # 学習データ
├── outputs_exp004/               # 学習ログ
├── mistral-7b-custom-chat_exp004/  # 学習済みモデル
└── inference_log_exp004.txt      # 推論ログ
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

### 学んだこと
1. Google Colabの依存関係管理は頻繁に変更されるため、柔軟な対応が必要
2. 最新版のライブラリ同士は互換性がない場合があり、段階的なアップグレードが重要
3. bitsandbytesはCUDAバージョンとの互換性が厳しく、GitHubからの直接インストールが確実
4. SFTTrainerのAPIは頻繁に変更されるため、公式ドキュメントの確認が不可欠
5. 少量データ (500件) でのファインチューニングは、プロンプト設計が特に重要

### 所要時間
- 環境構築: 約2時間 (試行錯誤含む)
- データ作成: 約10分
- 学習: 約30分
- 評価: 約10分

合計: 約3時間

---

## ライセンス

本実験は学習・研究目的で実施されました。  
使用したモデル (Mistral-7B-Instruct-v0.2) のライセンスに従ってください。