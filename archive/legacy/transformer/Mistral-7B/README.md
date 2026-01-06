# Mistral-7B Fine-tuned Chatbot Experiments

このリポジトリは、オープンソース大規模言語モデル **Mistral-7B** を用いた
LoRAによる軽量ファインチューニング済みチャットボットの実験プロジェクトです。  
Google Colabやローカル環境での学習・推論・Web UIでの対話までを含みます。

---

## Overview

- **モデル:** [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B)
- **目的:** LoRAによる軽量ファインチューニングの検証およびチャット応答生成
- **技術:** Python, Hugging Face Transformers, PEFT, BitsAndBytes, Gradio
- **開発環境:** Google Colab / ローカルGPU

---

## Repository Structure

Mistral-7B/
├── colab/
│ └── chatbot_experiment.ipynb # Colabでの実験用Notebook
│
├── src/
│ ├── mistral-chat(v2)
│ │ ├── setup_environment.py # ディレクトリ作成・環境初期化
│ │ ├── model_loader.py # モデル・トークナイザーのロード
│ │ ├── data_loader.py # JSON形式データの読み込み
│ │ ├── preprocessing.py # データの前処理・トークナイズ
│ │ ├── lora_setup.py # LoRA設定・適用
│ │ ├── train.py # Hugging Face Trainerでの学習
│ │ ├── save_model.py # Fine-tunedモデルの保存
│ │ ├── chatbot.py # 推論パイプライン・チャット関数
│ │ └── gradio_app.py # Web UI（Gradio）でのチャットボット
│ │
│ └── mistral-chat(v3)
│    ├── 01_setup_environment.py
│    ├── 02_load_model.py
│    ├── 03_preprocess_data.py
│    ├── 04_configure_lora.py
│    ├── 05_train.py
│    ├── 06_save_model.py
│    ├── 07_test_inference.py
│    ├── 08_launch_gradio_v3.py        統合版UI
│    ├── 09_evaluation_system.py       自動評価
│    ├── 10_data_augmentation.py       データ拡張
│    ├── 11_visualize_results.py       可視化
│    ├── 12_hyperparameter_sweep.py    実験管理
│    ├── utils.py
│    └── utils_v3.py                   v3専用ユーティリティ
│
├── data/ # 学習・評価用データ
│ └── training_data.json
├── results/ # 実験結果・生成サンプル・評価ログ
│ └── chat_log.txt
├── README.md
└── requirements.txt


### 各フォルダ・ファイルの概要

| フォルダ/ファイル | 役割 |
|------------------|------|
| `colab/`         | Notebookでの実験・開発ログの記録。テスト対話もここに保存 |
| `src/`           | 再利用可能なコード一式。学習・推論・Web UIを含む |
| `data/`          | 学習用・評価用データを格納 |
| `results/`       | ポートフォリオ提出用の整形済み生成ログ・サンプル |
| `requirements.txt` | 必要ライブラリ・バージョン管理 |
| `README.md`      | プロジェクト概要、実行手順、成果物の説明 |

---

## Technical Details

### Model Specifications

- **Base Model:** Mistral-7B-v0.1 (7.3B parameters)
- **Quantization:** 8-bit (bitsandbytes)
- **LoRA Config:**
  - r: 16
  - lora_alpha: 32
  - target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
  - dropout: 0.05

### Hardware Requirements

- **Minimum:** 16GB RAM, 8GB VRAM (GPU)
- **Recommended:** 32GB RAM, 16GB+ VRAM
- **Colab:** T4 GPU (無料枠で動作可能)

### Training Parameters

- Batch size: 4
- Gradient accumulation: 4
- Learning rate: 2e-4
- Epochs: 3
- Max sequence length: 512


## Setup & Usage

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Mistral-7B.git
cd Mistral-7B

2. Install dependencies

pip install -r requirements.txt

3. Prepare directories

from src.setup_environment import prepare_directories
prepare_directories()

4. Load dataset

from src.data_loader import load_json_dataset
dataset = load_json_dataset("data/training_data.json")

5. Preprocess dataset

from src.preprocessing import preprocess_dataset
from src.model_loader import load_mistral_model

tokenizer, _ = load_mistral_model()
processed_dataset = preprocess_dataset(dataset, tokenizer)

6. Apply LoRA

from src.lora_setup import apply_lora
from src.model_loader import load_mistral_model

tokenizer, model = load_mistral_model()
model = apply_lora(model)

7. Train model

from src.train import train_model
trainer = train_model(model, tokenizer, processed_dataset, output_dir="outputs")

8. Save model

from src.save_model import save_model_and_tokenizer
save_model_and_tokenizer(model, tokenizer, output_dir="mistral7b_finetuned")

9. Run Gradio Web UI
python src/gradio_app.py
 - ブラウザでチャットボットと対話可能
 - LoRA Fine-tunedモデルを即時使用

### License
MIT License © 2025 小山



