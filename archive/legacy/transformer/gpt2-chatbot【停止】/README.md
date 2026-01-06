# GPT-2 Fine-tuned Chatbot

GPT-2を会話データでファインチューニングしたチャットボットの実験記録。

## 概要
- **ベースモデル**: GPT-2 (124M parameters)
- **学習データ**: 100件の高品質な英語会話
- **学習環境**: Google Colab (T4 GPU)

## クイックスタート

### 1. セットアップ
```bash
pip install -r requirements.txt
```

### 2. ファインチューニング
```bash
python scripts/01_train.py
```

### 3. チャットボット起動
```bash
python scripts/03_chatbot_ui.py
```

## ディレクトリ構成
- `data/`: 学習用会話データ
- `scripts/`: 実行用Pythonスクリプト
- `results/`: 実験結果とチャットログ
- `docs/`: 詳細ドキュメント

## 実験結果
- [評価結果](results/evaluation_summary.md)
- [チャットログサンプル](results/chat_logs/)
- [パラメータ調整履歴](docs/tuning_log.md)

## 学んだこと
詳細は [lessons_learned.md](docs/lessons_learned.md) を参照。