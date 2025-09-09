# GPT Learning Project

このプロジェクトは、GPT（Generative Pre-trained Transformer）とTransformerアーキテクチャについて体系的に学習するためのリポジトリです。理論的な理解と実践的な実装の両面から、深層学習における最重要技術の一つを習得することを目的としています。

## 🎯 学習目標

- GPTとTransformerの基本概念と動作原理の理解
- 実際のコードを通じた実装レベルでの理解
- ファインチューニング手法の習得
- 小規模チャットボット開発による実践的スキルの向上

## 📁 プロジェクト構成

### 🗂️ `notebooks/`
実践的な学習とコード実行のためのJupyter Notebookを格納

#### `01_basics/`
GPTとトークナイザーの基礎学習
- **`gpt_test.ipynb`**: GPTモデルの基本動作確認とテストコード
  - モデルのロードと初期化
  - 基本的なテキスト生成
  - パラメータの確認と調整
- **`tokenizer_exploration.ipynb`**: トークナイザーの仕組みと動作探索
  - 各種トークナイザーの比較
  - エンコード・デコード処理の理解
  - 特殊トークンの扱い

#### `02_chatbot_project/` *(予定)*
GPT2を使用した小規模チャットボット開発プロジェクト
- **`gpt2_setup.ipynb`**: GPT2モデルのセットアップと環境構築
- **`chatbot_implementation.ipynb`**: チャットボットのメイン実装
- **`model_analysis.ipynb`**: GPT2の内部構造とレイヤー分析
- **`interaction_testing.ipynb`**: 対話テストと品質評価

### 📝 `notes/`
理論的な学習内容とナレッジベース（Markdown形式）

- **`fundamentals.md`**: GPTの基礎概念とアーキテクチャ
  - Transformerの基本構造
  - Self-Attentionメカニズム
  - 位置エンコーディング
  - GPTの進化史（GPT-1からGPT-4まで）

- **`transformer-deep-dive.md`**: Transformerアーキテクチャの詳細解説
  - Multi-Head Attentionの数学的理解
  - Encoder-Decoderの違い
  - 計算複雑度とスケーラビリティ

- **`implementation-notes.md`**: 実装時の注意点とベストプラクティス
  - 効率的なコーディング手法
  - デバッグのコツ
  - パフォーマンス最適化

- **`finetuning-strategies.md`**: ファインチューニングの手法と戦略
  - データ準備とプリプロセシング
  - ハイパーパラメータ調整
  - 評価指標と検証方法

- **`chatbot-development.md`** *(予定)*: チャットボット開発で学んだ知見
  - 対話システムの設計パターン
  - レスポンス品質の改善手法
  - 実装上の課題と解決策

## 🚀 使用方法

### 環境要件
```bash
Python 3.8+
torch>=1.9.0
transformers>=4.0.0
numpy
pandas
matplotlib
jupyter
```

### セットアップ
```bash
# 依存関係のインストール
pip install torch transformers numpy pandas matplotlib jupyter

# Jupyter Notebookの起動
jupyter notebook
```

### Google Colab での実行
各notebookはGoogle Colabでの実行を想定して作成されています。
```python
# Colabでの基本セットアップ
!pip install transformers torch
```

## 📚 学習の進め方

### 推奨学習順序
1. **理論学習**: `notes/fundamentals.md` → `notes/transformer-deep-dive.md`
2. **基礎実践**: `notebooks/01_basics/` 内のnotebook実行
3. **応用実践**: `notebooks/02_chatbot_project/` でのプロジェクト開発
4. **発展学習**: `notes/finetuning-strategies.md` と実際のファインチューニング

### 学習のコツ
- 理論と実践を交互に進める
- コードは必ず手を動かして実行する
- 疑問点は即座に調べて理解する
- 学んだ内容を自分の言葉でまとめる

## 🛠️ 開発環境

- **Python**: 3.8+
- **主要ライブラリ**: PyTorch, Transformers, NumPy
- **開発環境**: Jupyter Notebook / Google Colab
- **バージョン管理**: Git

## 🔗 参考資料

### 論文
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2 paper

### 書籍・記事
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)

### 公式ドキュメント
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

---
