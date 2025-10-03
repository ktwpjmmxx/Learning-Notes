# Whisper - 音声認識AIの学習記録

OpenAIが開発した音声認識モデル「Whisper」についての学習記録。

## 目次
- [概要](#概要)
- [学習目的](#学習目的)
- [フォルダ構成](#フォルダ構成)
- [学習の進め方](#学習の進め方)
- [参考資料](#参考資料)
- [学習進捗](#学習進捗)

## 概要

Whisperは、OpenAIが開発した自動音声認識(ASR: Automatic Speech Recognition)モデル。68万時間以上の多言語音声データで訓練されており、高精度な文字起こしや音声翻訳が可能。

### 特徴
- 多言語対応（99言語）
- 高精度な音声認識
- ノイズに強い
- 句読点や大文字小文字の自動付与
- 複数のモデルサイズから選択可能

## 学習目的

1. Whisperの基本的な仕組みとアーキテクチャの理解
2. 実際の音声データを使った文字起こし実装
3. ファインチューニングによるカスタマイズ手法の習得
4. リアルタイム音声認識の実装
5. 実務で使えるレベルの音声認識システムの構築

## フォルダ構成

```
Whisper/
├── README.md                    # このファイル（概要・進捗管理）
├── progress.md          　　　　 # 学習進捗の詳細記録
├── notebooks/                   # 実験・学習用ノートブック
│   ├── 01_basic_usage.ipynb    # 基本的な使い方
│   ├── 02_model_comparison.ipynb  # モデルサイズごとの比較
│   ├── 03_fine_tuning.ipynb    # ファインチューニング
│   └── 04_real_time_transcription.ipynb  # リアルタイム文字起こし
├── notes/                       # 理論・実装メモ
│   ├── architecture.md          # アーキテクチャの理解
│   ├── model_sizes.md           # モデルサイズと性能比較
│   ├── training_data.md         # 訓練データについて
│   └── implementation_tips.md   # 実装時のTips・注意点
├── code/                        # サンプルコード
│   ├── basic_transcription.py   # 基本的な文字起こし
│   ├── batch_processing.py      # バッチ処理
│   └── audio_preprocessing.py   # 音声前処理
└── audio_samples/               # テスト用音声ファイル
    ├── sample_ja.mp3            # 日本語サンプル
    ├── sample_en.mp3            # 英語サンプル
    └── README.md                # サンプル音声の説明
```

## 参考資料

### 公式ドキュメント
- [OpenAI Whisper GitHub](https://github.com/openai/whisper)
- [Whisper論文](https://arxiv.org/abs/2212.04356)
- [Hugging Face Whisper](https://huggingface.co/openai/whisper-large-v3)

### 関連技術
- Transformer アーキテクチャ
- Encoder-Decoder モデル
- 音声信号処理

### 状況
- **開始日**: 2025/10/03
- **学習日数**: 1日目

#### メモ

- Whisperは大規模データで事前学習されているため、そのまま使っても高精度
- モデルサイズが大きいほど精度は上がるが、推論時間も増加する
- 日本語の認識精度は非常に高い
- （随時追加）

#### 関連フォルダ

- [GPT](../GPT/) - GPT系モデルの学習記録
- [References/papers](../References/papers/) - 論文まとめ
- [Experiments](../Experiments/) - 実験記録

---

**最終更新**: 2025/10/03