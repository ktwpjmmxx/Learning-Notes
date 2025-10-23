# GPT-OSS Summarizer (Hugging Face / Colab版)

短いテキストを生成するための日本語対応の要約ツールです。Hugging Faceで公開されている**GPT-OSS**モデルを直接利用し、入力された長文を精度高く、かつ自然な表現で短くまとめます。

このプロジェクトは、**Google Colaboratory (Colab)** 環境での実行を前提としており、**外部APIキーは一切不要**です。モデルはColabのGPUを利用して直接実行されます。

---

## 主な機能

* **完全なオープンソース運用**: 外部APIサービスに依存せず、モデルを直接ロードして実行します。
* **日本語対応**: 日本語の複雑な文章構造を理解し、自然な要約を生成します。
* **Colab対応**: 複雑な環境構築なしで、すぐにブラウザ上で実行可能です。
* **設定可能な出力長**: 要約の長さ（短く/標準/長く）をオプションで指定できます。

---

## 技術スタック

* **言語**: Python
* **実行環境**: Google Colaboratory (GPU推奨)
* **LLMフレームワーク**: Hugging Face `transformers`
* **モデル**: **[使用する具体的なHugging Faceのモデル名]** (例: `rinna/japanese-gpt-neox-3.6b-instruction-sft`)
* **依存ライブラリ**: `transformers`, `torch`, `sentencepiece` (例), `requests`

---

## [実行方法（Google Colabでのクイックスタート）

### 1. ノートブックの準備とクローン

1.  Google Colabを開き、「**ファイル**」>「**新しいノートブック**」を作成します。
2.  「**ランタイム**」>「**ランタイムのタイプを変更**」から、ハードウェアアクセラレータを**GPU**に設定してください。（モデルのロードと推論には必須です）
3.  以下のコマンドを実行し、GitHubリポジトリをColab環境にクローンします。

    ```python
    !git clone [https://github.com/YourGitHubUsername/gpt-oss-summarizer.git](https://github.com/YourGitHubUsername/gpt-oss-summarizer.git)
    %cd gpt-oss-summarizer
    ```

### 2. 依存ライブラリのインストール

Colabのコードセルに以下のコマンドを入力し、必要なライブラリをインストールします。

```python
!pip install -r requirements.txt
```

### 3. モデルのロードと実行
このツールは、src/main.pyの実行時にHugging Faceからモデルを自動的にダウンロードし、ColabのGPUメモリにロードします。

外部APIキーの設定や環境変数の設定は不要です。

data/sample_input.txtを読み込んで要約する場合：

```python
# 最初の実行時にはモデルのダウンロードとロードに時間がかかります。
!python src/main.py --input_file data/sample_input.txt --length short
```

## 実行方法（Google Colabでのクイックスタート）

### 4. コマンド引数オプション

| オプション | 意味 | 規定値 | 例 |
| :---: | :---: | :---: | :---: |
| `--input_file` | 要約するテキストファイルのパス | なし | `--input_file article.txt` |
| `--length` | 出力する要約の長さ | `standard` | `--length short` (`short`, `standard`, `long` から選択) |
| `--output_file` | 要約結果を書き出すファイルパス | 標準出力 | `--output_file summary.txt` |

## 📂 フォルダ構成

```
gpt-oss-summarizer/
├── README.md             (このファイル)
├── requirements.txt      (依存ライブラリ一覧)
├── src/                  (ソースコード本体)
│   ├── main.py           (実行エントリーポイントとモデルのロード処理)
│   └── summarizer/       (要約ロジックやHugging Faceラッパー)
└── data/                 (サンプル入力データなど)
     └── sample\_input.txt (実験に利用するテキストサンプル)

```