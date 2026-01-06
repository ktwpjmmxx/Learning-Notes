# Mistral-7B Colab Chatbot Project

Google Colab (T4 GPU) 上で Mistral-7B を QLoRA で学習させ、チャットボット化するプロジェクトです。

## セットアップと実行手順

このプロジェクトは以下の順序で実行するように設計されています。

1. **環境構築**
   `python 01_install_deps.py` を実行し、ライブラリをインストールします。
   （Colabの場合はセルで `!python 01_install_deps.py`）

2. **データセット作成**
   `python 00_generate_data.py` を実行します。
   これを行うと、200件の会話データが含まれた `train_data.json` が自動生成されます。

3. **データ確認**
   `python 02_process_data.py` を実行し、データが正しく読み込めるか確認します。

4. **学習 (Fine-Tuning)**
   `python 04_train.py` を実行します。
   学習が完了すると `mistral-7b-custom-chat` フォルダにアダプタが保存されます。

5. **推論 (チャット)**
   `python 05_inference.py` を実行し、ボットと会話します。

## 注意事項
- Google Colabのランタイム設定で「T4 GPU」を選択してください。
- 学習中にメモリ不足になる場合は `config.py` のバッチサイズを下げてください。