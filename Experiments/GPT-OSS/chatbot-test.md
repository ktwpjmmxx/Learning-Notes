# GPT-OSS Chatbot（Google Colab + Gradio GUI + ファインチューニング取扱説明書）

## 1. 概要

このプロジェクトは、**Google Colab 上で稼働する GPT-OSS ベースのチャットボット** を構築することを目的としています。  
ユーザーが入力したプロンプトに対して自動的に応答を生成し、Gradioを用いてGUIで可視化します。  

特徴:
- GPT-OSS 互換モデルを利用可能（例: `"gpt-oss/gpt2-mini"` など）
- Colab GPU上で動作（半精度 float16 に対応）
- 単発応答型（履歴を保持しないためVRAM負荷を軽減）
- ファインチューニング用の雛形コードを含み、拡張が可能

---

## 2. コード（Colabでそのまま実行可能）

```python
# ==============================
# 必要ライブラリのインストール
# ==============================
!pip install transformers gradio datasets --quiet

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import gradio as gr

# ==============================
# 2.1 モデル・トークナイザーのロード
# ==============================
# モデル名は任意のGPT-OSSモデルに変更可能
MODEL_NAME = "gpt-oss/gpt2-mini"  

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",           
    torch_dtype=torch.float16    # VRAM節約のため半精度
)

# ==============================
# 2.2 Chatbotパイプライン作成
# ==============================
chatbot = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=256,             # 応答長を短めに設定（負荷軽減）
    temperature=0.7,            # 創造性の度合い（低いほど保守的、高いほど自由）
    top_p=0.9,                  # nucleus samplingによる確率的サンプリング
    repetition_penalty=1.1      # 同じ表現の繰り返し抑制
)

# ==============================
# 2.3 単発応答関数（履歴なし）
# ==============================
def chat_with_bot(prompt: str):
    """
    prompt: ユーザー入力（履歴は保持せず単発で応答）
    """
    response = chatbot(prompt)[0]["generated_text"]
    return response

# ==============================
# 2.4 Gradio GUI構築
# ==============================
iface = gr.Interface(
    fn=chat_with_bot,
    inputs="text",
    outputs="text",
    title="GPT-OSS Chatbot",
    description="プロンプトを入力するとチャットボットが応答します（履歴なし・軽量動作）"
)

# Colab上でGUIを起動
iface.launch()

# ==============================
# 3. ファインチューニング手順（雛形）
# ==============================

"""
ファインチューニングを行う場合は以下の手順を追加してください。

1. データセット準備
   - Hugging Face Datasets または独自データをJSON/CSVで用意
   - 形式例: {"prompt": "こんにちは", "response": "こんにちは！元気ですか？"}

2. トークナイズ処理
   - dataset = dataset.map(lambda e: tokenizer(e["prompt"] + e["response"], truncation=True))

3. TrainingArguments 設定
   from transformers import Trainer, TrainingArguments
   training_args = TrainingArguments(
       output_dir="./fine_tuned_model",
       per_device_train_batch_size=2,
       per_device_eval_batch_size=2,
       num_train_epochs=3,
       logging_steps=50,
       save_steps=200,
       fp16=True
   )

4. Trainer 実行
   trainer = Trainer(
       model=model,
       args=training_args,
       train_dataset=train_dataset,
       eval_dataset=eval_dataset,
       tokenizer=tokenizer
   )
   trainer.train()
   trainer.save_model("./fine_tuned_model")

5. ファインチューニング後の利用
   model = AutoModelForCausalLM.from_pretrained("./fine_tuned_model")
   tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")
   chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer)
"""

# ==============================
# 4. 注意点（Colab向け）
# ==============================
"""
- GPU設定: 「ランタイム」→「ランタイムのタイプを変更」→「GPU」を選択
- VRAM制限: 無料版Colabは約12GB。大規模モデルはOOMの可能性あり
- 履歴なし設計: 文脈保持を行わないため負荷軽減（必要なら履歴型に変更可能）
- セッション制限: 無料Colabは数時間で切断。長時間学習は有料版推奨
- 学習データ品質: ファインチューニングの応答品質はデータ次第
"""
