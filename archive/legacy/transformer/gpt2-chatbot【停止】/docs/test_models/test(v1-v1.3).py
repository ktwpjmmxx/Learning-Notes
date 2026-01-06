### Version1.0

import gradio as gr
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

print("モデルを読み込み中...")

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer_original = GPT2Tokenizer.from_pretrained("gpt2")
model_original = GPT2LMHeadModel.from_pretrained("gpt2")
model_original.to(device)
model_original.eval()

tokenizer_finetuned = GPT2Tokenizer.from_pretrained("./gpt2-finetuned-final")
model_finetuned = GPT2LMHeadModel.from_pretrained("./gpt2-finetuned-final")
model_finetuned.to(device)
model_finetuned.eval()

print("モデル読み込み完了")

def chat_original(message, history):
    prompt = f"Human: {message}\nAssistant:"
    inputs = tokenizer_original.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model_original.generate(
            inputs,
            max_length=len(inputs[0]) + 50,
            pad_token_id=tokenizer_original.eos_token_id,
            eos_token_id=tokenizer_original.eos_token_id,
        )
    
    full_response = tokenizer_original.decode(outputs[0], skip_special_tokens=True)
    response = full_response.split("Assistant:")[-1].strip()
    response = response.split('\n')[0].strip()
    return response

def chat_finetuned(message, history):
    prompt = f"Human: {message}\nAssistant:"
    inputs = tokenizer_finetuned.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model_finetuned.generate(
            inputs,
            max_length=len(inputs[0]) + 50,
            pad_token_id=tokenizer_finetuned.eos_token_id,
            eos_token_id=tokenizer_finetuned.eos_token_id,
        )
    
    full_response = tokenizer_finetuned.decode(outputs[0], skip_special_tokens=True)
    response = full_response.split("Assistant:")[-1].strip()
    response = response.split('\n')[0].strip()
    return response

with gr.Blocks(title="GPT-2 Before/After") as demo:
    gr.Markdown("# GPT-2 Fine-tuning: Before/After Comparison")
    gr.Markdown("Left: Original GPT-2 | Right: Fine-tuned GPT-2")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Original GPT-2 (Before)")
            chatbot_original = gr.ChatInterface(
                fn=chat_original,
                examples=["Hello, how are you?", "What is your name?", "Can you help me?"],
            )
        
        with gr.Column():
            gr.Markdown("### Fine-tuned GPT-2 (After)")
            chatbot_finetuned = gr.ChatInterface(
                fn=chat_finetuned,
                examples=["Hello, how are you?", "What is your name?", "Can you help me?"],
            )

demo.launch(share=True)

#### Version1.1

import gradio as gr
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

print("モデルを読み込み中...")

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-finetuned-final")
model = GPT2LMHeadModel.from_pretrained("./gpt2-finetuned-final")
model.to(device)
model.eval()

print("モデル読み込み完了")

def chat(message, history):
    prompt = f"Human: {message}\nAssistant:"
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=len(inputs[0]) + 60,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.5,
            no_repeat_ngram_size=3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_response.split("Assistant:")[-1].strip()
    response = response.split('\n')[0].strip()
    
    if not response:
        response = "I'm here to help! What would you like to know?"
    
    return response

interface = gr.ChatInterface(
    fn=chat,
    title="Fine-tuned GPT-2 Chatbot",
    description="GPT-2 fine-tuned on conversational data",
    examples=[
        "Hello, how are you?",
        "What is your name?",
        "Can you help me?",
        "Tell me something interesting",
        "What can you do?"
    ],
)

interface.launch(share=True)

**ver1.0からの改善**
- repetition_penalty=1.3 - 繰り返し防止
- no_repeat_ngram_size=3 - 同じフレーズの繰り返しを防ぐ
- do_sample=True - サンプリングを有効化
- te9mperature=0.8 - 多様性を追加

#### Version1.2

import gradio as gr
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

print("モデルを読み込み中...")

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-finetuned-final")
model = GPT2LMHeadModel.from_pretrained("./gpt2-finetuned-final")
model.to(device)
model.eval()

print("モデル読み込み完了")

def chat(message, history):
    prompt = f"\n\nHuman: {message}\n\nAssistant:"
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=len(inputs[0]) + 30,  # より短く制限
            temperature=0.7,  # 少し保守的に
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.6,  # さらに強化
            no_repeat_ngram_size=4,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=True,  # 早期終了を有効化
        )
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # "Assistant:"以降を抽出
    if "Assistant:" in full_response:
        response = full_response.split("Assistant:")[-1].strip()
    else:
        response = full_response[len(prompt):].strip()
    
    # 改行や不要な部分を削除
    response = response.split('\n')[0].strip()
    response = response.split('Human:')[0].strip()
    
    # 長すぎる場合は最初の文のみ
    if len(response) > 150:
        sentences = response.split('.')
        response = sentences[0] + '.' if sentences[0] else response[:100]
    
    if not response or len(response) < 3:
        response = "I'm here to help! Could you tell me more?"
    
    return response

interface = gr.ChatInterface(
    fn=chat,
    title="Fine-tuned GPT-2 Chatbot (Optimized)",
    description="GPT-2 with improved response control",
    examples=[
        "Hello, how are you?",
        "What is your name?",
        "Can you help me?",
        "Tell me a joke",
        "What's the weather like?"
    ],
)

interface.launch(share=True)

**ver1.1からの改善**
- max_length を30に短縮 - 長すぎる応答を防ぐ
- repetition_penalty=1.6 - さらに強化
- early_stopping=True - 適切なタイミングで終了
- 応答の後処理を強化 - 150文字以上は最初の文のみ抽出

**ポイント**
 - 大量の低品質データより、少量の高品質データ
 - データの内容が出力に直結 - 学習データに含まれるトピックがそのまま出力に現れ
