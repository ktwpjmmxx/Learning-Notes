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