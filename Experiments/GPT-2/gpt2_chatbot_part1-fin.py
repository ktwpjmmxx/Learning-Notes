# --- Colab用：GPT-2 チャットボット（完全版） ---

!pip install transformers torch gradio --quiet

import gradio as gr
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import re
import random

# --- モデルロード ---
model_name = "gpt2-medium"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# --- 応答関数（多様性版） ---
def respond_diverse(prompt):
    """多様性を確保した名前生成"""
    
    # 複数の異なるプロンプトパターンをランダム選択
    prompts = [
        "My name is",
        "I am",
        "Call me",
        "Hi, I'm", 
        "Hello, my name is"
    ]
    
    selected_prompt = random.choice(prompts)
    inputs = tokenizer.encode(selected_prompt, return_tensors="pt").to(device)
    
    # attention_maskを作成（warning対策）
    attention_mask = torch.ones_like(inputs).to(device)
    
    # 温度を毎回少しずつ変える
    temp = random.uniform(0.8, 1.5)
    
    outputs = model.generate(
        inputs,
        attention_mask=attention_mask,
        max_new_tokens=4,
        do_sample=True,
        top_k=random.randint(30, 100),
        top_p=random.uniform(0.85, 0.95),
        temperature=temp,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.5
    )
    
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    text = text.replace(selected_prompt, "").strip()
    
    # 名前抽出（フィルター強化）
    clean_text = re.sub(r'[.,!?;:"\']', '', text)
    words = clean_text.split()
    
    if words:
        name = words[0]
        
        # NGワードリスト
        ng_words = ['dr', 'what', 'the', 'and', 'but', 'pleasure', 'fetch', 'xxx', 'sex']
        
        # より厳格な名前チェック
        if (name and 
            name[0].isupper() and 
            name.isalpha() and 
            len(name) > 2 and  # 最低3文字
            len(name) < 15 and 
            name.lower() not in ng_words and  # NGワード除外
            not name.lower().startswith(('dr', 'mr', 'ms'))  # 敬称除外
           ):
            return name
    
    # 最終フォールバック（重複回避のため増量）
    fallback_names = [
        "Sam", "Jordan", "Casey", "Taylor", "Morgan", "Avery", "Riley", "Quinn",
        "Blake", "Drew", "Sage", "River", "Hayden", "Emery", "Rowan", "Finley",
        "Kai", "Lane", "Reese", "Cameron", "Parker", "Skyler", "Logan", "Peyton"
    ]
    return random.choice(fallback_names)

# --- Gradio インターフェース ---
iface = gr.Interface(
    fn=respond_diverse,
    inputs="text",
    outputs="text",
    title="GPT-2 Medium チャットボット（多様性版）",
    description="同じ質問でも毎回違う名前を生成します"
)

iface.launch()

# --- テスト実行 ---
prompt = "Could you tell me about your name ?"

print("=== 多様性チャレンジ 10回連続テスト ===")
for i in range(10):
    output = respond_diverse(prompt)
    print(f"--- {i+1}回目 ---")
    print(output)