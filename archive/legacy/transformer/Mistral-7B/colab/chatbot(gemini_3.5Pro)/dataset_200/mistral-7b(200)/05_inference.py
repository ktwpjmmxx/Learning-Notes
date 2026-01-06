import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import config
from utils import format_prompt

def chat():
    print("推論用モデルロード中...")
    bnb_config = BitsAndBytesConfig(**config.BNB_CONFIG)
    base_model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME, quantization_config=bnb_config, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 学習済みアダプタを結合
    model = PeftModel.from_pretrained(base_model, config.NEW_MODEL_NAME)
    model.eval()

    print("\n=== チャットボット起動 (終了は exit と入力) ===")
    while True:
        txt = input("You: ")
        if txt.lower() in ["exit", "quit"]: break
        
        prompt = format_prompt(txt)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=200, do_sample=True, top_p=0.9, temperature=0.7
            )
        
        res = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Bot: {res.split('[/INST]')[-1].strip()}\n")

if __name__ == "__main__":
    chat()