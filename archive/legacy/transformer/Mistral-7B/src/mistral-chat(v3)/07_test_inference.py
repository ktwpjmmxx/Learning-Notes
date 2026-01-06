import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import PeftModel

# メモリクリーンアップ
print("🧹 Cleaning up memory...")
try:
    del model, trainer
except:
    pass

torch.cuda.empty_cache()
gc.collect()
print("✅ Memory cleaned!")

# 設定
LOCAL_ADAPTER_DIR = "mistral7b_finetuned"
BASE_MODEL_ID = "mistralai/Mistral-7B-v0.1"

# 4bit量子化設定
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print("📥 Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

print("📥 Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, LOCAL_ADAPTER_DIR)

print("✅ Model loaded successfully!")

# 簡単なテスト
test_prompts = [
    "User: Hello, how are you?\nBot:",
    "User: Tell me a joke\nBot:",
    "User: What is AI?\nBot:",
]

print("\n🧪 Running quick tests...\n")

for prompt in test_prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    bot_response = result.split("Bot:")[-1].strip().split("\n")[0]
    
    print(f"Prompt: {prompt.split('Bot:')[0]}")
    print(f"Response: {bot_response}\n")