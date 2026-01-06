from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
import torch

MODEL_NAME = "mistralai/Mistral-7B-v0.1"

# 4bit量子化設定
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

print("📥 Loading model and tokenizer...")

# トークナイザー読み込み
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # 明示的に指定

# モデル読み込み
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

model.config.pad_token_id = tokenizer.eos_token_id
model.config.use_cache = False

print("✅ Model loaded successfully!")
print(f"📊 Model size: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")

# データセット読み込み
print("\n📥 Loading dataset...")
dataset = load_dataset("json", data_files="training_data.json")

# データセットを学習用と評価用に分割（85/15）
dataset = dataset["train"].train_test_split(test_size=0.15, seed=42)

print(f"✅ Dataset loaded!")
print(f"   Training samples: {len(dataset['train'])}")
print(f"   Validation samples: {len(dataset['test'])}")