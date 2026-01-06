# src/gpt2_chatbot/inference.py
# ==============================================
# GPT-2 Inference Script
# ==============================================

from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# ============================
# Step 1: Load model and tokenizer
# ============================
model_path = "./gpt2-100-finetuned-final"  # fine-tunedモデルの保存先
print(f"Loading model from {model_path}...")

tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

print("✓ Model loaded successfully!")

# ============================
# Step 2: Inference
# ============================

prompt = "Human: Tell me a joke\n\nAssistant:"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

output_ids = model.generate(
    input_ids,
    max_length=100,
    temperature=0.8,
    top_p=0.9,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
)

generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("\n=== Generated Text ===")
print(generated_text)
