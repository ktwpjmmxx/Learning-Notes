# src/save_model.py
import os

def save_model_and_tokenizer(model, tokenizer, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Model saved to {output_dir}")
