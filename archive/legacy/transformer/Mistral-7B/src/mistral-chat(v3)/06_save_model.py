OUTPUT_DIR = "mistral7b_finetuned"

# LoRAアダプタのみ保存（軽量）
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"✅ Model saved to {OUTPUT_DIR}")

# モデル情報の保存
model_info = {
    "model_name": MODEL_NAME,
    "adapter_path": OUTPUT_DIR,
    "lora_r": lora_config.r,
    "lora_alpha": lora_config.lora_alpha,
    "target_modules": lora_config.target_modules,
    "final_train_loss": train_result.training_loss,
    "total_training_steps": train_result.global_step,
}

with open(f"{OUTPUT_DIR}/model_info.json", "w") as f:
    json.dump(model_info, f, indent=2)

print("📄 Model info saved!")