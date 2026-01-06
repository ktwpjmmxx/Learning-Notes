from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from utils_v3 import print_model_info

# モデルをLoRA学習用に準備
print("⚙️ Preparing model for k-bit training...")
model = prepare_model_for_kbit_training(model)

# LoRA設定（改善版）
lora_config = LoraConfig(
    r=16,  # 8→16に増加（より多くの表現力）
    lora_alpha=32,
    target_modules=[
        "q_proj", 
        "k_proj",  # 追加
        "v_proj", 
        "o_proj"   # 追加
    ],
    lora_dropout=0.05,  # 0.1→0.05に減少（過学習を少し緩和）
    bias="none",
    task_type="CAUSAL_LM",
)

# LoRA適用
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print("✅ LoRA configuration applied!")

# 学習可能パラメータの詳細をJSON保存
trainable_params = {
    "total_params": sum(p.numel() for p in model.parameters()),
    "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    "trainable_percentage": 100 * sum(p.numel() for p in model.parameters() if p.requires_grad) / sum(p.numel() for p in model.parameters())
}

with open("results/training/lora_config.json", "w") as f:
    json.dump({
        "lora_r": lora_config.r,
        "lora_alpha": lora_config.lora_alpha,
        "target_modules": lora_config.target_modules,
        "lora_dropout": lora_config.lora_dropout,
        **trainable_params
    }, f, indent=2)

print_model_info(model)