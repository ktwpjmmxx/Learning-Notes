import torch

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
NEW_MODEL_NAME = "mistral-7b-custom-chat_exp005"  # exp004 → exp005
DATA_PATH = "data/train_data_500v3.json"  # v2 → v3

# ==========================================
# 量子化設定 (BitsAndBytes) - 変更なし
# ==========================================
BNB_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": torch.float16,
    "bnb_4bit_use_double_quant": True,
}

# ==========================================
# 学習パラメータ (Exp-005: 日本語出力強化版)
# ==========================================
TRAIN_PARAMS = {
    "output_dir": "outputs_exp005",  # exp004 → exp005
    
    # === エポック数と学習率の調整 ===
    "num_train_epochs": 2,      # 0.5 → 2 (4倍に増加)
    "learning_rate": 1e-5,      # 5e-6 → 1e-5 (2倍に増加)
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.1,
    
    # === バッチサイズ ===
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,  # 4 → 8 (安定性向上)
    
    # === 評価とロギング ===
    "eval_strategy": "steps",
    "eval_steps": 10,
    "logging_steps": 5,
    "save_strategy": "steps",
    "save_steps": 10,
    "save_total_limit": 3,
    
    # === Early Stopping ===
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
    
    "per_device_eval_batch_size": 1,
    "report_to": "none",
    
    # === GPU設定 ===
    "fp16": True,
    "bf16": False,
    
    # === 正則化 ===
    "optim": "paged_adamw_8bit",
    "weight_decay": 0.01,
    "max_grad_norm": 0.5,
    "gradient_checkpointing": True,
}

# ==========================================
# LoRAパラメータ - 変更なし
# ==========================================
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1