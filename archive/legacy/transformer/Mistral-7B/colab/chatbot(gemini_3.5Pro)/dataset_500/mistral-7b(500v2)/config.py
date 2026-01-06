import torch

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
NEW_MODEL_NAME = "mistral-7b-custom-chat_exp004"
DATA_PATH = "data/train_data_500v2.json"

# ==========================================
# 量子化設定 (BitsAndBytes)
# ==========================================
BNB_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": torch.float16,  # PyTorch 2.1では float16を使用
    "bnb_4bit_use_double_quant": True,
}

# ==========================================
# 学習パラメータ (過学習対策強化版)
# ==========================================
TRAIN_PARAMS = {
    "output_dir": "outputs_exp004",
    
    # エポック数と学習率
    "num_train_epochs": 0.5,  # データを半周のみ
    "learning_rate": 5e-6,  # 低学習率で過学習を抑制
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.1,
    
    # バッチサイズ
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    
    # 評価とロギング
    "eval_strategy": "steps",  # evaluation_strategy から変更
    "eval_steps": 10,
    "logging_steps": 5,
    "save_strategy": "steps",
    "save_steps": 10,
    "save_total_limit": 3,
    
    # Early Stopping
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
    
    "per_device_eval_batch_size": 1,
    "report_to": "none",
    
    # GPU設定 (PyTorch 2.1 + float16)
    "fp16": True,
    "bf16": False,
    
    # 正則化
    "optim": "paged_adamw_8bit",
    "weight_decay": 0.01,
    "max_grad_norm": 0.5,
    "gradient_checkpointing": True,
}

# ==========================================
# LoRAパラメータ
# ==========================================
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1  # 過学習対策のため高めに設定