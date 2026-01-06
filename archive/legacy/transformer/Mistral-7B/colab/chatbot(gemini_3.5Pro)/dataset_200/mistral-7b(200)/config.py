import torch

MODEL_NAME = "mistralai/Mistral-7B-v0.1"
NEW_MODEL_NAME = "mistral-7b-custom-chat"
DATA_PATH = "train_data.json"

# QLoRA設定 (BitsAndBytes)
BNB_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": torch.float16,
    "bnb_4bit_use_double_quant": False,
}

# LoRAパラメータ
LORA_R = 64
LORA_ALPHA = 16
LORA_DROPOUT = 0.1

# 学習パラメータ
TRAIN_PARAMS = {
    "num_train_epochs": 1,
    "per_device_train_batch_size": 4, # メモリ不足ならここを1か2にする
    "gradient_accumulation_steps": 1,
    "learning_rate": 2e-4,
    "weight_decay": 0.001,
    "fp16": False,
    "bf16": False,
    "max_grad_norm": 0.3,
    "max_steps": -1,
    "warmup_ratio": 0.03,
    "group_by_length": True,
    "lr_scheduler_type": "constant",
    "report_to": "tensorboard"
}