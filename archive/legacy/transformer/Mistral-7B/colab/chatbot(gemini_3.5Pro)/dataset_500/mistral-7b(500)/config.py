%%writefile config.py
import torch

# ベースモデル（変更なし）
MODEL_NAME = "mistralai/Mistral-7B-v0.1"

# 🆕 新しいモデル名（200件のモデルと区別するため変更）
NEW_MODEL_NAME = "mistral-7b-custom-chat_500"

# 🆕 データパス（500件用のファイルを指定）
DATA_PATH = "data/train_data_500.json"

# --- 量子化設定 (BNB_CONFIG の追加) ---
BNB_CONFIG_DEFAULT = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": torch.bfloat16,
    "bnb_4bit_use_double_quant": False,
}
BNB_CONFIG = BNB_CONFIG_DEFAULT.copy()

# 学習パラメータ
TRAIN_PARAMS = {
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 2,
    "max_steps": -1, # エポック数で制御するため -1
    "learning_rate": 2e-4,
    "fp16": True,
    "logging_steps": 1,
    "output_dir": "outputs",
    "optim": "paged_adamw_8bit",
    
    # 🆕 エポック数（データが増えたので、しっかり学習させるために 1 → 3 に変更推奨）
    "num_train_epochs": 3
}

# LoRAパラメータ（変更なし）
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05