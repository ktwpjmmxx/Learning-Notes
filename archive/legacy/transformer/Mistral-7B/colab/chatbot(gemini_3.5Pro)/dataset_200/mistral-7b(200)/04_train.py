from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import config
from utils import generate_prompt_for_training
import importlib.util

# 03_load_model.py をインポート
spec = importlib.util.spec_from_file_location("mod", "03_load_model.py")
load_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(load_mod)

def train():
    print("モデルロード開始...")
    model, tokenizer, peft_config = load_mod.get_model_and_tokenizer(inference_mode=False)
    dataset = load_dataset("json", data_files=config.DATA_PATH, split="train")
    
    training_args = TrainingArguments(output_dir="./results", **config.TRAIN_PARAMS)

    print("学習開始...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        formatting_func=generate_prompt_for_training,
        args=training_args
        # 余分な引数 (tokenizer, max_length, packingなど) を全て削除しました
    )
    
    trainer.train()
    
    print(f"モデル保存中: {config.NEW_MODEL_NAME}")
    trainer.model.save_pretrained(config.NEW_MODEL_NAME)
    tokenizer.save_pretrained(config.NEW_MODEL_NAME)
    print("完了")

if __name__ == "__main__":
    train()
