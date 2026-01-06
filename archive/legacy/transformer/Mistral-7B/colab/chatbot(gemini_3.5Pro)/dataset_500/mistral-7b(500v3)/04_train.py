import torch
import os
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerCallback,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import config
from utils import generate_prompt_for_training

# タイムアウト設定
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '900'

class OverfittingDetectionCallback(TrainerCallback):
    """過学習検出コールバック"""
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.train_losses = []
        self.eval_losses = []
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if 'loss' in logs:
                self.train_losses.append(logs['loss'])
            if 'eval_loss' in logs:
                self.eval_losses.append(logs['eval_loss'])
                
                if len(self.train_losses) > 0:
                    train_loss = self.train_losses[-1]
                    eval_loss = self.eval_losses[-1]
                    gap = eval_loss - train_loss
                    
                    print(f"\n[過学習チェック] Train: {train_loss:.4f}, Eval: {eval_loss:.4f}, Gap: {gap:.4f}")
                    
                    if gap > self.threshold:
                        print(f"⚠️ 警告: 過学習の兆候 (Gap > {self.threshold})")

def train():
    print("=" * 60)
    print("Mistral-7B 学習開始")
    print("=" * 60)
    
    # ==========================================
    # 1. トークナイザーロード
    # ==========================================
    print("\n[1/5] トークナイザーをロード中...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config.MODEL_NAME,
            trust_remote_code=True,
            use_fast=True  # 高速トークナイザーを使用
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        print("✓ トークナイザーロード完了")
    except Exception as e:
        print(f"✗ エラー: {e}")
        return

    # ==========================================
    # 2. モデルロード (量子化)
    # ==========================================
    print("\n[2/5] モデルをロード中...")
    print("※ 初回は15-30分かかります\n")
    
    try:
        bnb_config = BitsAndBytesConfig(**config.BNB_CONFIG)
        
        model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        print("\n✓ モデルロード完了")
        
    except Exception as e:
        print(f"\n✗ エラー: {e}")
        print("\n対処方法:")
        print("1. ランタイムを再起動")
        print("2. キャッシュクリア: !rm -rf ~/.cache/huggingface/hub/*")
        print("3. 再実行")
        return

    # ==========================================
    # 3. データロードとフォーマット
    # ==========================================
    print("\n[3/5] データをロード中...")
    try:
        dataset = load_dataset("json", data_files=config.DATA_PATH, split="train")
        
        # データをMistral形式にフォーマット
        def format_data(example):
            prompt = generate_prompt_for_training(example)
            return {"text": prompt}
        
        dataset = dataset.map(format_data, remove_columns=dataset.column_names)
        
        split_dataset = dataset.train_test_split(test_size=0.15, seed=42)
        train_data = split_dataset["train"]
        val_data = split_dataset["test"]
        
        print(f"✓ 学習データ: {len(train_data)}")
        print(f"✓ 検証データ: {len(val_data)}")
    except Exception as e:
        print(f"✗ エラー: {e}")
        return

    # ==========================================
    # 4. LoRA設定
    # ==========================================
    print("\n[4/5] LoRA設定を適用中...")
    try:
        model = prepare_model_for_kbit_training(model)
        
        peft_config = LoraConfig(
            r=config.LORA_R,
            lora_alpha=config.LORA_ALPHA,
            lora_dropout=config.LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        )
        model = get_peft_model(model, peft_config)
        
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"✓ 学習可能パラメータ: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    except Exception as e:
        print(f"✗ エラー: {e}")
        return

    # ==========================================
    # 5. トレーナー初期化と学習
    # ==========================================
    print("\n[5/5] トレーナーを初期化中...")
    
    training_args = TrainingArguments(**config.TRAIN_PARAMS)

    # 最新版SFTTrainerに対応（最小構成）
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        processing_class=tokenizer,
        args=training_args,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3,
                early_stopping_threshold=0.001
            ),
            OverfittingDetectionCallback(threshold=0.5)
        ]
    )

    print("\n" + "=" * 60)
    print("学習開始")
    print(f"  学習率: {config.TRAIN_PARAMS['learning_rate']}")
    print(f"  エポック: {config.TRAIN_PARAMS['num_train_epochs']}")
    print(f"  実効バッチサイズ: {config.TRAIN_PARAMS['per_device_train_batch_size'] * config.TRAIN_PARAMS['gradient_accumulation_steps']}")
    print("=" * 60 + "\n")
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n学習を中断しました")
    except Exception as e:
        print(f"\n✗ エラー: {e}")
        import traceback
        traceback.print_exc()
        return

    # ==========================================
    # 保存
    # ==========================================
    print("\n" + "=" * 60)
    print("モデルを保存中...")
    try:
        trainer.model.save_pretrained(config.NEW_MODEL_NAME)
        tokenizer.save_pretrained(config.NEW_MODEL_NAME)
        print("✓ 保存完了")
        
        if hasattr(trainer.state, 'log_history'):
            print("\n=== 学習統計 ===")
            train_losses = [log['loss'] for log in trainer.state.log_history if 'loss' in log]
            eval_losses = [log['eval_loss'] for log in trainer.state.log_history if 'eval_loss' in log]
            
            if train_losses:
                print(f"最終 Train Loss: {train_losses[-1]:.4f}")
            if eval_losses:
                print(f"最終 Eval Loss: {eval_losses[-1]:.4f}")
                print(f"最良 Eval Loss: {min(eval_losses):.4f}")
        
        print("\n学習完了!")
        print("=" * 60)
    except Exception as e:
        print(f"✗ エラー: {e}")

if __name__ == "__main__":
    train()