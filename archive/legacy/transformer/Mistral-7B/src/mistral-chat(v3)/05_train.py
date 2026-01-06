from transformers import TrainingArguments, Trainer, TrainerCallback
import numpy as np

# コールバック：各ステップでの損失を記録
class LossHistoryCallback(TrainerCallback):
    def __init__(self):
        self.train_losses = []
        self.eval_losses = []
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            if "loss" in logs:
                self.train_losses.append({
                    "step": state.global_step,
                    "loss": logs["loss"]
                })
            if "eval_loss" in logs:
                self.eval_losses.append({
                    "step": state.global_step,
                    "eval_loss": logs["eval_loss"]
                })

loss_callback = LossHistoryCallback()

# 学習設定（改善版）
training_args = TrainingArguments(
    output_dir="outputs",
    
    # バッチサイズ
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    
    # 学習ステップ
    num_train_epochs=3,  # max_stepsではなくエポック数で指定
    max_steps=-1,  # エポック数を優先
    
    # 学習率
    learning_rate=1e-4,  # 2e-4→1e-4に減少（よりゆっくり学習）
    lr_scheduler_type="cosine",  # コサインスケジューリング
    warmup_ratio=0.1,  # warmup_steps=5ではなく比率で指定
    
    # 評価
    evaluation_strategy="steps",  # ステップごとに評価
    eval_steps=20,  # 20ステップごと
    
    # ロギング
    logging_strategy="steps",
    logging_steps=10,
    
    # 保存
    save_strategy="steps",
    save_steps=50,
    save_total_limit=3,  # 2→3に増加
    load_best_model_at_end=True,  # 最良モデルをロード
    metric_for_best_model="eval_loss",  # 評価損失で判断
    
    # その他
    fp16=True,
    optim="paged_adamw_8bit",  # メモリ効率的なオプティマイザ
    report_to="none",
    
    # 過学習対策
    weight_decay=0.01,  # 正則化
)

print("✅ Training arguments configured!")
print(f"   Total training steps: ~{len(tokenized_train) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps) * training_args.num_train_epochs}")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,  # 評価データセット追加
    tokenizer=tokenizer,
    callbacks=[loss_callback]  # コールバック追加
)

print("🚀 Starting training...")
print("=" * 60)

# 学習開始
train_result = trainer.train()

print("\n" + "=" * 60)
print("✅ Training completed!")

# 学習結果の保存
trainer.save_model("outputs/final_model")
tokenizer.save_pretrained("outputs/final_model")

# 学習履歴の保存
from datetime import datetime

experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
training_history = {
    "experiment_id": experiment_id,  # 追加
    "timestamp": datetime.now().isoformat(),  # 追加
    "train_losses": loss_callback.train_losses,
    "eval_losses": loss_callback.eval_losses,
    "final_train_loss": train_result.training_loss,
    "total_steps": train_result.global_step,
}

with open(f"results/training/training_history_{experiment_id}.json", "w") as f:
    json.dump(training_history, f, indent=2)

# 学習曲線のプロット（簡易版）
print("\n📊 Training Summary:")
print(f"   Final Training Loss: {train_result.training_loss:.4f}")
if loss_callback.eval_losses:
    final_eval_loss = loss_callback.eval_losses[-1]["eval_loss"]
    print(f"   Final Validation Loss: {final_eval_loss:.4f}")
    print(f"   Overfitting Gap: {abs(train_result.training_loss - final_eval_loss):.4f}")