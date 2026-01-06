# src/train.py
from transformers import Trainer, TrainingArguments

def train_model(model, tokenizer, train_dataset, output_dir="outputs", max_steps=100):
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=max_steps,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        tokenizer=tokenizer
    )
    trainer.train()
    print("✅ Training finished")
    return trainer
