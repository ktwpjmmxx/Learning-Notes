# src/gpt2_chatbot/train.py
# ==============================================
# GPT-2 Fine-tuning Script (for small chatbot)
# ==============================================

from datasets import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
import torch

# ============================
# Step 1: Create conversation dataset
# ============================
print("=" * 60)
print("Step 1: Creating conversation dataset")
print("=" * 60)

conversations = [
    
]

dataset = Dataset.from_dict({"text": conversations})
print(f"Total conversations: {len(conversations)}")

# ============================
# Step 2: Preprocessing
# ============================
print("\n" + "=" * 60)
print("Step 2: Tokenizing dataset")
print("=" * 60)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def preprocess_function(examples):
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

print("Tokenizing...")
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset.column_names
)

train_size = int(0.85 * len(tokenized_dataset))
tokenized_train = tokenized_dataset.select(range(train_size))
tokenized_eval = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))

print(f"Training: {len(tokenized_train)} | Evaluation: {len(tokenized_eval)}")

# ============================
# Step 3: Fine-tuning
# ============================
print("\n" + "=" * 60)
print("Step 3: Fine-tuning GPT-2")
print("=" * 60)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.to(device)

training_args = TrainingArguments(
    output_dir="./gpt2-100-finetuned",
    num_train_epochs=10,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=20,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=20,
    save_steps=20,
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
)

print("\nFine-tuning started!")
print("=" * 60)
trainer.train()

print("\nFine-tuning completed!")
model.save_pretrained("./gpt2-100-finetuned-final")
tokenizer.save_pretrained("./gpt2-100-finetuned-final")
print("✓ Model saved at ./gpt2-100-finetuned-final")
