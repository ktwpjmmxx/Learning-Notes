!pip install datasets transformers torch accelerate -q

from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
import torch

# ステップ1: データセット読み込み
print("=" * 60)
print("ステップ1: データセットの読み込み")
print("=" * 60)

try:
    dataset = load_dataset("Anthropic/hh-rlhf")
    print(f"✓ 学習データ数: {len(dataset['train'])} 件")
    print(f"✓ テストデータ数: {len(dataset['test'])} 件")
except Exception as e:
    print(f"✗ エラー: {e}")
    raise

# ステップ2: トークナイザー準備
print("\n" + "=" * 60)
print("ステップ2: トークナイザー準備")
print("=" * 60)

try:
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    print("✓ トークナイザー準備完了")
except Exception as e:
    print(f"✗ エラー: {e}")
    raise

# ステップ3: データ選択
print("\n" + "=" * 60)
print("ステップ3: データ選択")
print("=" * 60)

try:
    train_dataset = dataset["train"].select(range(1000))
    eval_dataset = dataset["test"].select(range(100))
    print(f"✓ train_dataset: {len(train_dataset)} 件")
    print(f"✓ eval_dataset: {len(eval_dataset)} 件")
except Exception as e:
    print(f"✗ エラー: {e}")
    raise

# ステップ4: データ整形関数
print("\n" + "=" * 60)
print("ステップ4: データ整形")
print("=" * 60)

def preprocess_function(examples):
    texts = examples["chosen"]
    result = tokenizer(
        texts,
        truncation=True,
        max_length=512,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

try:
    print("データを整形中...")
    tokenized_train = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    print(f"✓ tokenized_train作成完了: {len(tokenized_train)} 件")
    
    tokenized_eval = eval_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=eval_dataset.column_names
    )
    print(f"✓ tokenized_eval作成完了: {len(tokenized_eval)} 件")
    
    # 変数が存在するか確認
    print(f"\n確認: tokenized_train is defined = {tokenized_train is not None}")
    print(f"確認: tokenized_eval is defined = {tokenized_eval is not None}")
    
except Exception as e:
    print(f"✗ データ整形でエラー: {e}")
    import traceback
    traceback.print_exc()
    raise

# ステップ5: モデル準備
print("\n" + "=" * 60)
print("ステップ5: モデル準備")
print("=" * 60)

try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用デバイス: {device}")
    
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.to(device)
    print("✓ モデル準備完了")
except Exception as e:
    print(f"✗ エラー: {e}")
    raise

# ステップ6: 学習設定
print("\n" + "=" * 60)
print("ステップ6: 学習設定")
print("=" * 60)

try:
    training_args = TrainingArguments(
        output_dir="./gpt2-finetuned",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",
    )
    print("✓ 学習設定完了")
except Exception as e:
    print(f"✗ エラー: {e}")
    raise

# ステップ7: Trainer作成
print("\n" + "=" * 60)
print("ステップ7: Trainer作成")
print("=" * 60)

try:
    # 再度確認
    print(f"tokenized_train exists: {'tokenized_train' in dir()}")
    print(f"tokenized_eval exists: {'tokenized_eval' in dir()}")
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
    )
    print("✓ Trainer作成完了")
except Exception as e:
    print(f"✗ Trainer作成でエラー: {e}")
    import traceback
    traceback.print_exc()
    raise

# ステップ8: ファインチューニング実行
print("\n" + "=" * 60)
print("ステップ8: ファインチューニング実行")
print("=" * 60)

trainer.train()

print("\n✓ ファインチューニング完了！")

# モデル保存
model.save_pretrained("./gpt2-finetuned-final")
tokenizer.save_pretrained("./gpt2-finetuned-final")
print("✓ モデル保存完了: ./gpt2-finetuned-final")