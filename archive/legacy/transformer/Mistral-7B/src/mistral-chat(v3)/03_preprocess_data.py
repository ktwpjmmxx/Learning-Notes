def preprocess_function(examples):
    """データを学習用フォーマットに変換"""
    inputs = examples["input"]
    outputs = examples["output"]
    
    # プロンプトフォーマット（システムプロンプトを追加）
    text_pairs = [
        f"User: {inp}\nBot: {out}" 
        for inp, out in zip(inputs, outputs)
    ]
    
    # トークン化
    model_inputs = tokenizer(
        text_pairs,
        max_length=512,
        truncation=True,
        padding="max_length",
    )
    
    # ラベル設定（cloneではなくリストのコピーを使用）
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    
    return model_inputs

# データセット処理
print("⚙️ Preprocessing dataset...")

tokenized_train = dataset["train"].map(
    preprocess_function, 
    batched=True,
    remove_columns=dataset["train"].column_names
)

tokenized_eval = dataset["test"].map(
    preprocess_function, 
    batched=True,
    remove_columns=dataset["test"].column_names
)

print("✅ Tokenization complete!")
print(f"   Training samples: {len(tokenized_train)}")
print(f"   Validation samples: {len(tokenized_eval)}")