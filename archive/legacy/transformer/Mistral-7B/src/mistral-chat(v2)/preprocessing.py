# src/preprocessing.py
def preprocess_dataset(dataset, tokenizer, max_length=512):
    """データのトークナイズと形式整形"""
    def preprocess_function(examples):
        inputs = examples["input"]
        outputs = examples["output"]
        text_pairs = [f"User: {i}\nBot: {o}" for i, o in zip(inputs, outputs)]
        model_inputs = tokenizer(
            text_pairs,
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        return model_inputs

    processed_dataset = dataset["train"].map(preprocess_function, batched=True)
    print("✅ Tokenization complete!")
    return processed_dataset
