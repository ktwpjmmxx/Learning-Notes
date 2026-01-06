"""
学習パイプライン全体を実行
python run_training.py --config config/training_config.yaml
"""

import argparse
from scripts import (
    setup_environment,
    load_model,
    preprocess_data,
    configure_lora,
    train,
    save_model,
    test_inference
)

def main():
    # 各ステップを順番に実行
    setup_environment.run()
    model, tokenizer, dataset = load_model.run()
    train_data, eval_data = preprocess_data.run(dataset, tokenizer)
    model = configure_lora.run(model)
    train.run(model, tokenizer, train_data, eval_data)
    save_model.run(model, tokenizer)
    test_inference.run()

if __name__ == "__main__":
    main()