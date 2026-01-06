# src/data_loader.py
from datasets import load_dataset

def load_json_dataset(file_path="data/training_data.json"):
    dataset = load_dataset("json", data_files=file_path)
    print("✅ Dataset loaded:", dataset)
    return dataset
