from datasets import load_dataset
import config
from utils import generate_prompt_for_training

print("データセットを確認中...")
try:
    dataset = load_dataset("json", data_files=config.DATA_PATH, split="train")
    print(f"データ件数: {len(dataset)}")
    print("--- データの先頭サンプル ---")
    print(generate_prompt_for_training(dataset[0]))
    print("----------------------------")
    print("データ確認OK")
except Exception as e:
    print(f"データ読み込みエラー: {e}")