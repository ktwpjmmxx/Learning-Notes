"""
環境のセットアップテスト
学習を始める前に実行して、全ての依存関係が正しく動作するか確認
"""
import sys

print("=" * 60)
print("環境テストを開始します")
print("=" * 60)

# ==========================================
# 1. バージョン確認
# ==========================================
print("\n[1] パッケージバージョン確認")
print("-" * 60)

packages_to_check = [
    "torch",
    "transformers",
    "bitsandbytes",
    "peft",
    "accelerate",
    "trl",
    "datasets"
]

for pkg_name in packages_to_check:
    try:
        pkg = __import__(pkg_name)
        version = getattr(pkg, "__version__", "不明")
        print(f"✓ {pkg_name:20s}: {version}")
    except ImportError as e:
        print(f"✗ {pkg_name:20s}: インポートエラー - {e}")
        sys.exit(1)

# ==========================================
# 2. CUDA確認
# ==========================================
print("\n[2] CUDA環境確認")
print("-" * 60)

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("⚠️ CUDAが利用できません。GPU利用を確認してください。")
    sys.exit(1)

# ==========================================
# 3. bitsandbytes動作確認
# ==========================================
print("\n[3] bitsandbytes動作確認")
print("-" * 60)

try:
    import bitsandbytes as bnb
    print(f"✓ bitsandbytes version: {bnb.__version__}")
    
    # 簡単な動作テスト
    test_tensor = torch.randn(10, 10).cuda()
    print("✓ CUDA tensor作成成功")
    
except Exception as e:
    print(f"✗ bitsandbytesエラー: {e}")
    sys.exit(1)

# ==========================================
# 4. データファイル確認
# ==========================================
print("\n[4] データファイル確認")
print("-" * 60)

import os
import config

if os.path.exists(config.DATA_PATH):
    import json
    with open(config.DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✓ データファイル存在: {config.DATA_PATH}")
    print(f"  データ件数: {len(data)}")
    if len(data) > 0:
        print(f"  サンプル: {list(data[0].keys())}")
else:
    print(f"✗ データファイルが見つかりません: {config.DATA_PATH}")
    print("  → 00_generate_data.py を実行してください")

# ==========================================
# 5. モデルロードテスト（軽量）
# ==========================================
print("\n[5] モデルアクセステスト")
print("-" * 60)

try:
    from transformers import AutoTokenizer
    print(f"トークナイザーをロード中: {config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, trust_remote_code=True)
    print(f"✓ トークナイザーロード成功")
    print(f"  語彙サイズ: {len(tokenizer)}")
    
    # テストトークン化
    test_text = "こんにちは"
    tokens = tokenizer(test_text)
    print(f"✓ トークン化テスト成功: '{test_text}' → {len(tokens['input_ids'])} tokens")
    
except Exception as e:
    print(f"✗ モデルアクセスエラー: {e}")

# ==========================================
# 完了
# ==========================================
print("\n" + "=" * 60)
print("✓ 環境テスト完了")
print("=" * 60)
print("\n次のステップ:")
print("1. データ生成:    python 00_generate_data.py")
print("2. データ確認:    python 02_process_data.py")
print("3. 学習実行:      python 04_train.py")
print("4. 推論テスト:    python 05_inference.py")
print("=" * 60)