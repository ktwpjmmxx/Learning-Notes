"""
01_setup_environment.py (CUDA 12.6対応版)
"""

!nvidia-smi

print("="*60)
print("🔧 環境セットアップ開始")
print("="*60)

# Step 1: 問題のあるパッケージを完全にクリーンアップ
print("\n📦 Step 1: クリーンアップ...")
!pip uninstall -y bitsandbytes sentence-transformers transformers accelerate -q
!pip cache purge

# Step 2: CUDA 12.xに対応したbitsandbytesをインストール
print("📦 Step 2: CUDA 12.x対応バージョンをインストール中...")

# まず、PyTorchとCUDAのバージョンを確認
import torch
print(f"   PyTorch: {torch.__version__}")
print(f"   CUDA: {torch.version.cuda}")

# bitsandbytes（CUDA 12.x対応の最新版）
print("\n   bitsandbytesをインストール中...")
!pip install -q bitsandbytes>=0.43.1

# その他のライブラリ
print("   その他のライブラリをインストール中...")
!pip install -q transformers>=4.36.0
!pip install -q datasets>=2.18.0
!pip install -q accelerate>=0.25.0
!pip install -q peft>=0.7.0
!pip install -q trl>=0.7.11
!pip install -q gradio

print("\n✅ パッケージインストール完了！")

# Step 3: bitsandbytesの動作確認（詳細）
print("\n🧪 Step 3: bitsandbytes動作確認...")
try:
    import bitsandbytes as bnb
    print(f"   ✅ bitsandbytes {bnb.__version__} インポート成功")
    
    # CUDA設定の確認
    print("\n   CUDA設定の確認:")
    print(f"      CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"      GPU: {torch.cuda.get_device_name(0)}")
        print(f"      CUDA version: {torch.version.cuda}")
        print(f"      GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # 4bit設定のテスト
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    print("   ✅ 4bit量子化設定が正常に作成されました")
    
except Exception as e:
    print(f"\n   ❌ エラー発生: {e}")
    print("\n   代替案: 量子化なしでモデルを読み込みます")
    print("   （メモリ使用量は増えますが、動作します）")

# Step 4: インストール確認
print("\n📋 Step 4: インストール確認...")
import importlib

packages = [
    'torch',
    'transformers',
    'datasets',
    'accelerate',
    'peft',
    'bitsandbytes',
    'trl',
    'gradio',
]

print("\n📊 インストール済みパッケージ:")
for pkg in packages:
    try:
        module = importlib.import_module(pkg)
        version = getattr(module, '__version__', 'N/A')
        print(f"   ✅ {pkg}: {version}")
    except ImportError as e:
        print(f"   ❌ {pkg}: インポート失敗")

# Step 5: ディレクトリ作成
print("\n📁 Step 5: ディレクトリ作成...")
import os

directories = [
    "logs",
    "outputs",
    "results",
    "results/training",
    "results/evaluation",
    "results/chat_logs"
]

for directory in directories:
    os.makedirs(directory, exist_ok=True)
    print(f"   ✅ {directory}/")

# Step 6: 実験設定の保存
print("\n💾 Step 6: 実験設定の保存...")
import json
from datetime import datetime
import sys

experiment_config = {
    "experiment_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "experiment_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
    "model_name": "mistralai/Mistral-7B-v0.1",
    "training_data_size": 100,
    "framework": "PEFT + LoRA",
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    "environment": "Google Colab",
    "python_version": sys.version.split()[0],
    "torch_version": torch.__version__,
    "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
}

# バージョン情報を安全に取得
version_packages = ['transformers', 'peft', 'bitsandbytes', 'accelerate', 'datasets']
for pkg in version_packages:
    try:
        module = importlib.import_module(pkg)
        experiment_config[f"{pkg}_version"] = getattr(module, '__version__', 'N/A')
    except Exception:
        experiment_config[f"{pkg}_version"] = "N/A"

with open("results/training/experiment_config.json", "w") as f:
    json.dump(experiment_config, f, indent=2)

print("   ✅ 実験設定を保存しました")

# 完了メッセージ
print("\n" + "="*60)
print("✅ セットアップ完了！")
print("="*60)
print(f"\n📊 実験ID: {experiment_config['experiment_id']}")
print(f"🖥️  GPU: {experiment_config['gpu']}")
print(f"🔥 CUDA: {experiment_config['cuda_version']}")
print(f"🐍 Python: {experiment_config['python_version']}")
print(f"🔥 PyTorch: {experiment_config['torch_version']}")

# 次のステップ表示
print("\n🚀 次のステップ:")
print("   02_load_model.py を実行してください")

print("\n💡 ヒント:")
print("   - 学習には約7-10分かかります")
print("   - GPU使用率は nvidia-smi で確認できます")