"""
Google Colab用 完全インストールスクリプト
動作確認済みバージョンセット

実行手順:
1. このスクリプトを実行
2. ランタイムを出荷時の設定にリセット
3. もう一度実行
4. ランタイムを再起動
5. 学習開始
"""
import subprocess
import sys

def run(cmd):
    print(f"\n実行中: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"警告: {result.stderr}")
    return result.returncode == 0

print("=" * 70)
print("Google Colab Mistral-7B 学習環境セットアップ")
print("=" * 70)

print("\n【重要】このスクリプト実行前に:")
print("メニュー: ランタイム > ランタイムを出荷時の設定にリセット")
print("を実行することを推奨します。")
print("=" * 70)
print("\nインストールを開始します...")

# ==========================================
# Step 1: 古いパッケージをアンインストール
# ==========================================
print("\n[Step 1/3] 古いパッケージをクリーンアップ中...")
cleanup = [
    "bitsandbytes",
    "transformers", 
    "tokenizers",
    "peft",
    "accelerate",
    "trl",
    "datasets"
]

for pkg in cleanup:
    run(f"pip uninstall -y {pkg}")

# ==========================================
# Step 2: 動作確認済みバージョンをインストール
# ==========================================
print("\n[Step 2/4] パッケージをインストール中...")
print("※ 3-5分かかります")

# PyTorch は既存のものを使用 (2.9 + CUDA 12.6)
packages = [
    "transformers==4.36.2",
    "tokenizers==0.15.0", 
    "peft==0.7.1",
    "accelerate==0.25.0",
    "trl==0.7.10",
    "datasets==2.16.1",
    "scipy"
]

for pkg in packages:
    success = run(f"pip install -q {pkg}")
    if success:
        print(f"✓ {pkg} インストール完了")
    else:
        print(f"✗ {pkg} インストール失敗")

# ==========================================
# Step 3: bitsandbytes を GitHubからインストール
# ==========================================
print("\n[Step 3/4] bitsandbytes (最新版) をインストール中...")
print("※ CUDA 12.6対応版を GitHubから取得します (3-5分)")
success = run("pip install -q --no-cache-dir git+https://github.com/TimDettmers/bitsandbytes.git")
if success:
    print("✓ bitsandbytes インストール完了")
else:
    print("✗ bitsandbytes インストール失敗")

for pkg in packages:
    success = run(f"pip install -q {pkg}")
    if success:
        print(f"✓ {pkg} インストール完了")
    else:
        print(f"✗ {pkg} インストール失敗")

# ==========================================
# Step 4: インストール確認
# ==========================================
print("\n[Step 4/4] インストール確認中...")

try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    
    import transformers
    print(f"✓ transformers: {transformers.__version__}")
    
    import bitsandbytes
    print(f"✓ bitsandbytes: インポート成功")
    
    import peft
    print(f"✓ peft: {peft.__version__}")
    
    print("\n" + "=" * 70)
    print("✓ インストール完了")
    print("=" * 70)
    print("\n次のステップ:")
    print("1. ランタイムを再起動 (メニュー: ランタイム > ランタイムを再起動)")
    print("2. python 00_generate_data.py を実行")
    print("3. python 04_train.py を実行")
    print("=" * 70)
    
except Exception as e:
    print(f"\n✗ エラー: {e}")
    print("\nランタイムを出荷時の設定にリセットしてから再実行してください")