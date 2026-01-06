import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

print("ライブラリをインストールしています...")

packages = [
    "torch",
    "transformers==4.37.2", 
    "bitsandbytes==0.41.3",  # 安定版に変更
    "peft==0.7.1",           # 安定版に変更
    "accelerate==0.27.2",    # 安定版に変更
    "datasets",
    "trl==0.7.10",           # 安定版
    "scipy"
]

for p in packages:
    try:
        print(f"Installing {p}...")
        install(p)
    except Exception as e:
        print(f"✗ Error installing {p}: {e}")

print("インストール完了。必ずランタイムを再起動してください。")