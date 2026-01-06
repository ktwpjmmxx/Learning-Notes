import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

print("ライブラリをインストールしています...")
packages = ["torch", "transformers", "bitsandbytes", "peft", "accelerate", "datasets", "trl"]
for p in packages:
    try:
        install(p)
    except Exception as e:
        print(f"Error installing {p}: {e}")
print("インストール完了。必要に応じてランタイムを再起動してください。")