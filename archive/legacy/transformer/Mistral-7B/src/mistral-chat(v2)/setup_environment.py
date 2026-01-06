# src/setup_environment.py
import os

def prepare_directories(dirs=None):
    """必要なディレクトリを作成"""
    if dirs is None:
        dirs = ["logs", "outputs", "results", "data"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("✅ Directories ready")
