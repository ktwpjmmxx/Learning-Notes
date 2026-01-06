"""
共通ユーティリティ関数
各スクリプトで使用する汎用的な関数を集約
"""

import os
import json
import torch
import gc
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging


# ============================================================
# ロギング設定
# ============================================================

def setup_logger(name: str = "mistral_training", log_file: Optional[str] = None) -> logging.Logger:
    """
    ロガーのセットアップ
    
    Args:
        name: ロガー名
        log_file: ログファイルのパス（Noneの場合はコンソールのみ）
    
    Returns:
        設定済みのロガー
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # フォーマット設定
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # コンソールハンドラ
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # ファイルハンドラ（オプション）
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# ============================================================
# ディレクトリ管理
# ============================================================

def create_directories(dirs: List[str]) -> None:
    """
    複数のディレクトリを一括作成
    
    Args:
        dirs: 作成するディレクトリのリスト
    """
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    print(f"✅ Created {len(dirs)} directories")


def get_project_structure() -> Dict[str, str]:
    """
    プロジェクトのディレクトリ構造を返す
    
    Returns:
        ディレクトリパスの辞書
    """
    return {
        "logs": "logs",
        "outputs": "outputs",
        "results": "results",
        "results_training": "results/training",
        "results_evaluation": "results/evaluation",
        "results_chat_logs": "results/chat_logs",
        "data": "data",
        "models": "models",
    }


# ============================================================
# JSON操作
# ============================================================

def save_json(data: Any, file_path: str, indent: int = 2) -> None:
    """
    データをJSONファイルに保存
    
    Args:
        data: 保存するデータ
        file_path: 保存先のパス
        indent: インデント幅
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    
    print(f"✅ Saved JSON to: {file_path}")


def load_json(file_path: str) -> Any:
    """
    JSONファイルを読み込み
    
    Args:
        file_path: 読み込むファイルのパス
    
    Returns:
        読み込んだデータ
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✅ Loaded JSON from: {file_path}")
    return data


# ============================================================
# メモリ管理
# ============================================================

def cleanup_memory() -> None:
    """
    GPUメモリとRAMをクリーンアップ
    """
    print("🧹 Cleaning up memory...")
    
    # PyTorchのキャッシュをクリア
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Pythonのガベージコレクション
    gc.collect()
    
    print("✅ Memory cleaned!")


def get_gpu_memory_info() -> Dict[str, float]:
    """
    GPU メモリ使用状況を取得
    
    Returns:
        メモリ情報の辞書
    """
    if not torch.cuda.is_available():
        return {"gpu_available": False}
    
    allocated = torch.cuda.memory_allocated() / 1024**3  # GB
    reserved = torch.cuda.memory_reserved() / 1024**3  # GB
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
    
    return {
        "gpu_available": True,
        "allocated_gb": round(allocated, 2),
        "reserved_gb": round(reserved, 2),
        "total_gb": round(total, 2),
        "free_gb": round(total - allocated, 2)
    }


def print_gpu_memory() -> None:
    """GPU メモリ使用状況を表示"""
    info = get_gpu_memory_info()
    
    if not info["gpu_available"]:
        print("⚠️ GPU not available")
        return
    
    print("📊 GPU Memory Status:")
    print(f"   Allocated: {info['allocated_gb']} GB")
    print(f"   Reserved: {info['reserved_gb']} GB")
    print(f"   Free: {info['free_gb']} GB")
    print(f"   Total: {info['total_gb']} GB")


# ============================================================
# モデル情報
# ============================================================

def count_parameters(model) -> Dict[str, Any]:
    """
    モデルのパラメータ数を計算
    
    Args:
        model: PyTorchモデル
    
    Returns:
        パラメータ情報の辞書
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_params": total_params,
        "total_params_millions": round(total_params / 1e6, 2),
        "total_params_billions": round(total_params / 1e9, 2),
        "trainable_params": trainable_params,
        "trainable_params_millions": round(trainable_params / 1e6, 2),
        "trainable_percentage": round(100 * trainable_params / total_params, 4)
    }


def print_model_info(model) -> None:
    """モデル情報を表示"""
    info = count_parameters(model)
    
    print("📊 Model Information:")
    print(f"   Total parameters: {info['total_params_billions']:.2f}B ({info['total_params']:,})")
    print(f"   Trainable parameters: {info['trainable_params_millions']:.2f}M ({info['trainable_params']:,})")
    print(f"   Trainable percentage: {info['trainable_percentage']:.4f}%")


# ============================================================
# 実験管理
# ============================================================

def create_experiment_id() -> str:
    """
    実験IDを生成（タイムスタンプベース）
    
    Returns:
        実験ID（例: "20250112_143025"）
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_experiment_config(config: Dict[str, Any], experiment_id: str = None) -> str:
    """
    実験設定を保存
    
    Args:
        config: 実験設定の辞書
        experiment_id: 実験ID（Noneの場合は自動生成）
    
    Returns:
        実験ID
    """
    if experiment_id is None:
        experiment_id = create_experiment_id()
    
    config["experiment_id"] = experiment_id
    config["timestamp"] = datetime.now().isoformat()
    
    save_json(config, f"results/training/experiment_{experiment_id}.json")
    
    return experiment_id


# ============================================================
# データセット統計
# ============================================================

def calculate_dataset_stats(dataset) -> Dict[str, Any]:
    """
    データセットの統計情報を計算
    
    Args:
        dataset: HuggingFace Dataset
    
    Returns:
        統計情報の辞書
    """
    if len(dataset) == 0:
        return {"error": "Empty dataset"}
    
    # トークン長の統計（input_idsがある場合）
    if "input_ids" in dataset.column_names:
        lengths = [len(item) for item in dataset["input_ids"]]
        
        return {
            "num_samples": len(dataset),
            "avg_length": round(sum(lengths) / len(lengths), 2),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "total_tokens": sum(lengths)
        }
    
    return {
        "num_samples": len(dataset),
        "columns": dataset.column_names
    }


def print_dataset_stats(train_dataset, eval_dataset=None) -> None:
    """データセット統計を表示"""
    print("📊 Dataset Statistics:")
    
    train_stats = calculate_dataset_stats(train_dataset)
    print(f"\n   Training set:")
    for key, value in train_stats.items():
        print(f"      {key}: {value}")
    
    if eval_dataset:
        eval_stats = calculate_dataset_stats(eval_dataset)
        print(f"\n   Evaluation set:")
        for key, value in eval_stats.items():
            print(f"      {key}: {value}")


# ============================================================
# 学習履歴の分析
# ============================================================

def analyze_training_history(history: Dict[str, List]) -> Dict[str, Any]:
    """
    学習履歴を分析
    
    Args:
        history: 学習履歴の辞書
    
    Returns:
        分析結果の辞書
    """
    analysis = {}
    
    # 訓練損失の分析
    if "train_losses" in history and history["train_losses"]:
        train_losses = [item["loss"] for item in history["train_losses"]]
        analysis["train_loss"] = {
            "initial": train_losses[0],
            "final": train_losses[-1],
            "min": min(train_losses),
            "max": max(train_losses),
            "reduction": train_losses[0] - train_losses[-1],
            "reduction_percentage": round(100 * (train_losses[0] - train_losses[-1]) / train_losses[0], 2)
        }
    
    # 評価損失の分析
    if "eval_losses" in history and history["eval_losses"]:
        eval_losses = [item["eval_loss"] for item in history["eval_losses"]]
        analysis["eval_loss"] = {
            "initial": eval_losses[0],
            "final": eval_losses[-1],
            "min": min(eval_losses),
            "best_step": history["eval_losses"][eval_losses.index(min(eval_losses))]["step"]
        }
        
        # 過学習の検出
        if "train_loss" in analysis:
            gap = analysis["eval_loss"]["final"] - analysis["train_loss"]["final"]
            analysis["overfitting"] = {
                "gap": round(gap, 4),
                "status": "High risk" if gap > 0.5 else "Moderate" if gap > 0.2 else "Low risk"
            }
    
    return analysis


def print_training_analysis(history: Dict[str, List]) -> None:
    """学習分析結果を表示"""
    analysis = analyze_training_history(history)
    
    print("📊 Training Analysis:")
    
    if "train_loss" in analysis:
        print(f"\n   Training Loss:")
        print(f"      Initial: {analysis['train_loss']['initial']:.4f}")
        print(f"      Final: {analysis['train_loss']['final']:.4f}")
        print(f"      Reduction: {analysis['train_loss']['reduction']:.4f} ({analysis['train_loss']['reduction_percentage']:.2f}%)")
    
    if "eval_loss" in analysis:
        print(f"\n   Evaluation Loss:")
        print(f"      Initial: {analysis['eval_loss']['initial']:.4f}")
        print(f"      Final: {analysis['eval_loss']['final']:.4f}")
        print(f"      Best: {analysis['eval_loss']['min']:.4f} (Step {analysis['eval_loss']['best_step']})")
    
    if "overfitting" in analysis:
        print(f"\n   Overfitting Analysis:")
        print(f"      Train-Eval Gap: {analysis['overfitting']['gap']:.4f}")
        print(f"      Status: {analysis['overfitting']['status']}")


# ============================================================
# テストユーティリティ
# ============================================================

def run_test_prompts(model, tokenizer, test_prompts: List[str], max_new_tokens: int = 100) -> List[Dict[str, str]]:
    """
    複数のテストプロンプトを実行
    
    Args:
        model: モデル
        tokenizer: トークナイザー
        test_prompts: テストプロンプトのリスト
        max_new_tokens: 最大生成トークン数
    
    Returns:
        結果のリスト
    """
    results = []
    
    print(f"🧪 Running {len(test_prompts)} test prompts...\n")
    
    for i, prompt in enumerate(test_prompts, 1):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id
            )
        
        result_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        bot_response = result_text.split("Bot:")[-1].strip().split("\n")[0]
        
        results.append({
            "prompt": prompt,
            "response": bot_response
        })
        
        print(f"Test {i}/{len(test_prompts)}:")
        print(f"  Prompt: {prompt.split('Bot:')[0].strip()}")
        print(f"  Response: {bot_response}\n")
    
    return results


# ============================================================
# 設定ファイル管理
# ============================================================

def load_config(config_path: str = "config/training_config.yaml") -> Dict[str, Any]:
    """
    設定ファイルを読み込み（YAML or JSON）
    
    Args:
        config_path: 設定ファイルのパス
    
    Returns:
        設定の辞書
    """
    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
        try:
            import yaml
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except ImportError:
            print("⚠️ PyYAML not installed. Install with: pip install pyyaml")
            return {}
    
    elif config_path.endswith('.json'):
        return load_json(config_path)
    
    else:
        raise ValueError(f"Unsupported config format: {config_path}")


# ============================================================
# バージョン情報
# ============================================================

def print_environment_info() -> Dict[str, str]:
    """
    環境情報を表示・返却
    
    Returns:
        環境情報の辞書
    """
    import sys
    import transformers
    import peft
    
    info = {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "peft_version": peft.__version__,
        "cuda_available": str(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    }
    
    print("🔧 Environment Information:")
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    return info


# ============================================================
# 使用例（テスト用）
# ============================================================

if __name__ == "__main__":
    # ロガーのテスト
    logger = setup_logger(log_file="logs/test.log")
    logger.info("Testing logger...")
    
    # ディレクトリ作成のテスト
    dirs = get_project_structure()
    create_directories(list(dirs.values()))
    
    # GPU情報のテスト
    print_gpu_memory()
    
    # 環境情報のテスト
    env_info = print_environment_info()
    
    # JSON保存のテスト
    test_data = {"test": "data", "number": 123}
    save_json(test_data, "logs/test.json")
    
    print("\n✅ All utils tests passed!")