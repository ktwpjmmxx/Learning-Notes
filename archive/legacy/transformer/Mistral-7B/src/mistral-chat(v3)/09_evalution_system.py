"""
09_evaluation_system.py
モデルの性能を自動評価するシステム
定量的・定性的評価を実施し、レポートを自動生成
"""

import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datetime import datetime
from typing import List, Dict, Any
import json
import os

# utils読み込み
from utils import cleanup_memory, save_json, print_gpu_memory
from utils_v3 import (
    calculate_keyword_coverage,
    check_response_quality,
    generate_evaluation_report
)

# ==========================================
# 設定
# ==========================================

LOCAL_ADAPTER_DIR = "mistral7b_finetuned"
BASE_MODEL_ID = "mistralai/Mistral-7B-v0.1"
RESULTS_DIR = "results/evaluation"

# ディレクトリ作成
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==========================================
# 1. モデルのロード
# ==========================================

print("="*60)
print("🔍 Model Evaluation System")
print("="*60)

# メモリクリーンアップ
cleanup_memory()

# 4bit量子化設定
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print("\n📥 Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

print("📥 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

print("📥 Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, LOCAL_ADAPTER_DIR)

print("✅ Model loaded successfully!\n")
print_gpu_memory()

# ==========================================
# 2. 評価用テストケース
# ==========================================

TEST_CASES = [
    {
        "category": "Technical Explanation",
        "input": "What is machine learning?",
        "expected_keywords": ["data", "algorithm", "learn", "pattern", "predict", "model"]
    },
    {
        "category": "Technical Explanation",
        "input": "Explain the Transformer architecture",
        "expected_keywords": ["attention", "encoder", "decoder", "layer", "self-attention"]
    },
    {
        "category": "Coding Question",
        "input": "How do I read a file in Python?",
        "expected_keywords": ["open", "read", "file", "with", "close"]
    },
    {
        "category": "Coding Question",
        "input": "What are Python decorators?",
        "expected_keywords": ["function", "decorator", "modify", "@", "wrapper"]
    },
    {
        "category": "Problem Solving",
        "input": "How to fix overfitting in neural networks?",
        "expected_keywords": ["dropout", "regularization", "data", "validation", "early"]
    },
    {
        "category": "Problem Solving",
        "input": "How to improve model training speed?",
        "expected_keywords": ["batch", "gpu", "precision", "optimization", "parallel"]
    },
    {
        "category": "Comparison",
        "input": "What's the difference between AI and ML?",
        "expected_keywords": ["AI", "machine learning", "subset", "broader", "intelligence"]
    },
    {
        "category": "Comparison",
        "input": "PyTorch vs TensorFlow comparison",
        "expected_keywords": ["pytorch", "tensorflow", "dynamic", "static", "graph"]
    },
    {
        "category": "Conversational",
        "input": "Hello! How are you today?",
        "expected_keywords": ["hello", "good", "fine", "help", "today"]
    },
    {
        "category": "Conversational",
        "input": "How can I get started with AI?",
        "expected_keywords": ["learn", "python", "course", "practice", "project"]
    },
]

print(f"📋 Prepared {len(TEST_CASES)} test cases\n")

# ==========================================
# 3. 評価関数
# ==========================================

def generate_response(prompt: str, max_new_tokens: int = 200) -> str:
    """
    プロンプトから応答を生成
    
    Args:
        prompt: 入力プロンプト
        max_new_tokens: 最大生成トークン数
    
    Returns:
        生成された応答
    """
    full_prompt = f"User: {prompt}\nBot:"
    inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
    
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
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = result.split("Bot:")[-1].strip()
    
    # クリーンアップ
    if "User:" in response:
        response = response.split("User:")[0].strip()
    if "\n" in response:
        response = response.split("\n")[0].strip()
    
    return response


def evaluate_single_case(test_case: Dict[str, Any]) -> Dict[str, Any]:
    """
    単一のテストケースを評価
    
    Args:
        test_case: テストケース
    
    Returns:
        評価結果
    """
    # 応答生成
    response = generate_response(test_case["input"])
    
    # キーワードカバレッジ
    keyword_score = calculate_keyword_coverage(
        response,
        test_case["expected_keywords"]
    )
    
    # 品質チェック
    quality = check_response_quality(response)
    
    # 結果
    result = {
        "category": test_case["category"],
        "input": test_case["input"],
        "response": response,
        "keyword_coverage": keyword_score,
        "quality_score": quality["score"],
        "quality_checks": quality["checks"],
        "response_length": quality["word_count"],
        "matched_keywords": [
            kw for kw in test_case["expected_keywords"]
            if kw.lower() in response.lower()
        ]
    }
    
    return result


# ==========================================
# 4. 評価実行
# ==========================================

def run_evaluation() -> Dict[str, Any]:
    """
    全テストケースで評価を実行
    
    Returns:
        評価結果の辞書
    """
    print("="*60)
    print("🚀 Starting Evaluation")
    print("="*60)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "model_path": LOCAL_ADAPTER_DIR,
        "base_model": BASE_MODEL_ID,
        "test_cases": [],
        "summary": {}
    }
    
    # メトリクスの集計用
    all_keyword_scores = []
    all_quality_scores = []
    all_response_lengths = []
    category_scores = {}
    
    # 各テストケースを評価
    for i, test_case in enumerate(TEST_CASES, 1):
        print(f"\n[{i}/{len(TEST_CASES)}] {test_case['category']}")
        print(f"  Input: {test_case['input']}")
        
        # 評価実行
        result = evaluate_single_case(test_case)
        
        # 結果表示
        print(f"  Response: {result['response'][:80]}...")
        print(f"  ✓ Keyword: {result['keyword_coverage']:.1%} | "
              f"Quality: {result['quality_score']:.1%} | "
              f"Length: {result['response_length']} words")
        
        # 結果保存
        results["test_cases"].append(result)
        
        # メトリクス集計
        all_keyword_scores.append(result["keyword_coverage"])
        all_quality_scores.append(result["quality_score"])
        all_response_lengths.append(result["response_length"])
        
        # カテゴリ別集計
        category = result["category"]
        if category not in category_scores:
            category_scores[category] = []
        category_scores[category].append({
            "keyword": result["keyword_coverage"],
            "quality": result["quality_score"]
        })
    
    # サマリー計算
    import numpy as np
    
    results["summary"] = {
        "total_tests": len(TEST_CASES),
        "avg_keyword_coverage": float(np.mean(all_keyword_scores)),
        "avg_quality_score": float(np.mean(all_quality_scores)),
        "avg_response_length": float(np.mean(all_response_lengths)),
        "overall_score": float(
            np.mean(all_keyword_scores) * 0.4 +
            np.mean(all_quality_scores) * 0.6
        )
    }
    
    # カテゴリ別サマリー
    category_summary = {}
    for category, scores in category_scores.items():
        category_summary[category] = {
            "avg_keyword": float(np.mean([s["keyword"] for s in scores])),
            "avg_quality": float(np.mean([s["quality"] for s in scores])),
            "count": len(scores)
        }
    
    results["category_summary"] = category_summary
    
    return results


# ==========================================
# 5. 結果保存と表示
# ==========================================

def save_and_display_results(results: Dict[str, Any]) -> None:
    """
    結果を保存して表示
    
    Args:
        results: 評価結果
    """
    # タイムスタンプ
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSON保存
    json_path = f"{RESULTS_DIR}/evaluation_{timestamp}.json"
    save_json(results, json_path)
    
    # Markdownレポート生成
    md_path = f"{RESULTS_DIR}/evaluation_{timestamp}.md"
    generate_evaluation_report(results, md_path)
    
    # サマリー表示
    print("\n" + "="*60)
    print("📊 Evaluation Summary")
    print("="*60)
    
    summary = results["summary"]
    print(f"\n🎯 Overall Performance:")
    print(f"   Overall Score:       {summary['overall_score']:.2%}")
    print(f"   Keyword Coverage:    {summary['avg_keyword_coverage']:.2%}")
    print(f"   Quality Score:       {summary['avg_quality_score']:.2%}")
    print(f"   Avg Response Length: {summary['avg_response_length']:.1f} words")
    
    # カテゴリ別サマリー
    print(f"\n📋 Performance by Category:")
    for category, scores in results["category_summary"].items():
        print(f"\n   {category}:")
        print(f"      Keyword: {scores['avg_keyword']:.1%} | "
              f"Quality: {scores['avg_quality']:.1%} | "
              f"Tests: {scores['count']}")
    
    print("\n" + "="*60)
    print("✅ Evaluation Complete!")
    print("="*60)
    print(f"\n📁 Results saved:")
    print(f"   JSON:     {json_path}")
    print(f"   Markdown: {md_path}")
    
    # 改善提案
    print(f"\n💡 Recommendations:")
    if summary['avg_keyword_coverage'] < 0.6:
        print("   ⚠️ Keyword coverage is low. Consider:")
        print("      - Expanding training data with domain-specific examples")
        print("      - Adjusting LoRA target modules")
    
    if summary['avg_quality_score'] < 0.7:
        print("   ⚠️ Response quality needs improvement. Consider:")
        print("      - Increasing training steps")
        print("      - Adjusting repetition penalty")
    
    if summary['avg_response_length'] < 30:
        print("   ℹ️ Responses are relatively short. This may be intentional.")
    
    if summary['overall_score'] >= 0.75:
        print("   ✨ Excellent performance! Model is production-ready.")


# ==========================================
# 6. メイン実行
# ==========================================

if __name__ == "__main__":
    try:
        # 評価実行
        results = run_evaluation()
        
        # 結果保存・表示
        save_and_display_results(results)
        
        # メモリクリーンアップ
        print("\n🧹 Cleaning up...")
        cleanup_memory()
        
        print("\n✅ All done!")
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()