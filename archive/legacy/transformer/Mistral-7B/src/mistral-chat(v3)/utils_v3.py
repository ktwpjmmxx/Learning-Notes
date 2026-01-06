import json
import numpy as np
from typing import List, Dict, Any
from datetime import datetime
import re


# ============================================================
# データ拡張
# ============================================================

def generate_enhanced_dataset() -> List[Dict[str, str]]:
    """
    高品質な拡張データセットを生成
    カテゴリ別に整理された学習データ
    
    Returns:
        拡張データセットのリスト
    """
    
    dataset = []
    
    # カテゴリ1: 技術説明
    technical_data = [
        {
            "input": "Explain the Transformer architecture",
            "output": "The Transformer is a neural network architecture that uses self-attention mechanisms to process sequential data. It consists of an encoder and decoder, both using multi-head attention layers. The key innovation is the ability to capture long-range dependencies without recurrent connections, making it highly parallelizable and efficient for training."
        },
        {
            "input": "What is LoRA fine-tuning?",
            "output": "LoRA (Low-Rank Adaptation) is an efficient fine-tuning technique that adds small trainable rank decomposition matrices to frozen pre-trained model weights. This dramatically reduces the number of trainable parameters (often by 10,000x) while maintaining performance, making it ideal for adapting large language models with limited compute resources."
        },
        {
            "input": "How does batch normalization work?",
            "output": "Batch normalization normalizes layer inputs across a mini-batch by subtracting the batch mean and dividing by batch standard deviation. This stabilizes training, allows higher learning rates, and acts as a regularizer. It includes learnable scale and shift parameters to preserve the network's representational power."
        },
        {
            "input": "What is the attention mechanism?",
            "output": "Attention mechanisms allow models to focus on relevant parts of the input when producing outputs. They compute weighted sums of input features, where weights are determined by learned compatibility scores between queries and keys. This enables models to capture long-range dependencies and contextual relationships effectively."
        },
        {
            "input": "Explain gradient descent optimization",
            "output": "Gradient descent is an iterative optimization algorithm that minimizes a loss function by moving in the direction of steepest descent (negative gradient). It updates parameters by subtracting the gradient multiplied by a learning rate. Variants like SGD, Adam, and RMSprop improve convergence speed and stability."
        }
    ]
    
    # カテゴリ2: プログラミング
    coding_data = [
        {
            "input": "How do I use list comprehension in Python?",
            "output": "List comprehension provides a concise way to create lists. Syntax: [expression for item in iterable if condition]. Example: squares = [x**2 for x in range(10) if x % 2 == 0] creates a list of squares of even numbers. It's faster and more Pythonic than traditional loops."
        },
        {
            "input": "What are Python decorators?",
            "output": "Decorators are functions that modify the behavior of other functions or classes. They use the @decorator syntax and are commonly used for logging, timing, authentication, and caching. Example: @staticmethod or @property. Decorators take a function as input and return a modified version."
        },
        {
            "input": "Explain async/await in Python",
            "output": "async/await enables asynchronous programming in Python. Functions defined with 'async def' return coroutines that can be paused and resumed. 'await' suspends execution until an awaitable completes. This is ideal for I/O-bound operations, allowing concurrent execution without threads."
        },
        {
            "input": "How to handle exceptions in Python?",
            "output": "Use try-except blocks to handle exceptions. The try block contains code that might raise exceptions, except blocks catch specific exception types, else runs if no exception occurs, and finally always executes. Best practice: catch specific exceptions, avoid bare except clauses, and log errors appropriately."
        },
        {
            "input": "What is the difference between list and tuple?",
            "output": "Lists are mutable (can be modified after creation) and use square brackets []. Tuples are immutable (cannot be changed) and use parentheses (). Tuples are faster and use less memory, making them ideal for fixed data. Lists are better when you need to modify elements."
        }
    ]
    
    # カテゴリ3: 問題解決
    problem_solving = [
        {
            "input": "How to prevent overfitting in neural networks?",
            "output": "Key strategies to prevent overfitting: 1) Increase training data or use data augmentation, 2) Add dropout layers, 3) Apply L1/L2 regularization, 4) Use early stopping by monitoring validation loss, 5) Reduce model complexity, 6) Use batch normalization, 7) Apply cross-validation. Monitor the gap between training and validation metrics."
        },
        {
            "input": "How to fix memory leaks in Python?",
            "output": "To prevent memory leaks: 1) Use context managers (with statements) for resources, 2) Avoid circular references or use weakref, 3) Delete large objects explicitly with del, 4) Use generators instead of lists for large datasets, 5) Profile memory with tracemalloc or memory_profiler, 6) In PyTorch, call torch.cuda.empty_cache() to free GPU memory."
        },
        {
            "input": "How to improve model training speed?",
            "output": "Speed up training by: 1) Using mixed precision training (fp16), 2) Increasing batch size if memory allows, 3) Enabling gradient accumulation, 4) Using efficient data loaders with multiple workers, 5) Applying gradient checkpointing for large models, 6) Using faster optimizers like AdamW, 7) Leveraging distributed training across multiple GPUs."
        },
        {
            "input": "How to handle imbalanced datasets?",
            "output": "Address class imbalance through: 1) Oversampling minority class (SMOTE), 2) Undersampling majority class, 3) Using class weights in loss function, 4) Applying focal loss, 5) Using ensemble methods, 6) Choosing appropriate metrics (F1-score, AUC instead of accuracy), 7) Generating synthetic samples."
        },
        {
            "input": "How to debug deep learning models?",
            "output": "Debugging strategies: 1) Start with a small dataset and overfit, 2) Check data preprocessing and labels, 3) Verify loss function and metrics, 4) Monitor gradients for vanishing/exploding, 5) Use gradient clipping, 6) Visualize activations and weights, 7) Compare with baseline models, 8) Log intermediate outputs, 9) Use tensorboard for visualization."
        }
    ]
    
    # カテゴリ4: 比較・対照
    comparison_data = [
        {
            "input": "What's the difference between AI and ML?",
            "output": "AI (Artificial Intelligence) is the broader concept of machines performing tasks that typically require human intelligence. ML (Machine Learning) is a subset of AI that learns from data without explicit programming. So: ML ⊂ AI. Deep Learning is a further subset: Deep Learning ⊂ ML ⊂ AI."
        },
        {
            "input": "PyTorch vs TensorFlow comparison",
            "output": "PyTorch uses dynamic computation graphs (define-by-run), making debugging easier and preferred in research. TensorFlow historically used static graphs but TensorFlow 2.x defaults to eager execution. TensorFlow has better production deployment tools (TF Lite, TF Serving), while PyTorch is more intuitive and Python-like."
        },
        {
            "input": "Difference between CNN and RNN?",
            "output": "CNNs (Convolutional Neural Networks) excel at spatial pattern recognition in grid-like data (images). They use local connectivity and weight sharing. RNNs (Recurrent Neural Networks) process sequential data with temporal dependencies. They maintain hidden states to remember past information. Use CNNs for images, RNNs for sequences."
        },
        {
            "input": "Adam vs SGD optimizer comparison",
            "output": "SGD (Stochastic Gradient Descent) is simple, requires careful learning rate tuning, but can achieve better generalization. Adam (Adaptive Moment Estimation) adapts learning rates per parameter, works well with default settings, and converges faster. Adam is generally preferred for quick experimentation, SGD for final performance tuning."
        },
        {
            "input": "BERT vs GPT architecture differences",
            "output": "BERT uses bidirectional Transformers (encoder-only), trained with masked language modeling to understand context from both directions. Excellent for classification and question-answering. GPT uses unidirectional Transformers (decoder-only), trained autoregressively left-to-right. Superior for text generation. BERT understands, GPT generates."
        }
    ]
    
    # カテゴリ5: 対話的質問
    conversational_data = [
        {
            "input": "Hi! How can I get started with machine learning?",
            "output": "Great question! Start with: 1) Learn Python basics, 2) Study fundamental math (linear algebra, calculus, statistics), 3) Take online courses (Coursera's ML course, fast.ai), 4) Practice on Kaggle with beginner competitions, 5) Build small projects, 6) Read papers and blogs. Focus on understanding concepts before diving into complex models."
        },
        {
            "input": "What should I learn first in AI?",
            "output": "Begin with foundational concepts: 1) Python programming and NumPy, 2) Basic machine learning algorithms (linear regression, decision trees), 3) Neural network fundamentals, 4) PyTorch or TensorFlow basics, 5) Computer vision or NLP (choose one domain first). Don't rush to advanced topics - solid foundations are crucial."
        },
        {
            "input": "I'm confused about neural networks. Can you help?",
            "output": "Neural networks are inspired by biological neurons. They consist of layers of interconnected nodes (neurons). Each connection has a weight that's learned during training. Data flows forward through layers, each applying weights and activation functions. Training adjusts weights using backpropagation to minimize prediction errors. Think of it as learning from mistakes iteratively."
        },
        {
            "input": "What projects should beginners build?",
            "output": "Beginner-friendly projects: 1) MNIST digit recognition (computer vision basics), 2) Sentiment analysis on movie reviews (NLP basics), 3) House price prediction (regression), 4) Iris flower classification (classic ML), 5) Simple chatbot, 6) Stock price visualization. Start small, gradually increase complexity, and focus on understanding over performance."
        },
        {
            "input": "How long does it take to learn AI?",
            "output": "Timeline varies by background and time commitment. With consistent effort: 3-6 months for basics (Python, ML fundamentals), 6-12 months for intermediate skills (deep learning, frameworks), 1-2 years for professional competency. Daily practice is more important than duration. Focus on building projects alongside learning theory."
        }
    ]
    
    # すべてのカテゴリを結合
    dataset.extend(technical_data)
    dataset.extend(coding_data)
    dataset.extend(problem_solving)
    dataset.extend(comparison_data)
    dataset.extend(conversational_data)
    
    return dataset


# ============================================================
# 評価メトリクス
# ============================================================

def calculate_keyword_coverage(response: str, keywords: List[str]) -> float:
    """
    応答中のキーワードカバレッジを計算
    
    Args:
        response: モデルの応答
        keywords: 期待されるキーワードリスト
    
    Returns:
        カバレッジスコア (0.0-1.0)
    """
    if not keywords:
        return 1.0
    
    response_lower = response.lower()
    matched = sum(1 for kw in keywords if kw.lower() in response_lower)
    return matched / len(keywords)


def check_response_quality(response: str) -> Dict[str, Any]:
    """
    応答品質を複数の基準でチェック
    
    Args:
        response: モデルの応答
    
    Returns:
        品質評価の辞書
    """
    checks = {
        "has_content": len(response.strip()) > 0,
        "minimum_length": len(response.split()) >= 10,
        "not_too_short": len(response.split()) >= 20,
        "not_too_long": len(response.split()) <= 300,
        "not_repetitive": not _is_repetitive(response),
        "no_gibberish": not _contains_gibberish(response),
        "proper_sentence": response.strip()[-1] in '.!?' if response.strip() else False,
        "no_code_artifacts": "<" not in response and ">" not in response,
    }
    
    score = sum(checks.values()) / len(checks)
    
    return {
        "score": score,
        "checks": checks,
        "word_count": len(response.split())
    }


def _is_repetitive(text: str, threshold: float = 0.4) -> bool:
    """繰り返しが多いかチェック"""
    words = text.split()
    if len(words) < 10:
        return False
    
    unique_ratio = len(set(words)) / len(words)
    return unique_ratio < threshold


def _contains_gibberish(text: str) -> bool:
    """意味不明な文字列を含むかチェック"""
    # 同じ文字が5回以上連続
    if re.search(r'(.)\1{4,}', text):
        return True
    
    # 子音のみが6文字以上連続
    if re.search(r'[bcdfghjklmnpqrstvwxyz]{6,}', text.lower()):
        return True
    
    return False


# ============================================================
# 評価レポート生成
# ============================================================

def generate_evaluation_report(results: Dict[str, Any], output_path: str) -> None:
    """
    評価結果からMarkdownレポートを生成
    
    Args:
        results: 評価結果の辞書
        output_path: 出力ファイルパス
    """
    md = "# Model Evaluation Report\n\n"
    md += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # サマリー
    if "summary" in results:
        summary = results["summary"]
        md += "## Summary\n\n"
        md += f"- **Overall Score:** {summary.get('overall_score', 0):.2%}\n"
        md += f"- **Keyword Coverage:** {summary.get('avg_keyword_coverage', 0):.2%}\n"
        md += f"- **Quality Score:** {summary.get('avg_quality_score', 0):.2%}\n"
        md += f"- **Avg Response Length:** {summary.get('avg_response_length', 0):.1f} words\n"
        md += f"- **Total Tests:** {summary.get('total_tests', 0)}\n\n"
    
    # 詳細結果
    if "test_cases" in results:
        md += "## Detailed Results\n\n"
        for i, test in enumerate(results["test_cases"], 1):
            md += f"### Test {i}: {test.get('category', 'Unknown')}\n\n"
            md += f"**Input:** {test.get('input', '')}\n\n"
            md += f"**Response:**\n> {test.get('response', '')}\n\n"
            md += f"**Metrics:**\n"
            md += f"- Keyword Coverage: {test.get('keyword_coverage', 0):.1%}\n"
            md += f"- Quality Score: {test.get('quality_score', 0):.1%}\n"
            md += f"- Response Length: {test.get('response_length', 0)} words\n\n"
            md += "---\n\n"
    
    # ファイル保存
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md)
    
    print(f"✅ Evaluation report saved to: {output_path}")


# ============================================================
# 実験比較
# ============================================================

def compare_experiments(exp_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    複数の実験結果を比較
    
    Args:
        exp_results: 実験結果のリスト
    
    Returns:
        比較結果の辞書
    """
    comparison = {
        "experiments": [],
        "best_experiment": None,
        "metrics_comparison": {}
    }
    
    best_score = -1
    best_exp = None
    
    for exp in exp_results:
        exp_summary = {
            "name": exp.get("name", "Unknown"),
            "overall_score": exp.get("summary", {}).get("overall_score", 0),
            "keyword_coverage": exp.get("summary", {}).get("avg_keyword_coverage", 0),
            "quality_score": exp.get("summary", {}).get("avg_quality_score", 0),
        }
        
        comparison["experiments"].append(exp_summary)
        
        # 最良実験の特定
        if exp_summary["overall_score"] > best_score:
            best_score = exp_summary["overall_score"]
            best_exp = exp_summary
    
    comparison["best_experiment"] = best_exp
    
    return comparison


# ============================================================
# データセット検証
# ============================================================

def validate_dataset(dataset: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    データセットの品質を検証
    
    Args:
        dataset: データセットのリスト
    
    Returns:
        検証結果の辞書
    """
    validation = {
        "total_samples": len(dataset),
        "issues": [],
        "statistics": {}
    }
    
    input_lengths = []
    output_lengths = []
    
    for i, item in enumerate(dataset):
        # 必須フィールドチェック
        if "input" not in item or "output" not in item:
            validation["issues"].append(f"Sample {i}: Missing required fields")
            continue
        
        # 空文字列チェック
        if not item["input"].strip() or not item["output"].strip():
            validation["issues"].append(f"Sample {i}: Empty input or output")
        
        # 長さの記録
        input_lengths.append(len(item["input"].split()))
        output_lengths.append(len(item["output"].split()))
    
    # 統計情報
    if input_lengths:
        validation["statistics"] = {
            "input_length": {
                "avg": np.mean(input_lengths),
                "min": np.min(input_lengths),
                "max": np.max(input_lengths)
            },
            "output_length": {
                "avg": np.mean(output_lengths),
                "min": np.min(output_lengths),
                "max": np.max(output_lengths)
            }
        }
    
    validation["is_valid"] = len(validation["issues"]) == 0
    
    return validation


if __name__ == "__main__":
    # テスト: データセット生成
    dataset = generate_enhanced_dataset()
    print(f"✅ Generated {len(dataset)} samples")
    
    # テスト: データセット検証
    validation = validate_dataset(dataset)
    print(f"✅ Validation: {validation['is_valid']}")
    print(f"   Issues: {len(validation['issues'])}")
    
    # テスト: 品質チェック
    test_response = "This is a test response with proper length and structure."
    quality = check_response_quality(test_response)
    print(f"✅ Quality score: {quality['score']:.2%}")