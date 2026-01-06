import json
import os
from collections import Counter
from transformers import GPT2Tokenizer

# ===== 設定 =====
DATA_PATH = "./training_data.json"
OUTPUT_PATH = "./data_statistics.md"

# ===== トークナイザー準備 =====
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# ===== データ読み込み =====
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

num_samples = len(data)

# ===== 統計情報算出 =====
prompt_lengths_chars = [len(item["input"]) for item in data]
completion_lengths_chars = [len(item["output"]) for item in data]

prompt_lengths_tokens = [len(tokenizer.encode(item["input"])) for item in data]
completion_lengths_tokens = [len(tokenizer.encode(item["output"])) for item in data]

avg_prompt_tokens = sum(prompt_lengths_tokens) / num_samples
avg_completion_tokens = sum(completion_lengths_tokens) / num_samples

avg_prompt_chars = sum(prompt_lengths_chars) / num_samples
avg_completion_chars = sum(completion_lengths_chars) / num_samples

# ===== トップ単語抽出 =====
def get_top_words(texts, n=10):
    words = " ".join(texts).lower().split()
    counts = Counter(words)
    return counts.most_common(n)

top_prompt_words = get_top_words([d["input"] for d in data])
top_completion_words = get_top_words([d["output"] for d in data])

# ===== Markdown生成 =====
md_content = f"""# Data Statistics Report — GPT-2 Chatbot Fine-tuning Dataset

## Dataset Overview
- **Dataset Name:** GPT-2 Chatbot Conversational Dataset  
- **Records:** {num_samples} dialogues  
- **Format:** JSON (`input`, `output`)  
- **Encoding:** UTF-8  

---

## Basic Statistics

| Metric | Prompt | Completion |
|--------|--------|------------|
| **Total Samples** | {num_samples} | {num_samples} |
| **Average Token Length** | {avg_prompt_tokens:.2f} | {avg_completion_tokens:.2f} |
| **Average Character Length** | {avg_prompt_chars:.2f} | {avg_completion_chars:.2f} |
| **Max Tokens** | {max(prompt_lengths_tokens)} | {max(completion_lengths_tokens)} |
| **Min Tokens** | {min(prompt_lengths_tokens)} | {min(completion_lengths_tokens)} |

---

## Top 10 Common Words

### Prompts
{"".join([f"- {w[0]} ({w[1]})\n" for w in top_prompt_words])}

### Completions
{"".join([f"- {w[0]} ({w[1]})\n" for w in top_completion_words])}

---

## Notes
- Prompts are short, typically 5–10 tokens.  
- Completions are conversational and friendly (10–15 tokens on average).  
- Dataset suitable for small-scale GPT-2 fine-tuning experiments.  

---

## Next Steps
1. Add multi-turn dialogues (contextual memory).  
2. Include tone variations ("formal", "casual").  
3. Add more domain-specific topics (science, culture, daily talk).  
4. Monitor loss and perplexity in future training logs.

---

_This file was auto-generated via `generate_data_statistics.py`._
"""

# ===== ファイル出力 =====
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(md_content)

print(f" Data statistics generated and saved to {OUTPUT_PATH}")
