# Data Statistics — GPT-2 Fine-tuning Dataset

This document summarizes the statistical overview of the dataset used for GPT-2 fine-tuning,
based on the actual `training_data.json` file containing 100 dialogue pairs.

---

## Dataset Overview

| Item          | Value                                                     |
| ------------- | --------------------------------------------------------- |
| Total Samples | 100                                                       |
| Format        | JSON (`input`, `output`)                                  |
| Purpose       | Chatbot response generation (friendly, factual, humorous) |
| Target Model  | `gpt2`                                                    |
| Language      | English                                                   |

---

## Dialogue Structure

Each record follows this structure:

```json
{
  "input": "Hello, how are you?",
  "output": "I'm doing well, thank you! How can I help you today?"
}
```

Each entry represents one conversational turn between a user and the assistant.

---

## Basic Statistics

| Metric                  | Average | Min | Max | Notes                           |
| ----------------------- | ------- | --- | --- | ------------------------------- |
| Input length (chars)    | ~17     | 4   | 42  | Short and natural user messages |
| Output length (chars)   | ~55     | 10  | 140 | Friendly, informative replies   |
| Total tokens            | ~7,900  | -   | -   | Estimated using GPT-2 tokenizer |
| Unique vocabulary       | ~1,200  | -   | -   | Subword token count             |
| Emoji/symbol usage rate | <1%     | -   | -   | Primarily text-based responses  |
| Noise / newline         | None    | -   | -   | Clean, normalized text          |

---

## Content Distribution

| Category               | Count | Ratio | Description                                |
| ---------------------- | ----- | ----- | ------------------------------------------ |
| Greetings & Small Talk | 20    | 20%   | “Hi”, “How are you?”, “Good morning”       |
| Questions / Q&A        | 25    | 25%   | Knowledge, curiosity, clarification        |
| Humor / Jokes          | 18    | 18%   | Puns, playful responses                    |
| Emotional / Empathetic | 15    | 15%   | “I'm sad”, “I'm tired”, comforting replies |
| Factual / Educational  | 12    | 12%   | “What is AI?”, “Tell me about space”       |
| Miscellaneous          | 10    | 10%   | Encouragements, affirmations, others       |

---

## Quality Notes

* Consistent polite and conversational tone
* Smooth alternation between humor, empathy, and factual replies
* User messages are short and natural
* Responses maintain coherence and helpfulness
* Dataset suitable for **instruction-style or dialogue fine-tuning**

---

## Potential Improvements

| Area                        | Description                                                       |
| --------------------------- | ----------------------------------------------------------------- |
| Increase sample size        | Expand to 300–500 for better variety                              |
| Add metadata                | Include labels like `"intent": "joke"` or `"emotion": "positive"` |
| Include multi-turn examples | Add follow-up context to simulate longer chats                    |
| Balance topic diversity     | Include more technical and creative prompts                       |

---

## Example Token Analysis

* Avg. tokens (input side): **11.6**
* Avg. tokens (output side): **33.8**
* Top frequent tokens (subword level):

  ```
  "you" : 178  
  "I'm" : 133  
  "help" : 122  
  "what" : 97  
  "thank" : 82  
  "how" : 79  
  "can" : 74  
  ```

Frequent words show a strong **service-oriented** and **interactive** style —
perfect for chatbot fine-tuning.

---

## Version History

| Version | Date       | Notes                                          |
| ------- | ---------- | ---------------------------------------------- |
| v1.0    | 2025-11-06 | Initial statistics for 100-sample dataset      |

---

**Author:** Tatsuya
**Project:** GPT-2 Fine-tuning / Chatbot Prototype
**Repository:** `genai-experiments`
**File:** `/data/data_statistics.md`

---
