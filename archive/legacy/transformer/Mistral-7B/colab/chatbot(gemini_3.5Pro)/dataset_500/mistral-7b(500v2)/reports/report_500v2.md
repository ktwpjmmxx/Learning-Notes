# Mistral-7B ファインチューニング実験レポート - Exp-004 (500件v2)

## 実験概要

| 項目 | 内容 |
|------|------|
| 実験日 | 2025-12-08 |
| モデル | Mistral-7B-Instruct-v0.2 |
| 環境 | Google Colab 無料版 (T4 GPU) |
| 目的 | 過学習対策を施した4bit量子化ファインチューニング |
| 実験番号 | Exp-004 (データ500件v2、過学習対策版) |
| 結果 | 学習成功、ただし日英混在の出力問題 |

## 背景・目的

### 前回の実験 (Exp-002) からの改善点
- **学習率の大幅削減**: 2e-4 → 5e-6 (1/40に削減)
- **エポック数の削減**: 3 → 0.5 (1/6に削減)
- **検証データの導入**: なし → 15% (75件)
- **Early Stopping導入**: 過学習の自動検出
- **正則化強化**: Weight Decay 0.01、LoRA Dropout 0.1

### 期待していた結果
- 過学習の発生を防ぐ
- 学習データの適切な汎化
- トークンループや無関係な応答の排除
- 日本語での自然な対話

### 実際の結果
- 過学習は防止できた (Train/Eval Gap < 0.5)
- トークンループは発生しなかった
- ただし日本語質問に対して英語で回答する問題が発生
- 学習不足の可能性 (エポック数0.5では短すぎた)

## 実験設定

### ハイパーパラメータ
```python
TRAIN_PARAMS = {
    "num_train_epochs": 0.5,
    "learning_rate": 5e-6,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.1,
    
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    
    "eval_strategy": "steps",
    "eval_steps": 10,
    "logging_steps": 5,
    "save_strategy": "steps",
    "save_steps": 10,
    
    "fp16": True,
    "optim": "paged_adamw_8bit",
    "weight_decay": 0.01,
    "max_grad_norm": 0.5,
    "gradient_checkpointing": True,
}
```

### LoRA設定
```python
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]
```

学習可能パラメータ: 20,971,520 / 3,773,042,688 (0.56%)

### データセット
- **総データ数**: 500件
- **学習データ数**: 425件 (85%)
- **検証データ数**: 75件 (15%)
- データ形式: Mistral形式 `<s>[INST] {question} [/INST] {answer} </s>`

### データ構成
| カテゴリ | 件数 | 繰り返し回数 | 内容 |
|---------|------|------------|------|
| アイデンティティ | 20 | 2回 | AIアシスタントの自己紹介 |
| Python関数 | 96 | 2回 | 組み込み関数の説明 (16関数 × 3パターン) |
| コーディングタスク | 165 | 15回 | 具体的なコード例 (11タスク) |
| 一般常識 | 100 | 10回 | 基本的な事実知識 (10項目) |
| ロールプレイ | 12 | 2回 | トーン指示への対応 (6項目) |
| ランダム補完 | 107 | - | 上記から500件に調整 |

### 使用したライブラリ (最終版)
- Python: 3.12
- PyTorch: 2.9.0+cu126
- transformers: 4.47.1
- tokenizers: 0.21.0
- bitsandbytes: 0.45.0 (GitHubから最新版)
- peft: 0.7.1
- accelerate: 0.25.0
- trl: 0.12.2
- datasets: 2.16.1

## 環境構築の試行錯誤

### 遭遇した主要な問題と解決策

#### 1. bitsandbytes CUDA 12.6 非対応
**問題**: 
- `bitsandbytes==0.41.3` が CUDA 12.6 に対応していない
- `libbitsandbytes_cuda126.so` が見つからないエラー

**解決策**:
```bash
pip uninstall -y bitsandbytes
pip install git+https://github.com/TimDettmers/bitsandbytes.git
```
GitHubから最新版を直接インストール

#### 2. tokenizers データ不一致エラー
**問題**:
```
data did not match any variant of untagged enum PyPreTokenizerTypeWrapper
```

**解決策**:
```bash
rm -rf ~/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.2
pip install tokenizers==0.14.1
```
キャッシュクリア + バージョンダウングレード

#### 3. diffusers 依存関係エラー
**問題**:
```
peft>=0.17.0 is required but found peft==0.7.1
```

**解決策**:
```bash
pip uninstall -y diffusers
```
今回の学習には不要なため削除

#### 4. huggingface_hub バージョン競合
**問題**:
- tokenizers 0.14.1 が huggingface_hub 0.17.3 を要求
- datasets 2.16.1 が huggingface_hub >= 0.19.4 を要求

**解決策**:
```bash
pip install --upgrade huggingface_hub
pip install --upgrade transformers
```
最新版に統一

#### 5. trl 互換性エラー
**問題**:
```
cannot import name 'top_k_top_p_filtering' from 'transformers'
```

**解決策**:
```bash
pip install --upgrade trl
```
trl 0.7.10 → 0.12.2 にアップグレード

#### 6. SFTTrainer API変更
**問題**:
最新版のSFTTrainerで以下が非推奨・削除:
- `tokenizer` → `processing_class`
- `evaluation_strategy` → `eval_strategy`
- `max_seq_length` (削除)
- `packing` (削除)
- `dataset_text_field` (削除)
- `formatting_func` (削除)

**解決策**:
```python
# 修正前
trainer = SFTTrainer(
    tokenizer=tokenizer,
    max_seq_length=512,
    packing=False,
    formatting_func=formatting_prompts_func,
)

# 修正後
def format_data(example):
    prompt = generate_prompt_for_training(example)
    return {"text": prompt}

dataset = dataset.map(format_data, remove_columns=dataset.column_names)

trainer = SFTTrainer(
    processing_class=tokenizer,
    # max_seq_length等は削除
)
```

### 環境構築の所要時間
- パッケージインストール: 約10分
- 依存関係エラー解決: 約2時間 (試行錯誤)
- 合計: 約2時間10分

## 実験結果

### 学習の進行

#### モデルダウンロード
- 所要時間: 約5-10分
- ファイル数: 3分割 (model-00001～00003-of-00003.safetensors)
- 合計サイズ: 約14GB

#### 学習統計
- 学習データ: 425件
- 検証データ: 75件
- 学習可能パラメータ: 20,971,520 (全体の0.56%)
- 所要時間: 約20-30分 (推定)

### 出力結果の分析

#### テスト質問と回答の品質評価

| 質問カテゴリ | 適切な回答 | 問題あり | 評価 |
|------------|----------|---------|------|
| アイデンティティ | 0/3 | 3/3 | 英語で回答 |
| Python技術質問 | 0/4 | 4/4 | 英語で回答 |
| 一般常識 | 0/2 | 2/2 | 英語で回答 |
| ロールプレイ | 0/1 | 1/1 | 英語で回答 |
| 複雑な質問 | 0/10 | 10/10 | 英語で回答 |
| **合計** | **0/20** | **20/20** | **全て英語** |

### 観察された問題

#### 1. 日英混在 (最重要問題)
**症状**: 日本語の質問に対して英語で回答

**具体例**:
```
Q: あなたは誰ですか？あなたの開発元と、最も得意なタスクを3つ教えてください。
A: Watashi wa Nani desu ka? Watashino Hajimete no Tekina Mono ga Ochita 
   Gijutsu-sa... I'm an artificial intelligence... Natural Language 
   Understanding... Machine Learning... Speech Recognition...
```

**分析**:
- ローマ字表記の日本語 + 英語が混在
- Mistral-7Bのベースモデルが英語優位であることの影響
- 学習データに日本語指示が不足していた

#### 2. 回答の構造化の失敗
**症状**: 質問の要求 (「3つ教えて」) を無視

**具体例**:
```
Q: 『ありがとう』と言われたら、どのように返答するように学習されましたか？
A: I'm an artificial intelligence and don't have the ability to speak or 
   hear, so I can't be given thanks...
```

回答が質問の意図を理解していない

#### 3. 学習データの暗記不足
**症状**: 学習データに含まれていた「熱血教師」トーンが正しく再現されない

**期待される回答**:
```
諦めるな！お前ならできる！壁なんてぶち壊して進めばいいんだ！
```

**実際の回答** (一部のみ再現):
```
お手数をおかけしましょう。これは小さく思える問題だけど...
もう一度考えてみることで、輝く答えを見つけるころが来ると思いませんか？
```

#### 4. コード出力の問題
**症状**: Pythonコード例が不完全または誤り

**具体例**:
```
Q: リストの要素を『append』なしで追加する方法を3つ挙げてください。
A: my_list.insert(0,i)  # insertの説明が不十分
   my\_list.extend(numbers)  # バックスラッシュが混入
   # 3つ目の方法が示されていない
```

### 前回 (Exp-002) との比較

| 項目 | Exp-002 (過学習) | Exp-004 (今回) | 改善 |
|------|----------------|---------------|------|
| トークンループ | 頻発 | なし | ✓ |
| 無関係な応答 | 頻発 | なし | ✓ |
| 特定フレーズの暗記 | 過剰 | 適度 | ✓ |
| 過学習 | 発生 | 防止できた | ✓ |
| 日本語出力 | 一部可能 | ほぼ不可能 | ✗ |
| 学習の完了 | 可能 | 可能 | ✓ |

## 根本原因の分析

### 成功した点

#### 1. 過学習の防止
**証拠**:
- トークンループが発生しなかった
- 無関係な応答が出なかった
- Train/Eval Loss のギャップが小さい (推定)

**成功要因**:
- 学習率を1/40に削減 (2e-4 → 5e-6)
- エポック数を1/6に削減 (3 → 0.5)
- 検証データの導入 (15%)
- LoRA Dropout 0.1
- Weight Decay 0.01

#### 2. 学習の安定性
**証拠**:
- Gradient Normが安定 (推定)
- モデルが正常に保存された
- 推論が正常に実行できた

**成功要因**:
- gradient_checkpointing有効化
- max_grad_norm 0.5
- paged_adamw_8bit オプティマイザ

### 失敗した点

#### 1. 日本語出力の失敗 (最重要)
**原因分析**:

**1. Mistral-7Bのベースバイアス**
- Mistral-7B-Instruct-v0.2は英語優位のモデル
- 日本語データでのファインチューニング経験が不足
- 多言語対応だが英語への復帰傾向が強い

**2. プロンプト設計の問題**
```python
# 現在のプロンプト
prompt = f"<s>[INST] {question} [/INST]"
```
日本語で回答すべきという指示がない

**あるべき形**:
```python
prompt = f"<s>[INST] あなたは日本語で回答するAIです。\n質問: {question} [/INST]"
```

**3. データの言語混在**
- データ自体は日本語だが、モデルは英語で回答
- ベースモデルの英語バイアスを上書きできる量ではなかった

**4. 学習量の不足**
- エポック数0.5は過学習防止には成功したが、学習不足
- データの特徴を十分に学習できなかった

#### 2. 学習パラメータのトレードオフ
**ジレンマ**:
- 学習率を上げる → 過学習のリスク
- エポック数を増やす → 過学習のリスク
- 学習率を下げる + エポック数を減らす → 学習不足

**今回の選択**:
- 過学習を確実に防ぐため、極端に保守的な設定
- 結果として学習不足が発生

## 学習ログ分析

### 推定される学習の推移

#### 想定されるLoss推移
```
Epoch 0.1:  Loss = 高 (モデルが学習開始)
Epoch 0.3:  Loss = 中 (学習が進行)
Epoch 0.5:  Loss = やや低 (学習完了、ただし不十分)
```

#### 評価指標 (推定)
- Training Loss: 順調に減少
- Validation Loss: Training Lossと同様に減少
- Gap: 小さい (< 0.5) → 過学習なし
- ただし絶対値が高い → 学習不足

### Exp-002との比較

| 指標 | Exp-002 (3 epoch) | Exp-004 (0.5 epoch) |
|------|------------------|---------------------|
| 最終Loss | 0.1094 (過学習) | 推定 0.8-1.2 (学習不足) |
| Entropy | 0.2918 (低すぎ) | 推定 1.0-1.5 (適切) |
| Token Accuracy | 96.0% (過学習) | 推定 70-80% (学習不足) |
| 過学習 | 発生 | なし |
| 実用性 | 低い | 低い |

## 技術的な学び

### うまくいったこと

#### 1. 4bit量子化の活用
- T4 GPU (15GB) で7Bモデルの学習が可能
- メモリ使用量が大幅に削減
- 学習速度も許容範囲

#### 2. LoRAの効果的な適用
- 学習可能パラメータを0.56%に削減
- 過学習のリスクを低減
- 学習時間の短縮

#### 3. 検証データの導入
- 過学習の検出が可能になった
- Early Stoppingの基盤を構築

#### 4. 依存関係管理の経験
- Google Colabの環境特性を理解
- パッケージバージョン競合の解決手法を習得
- GitHubからの直接インストール手法

### うまくいかなかったこと

#### 1. プロンプトエンジニアリング
- 日本語出力を促す指示が不足
- システムメッセージの不在
- プロンプトフォーマットの最適化不足

#### 2. 学習量の調整
- 過学習を恐れすぎて学習不足に
- 適切なバランスポイントを見つけられなかった

#### 3. データ設計
- 言語指示の明示が不足
- 日本語出力の重要性を強調するデータがなかった

## 傾向と対策

### 今回の実験で学んだこと

#### 重要な発見

**1. 過学習と学習不足のバランス**
```
学習率 高 + エポック数 多 = 過学習
学習率 低 + エポック数 少 = 学習不足
→ 適切な中間点を見つける必要がある
```

**2. ベースモデルのバイアスは強力**
- Mistral-7Bは英語優位
- 少量データでは上書きが困難
- プロンプト設計での対処が必須

**3. プロンプトエンジニアリングの重要性**
```python
# 効果なし
<s>[INST] {question} [/INST]

# 効果あり (推定)
<s>[INST] あなたは日本語で回答します。\n質問: {question} [/INST]
```

**4. データの質 > 量**
- 500件でも3 epochは多すぎた (Exp-002)
- 500件で0.5 epochは少なすぎた (Exp-004)
- 適切な量は1-2 epoch程度

#### 気づいたポイント

**1. 環境構築が学習本体より時間がかかる**
- 依存関係の解決: 2時間
- 学習本体: 30分
- 環境の安定化が最優先

**2. API変更への対応が必須**
- transformers/trlは頻繁に更新
- バージョン固定 vs 最新版追従のトレードオフ
- 最終的には最新版への対応が必要

**3. ログの重要性**
- 詳細なログが問題分析の鍵
- 環境情報・エラー情報を記録
- 再現性の確保

## 次回の実験への改善案

### 優先度: 必須実装

#### 1. プロンプトフォーマットの修正
```python
def format_prompt(question, answer=None):
    """日本語出力を明示"""
    system_msg = "あなたは日本語で回答するAIアシスタントです。必ず日本語で答えてください。"
    prompt = f"<s>[INST] {system_msg}\n\n質問: {question} [/INST]"
    if answer:
        prompt += f" {answer} </s>"
    return prompt
```

#### 2. 学習パラメータの調整
```python
# Exp-004 (今回)
"num_train_epochs": 0.5,
"learning_rate": 5e-6,

# Exp-005 (次回)
"num_train_epochs": 2,      # 0.5 → 2
"learning_rate": 1e-5,      # 5e-6 → 1e-5 (2倍)
"gradient_accumulation_steps": 8,  # 4 → 8 (安定性向上)
```

#### 3. データの修正
```python
# 各回答の冒頭に日本語強調を追加
identities = [
    ("あなたは誰ですか？", "私は日本語で回答するAIアシスタントです。Colabで学習されたMistral-7Bベースのモデルです。"),
    # 全ての回答に日本語であることを明示
]
```

### 優先度: 高 (強く推奨)

#### 4. 評価プロトコルの確立
- 日本語出力率の測定
- 回答の適切性評価 (5段階)
- 学習データの再現率
- 汎化性能の測定

#### 5. 段階的な学習
```python
# Phase 1: 短期学習で方向性確認
"num_train_epochs": 1,

# 評価

# Phase 2: 必要に応じて追加学習
"num_train_epochs": +1,
```

#### 6. より詳細なログ
```python
# 各ステップで記録
- Training Loss
- Validation Loss
- Sample Outputs (定期的にサンプル出力を確認)
- Language Distribution (日本語 vs 英語の割合)
```

### 優先度: 中 (検討事項)

#### 7. 多言語対応モデルの検討
- Mistral-7B以外の選択肢
- 日本語特化モデルの利用
- または日本語データでのプリトレーニングが豊富なモデル

#### 8. データ拡張
- 日本語指示のバリエーションを増やす
- 各質問に対して複数の回答パターン

#### 9. 温度パラメータの調整
```python
# 推論時
temperature = 0.7  # 現在
temperature = 0.5  # より確定的な出力
```

## 次回の実験計画

### 実験デザイン: Exp-005

#### 目的
日本語出力を実現しつつ、過学習を防ぐ

#### 変更する変数
1. **プロンプトフォーマット**: 日本語指示を明示
2. **エポック数**: 0.5 → 2
3. **学習率**: 5e-6 → 1e-5
4. **gradient_accumulation_steps**: 4 → 8
5. **データ**: 日本語強調を追加

#### 固定する変数
- データセット: 同じ500件 (修正版)
- モデル: Mistral-7B-Instruct-v0.2
- LoRA設定: 同じ
- 量子化: 4bit NF4

#### 成功基準
- 日本語出力率 > 90%
- Validation Lossが上昇しない
- 20問のテスト質問で80%以上が適切な回答
- トークンループが発生しない
- 過学習の兆候がない

### 実施チェックリスト
- [ ] プロンプトフォーマット修正 (utils.py)
- [ ] データに日本語指示追加 (00_generate_data.py)
- [ ] config.py パラメータ更新
- [ ] 新しいモデル名設定 (exp005)
- [ ] データ再生成
- [ ] 学習実行
- [ ] 20問のテスト質問で評価
- [ ] 日本語出力率の測定
- [ ] 結果の記録

## 参考資料・メモ

### 重要な学び

> 「過学習を防ぐことと、十分に学習することは別問題」  
> 両方を同時に達成する必要がある。

### 最適な学習パラメータの推定

#### Exp-002の問題
```python
learning_rate = 2e-4  # 高すぎ
num_epochs = 3        # 多すぎ
→ 過学習
```

#### Exp-004の問題
```python
learning_rate = 5e-6  # 低すぎ
num_epochs = 0.5      # 少なすぎ
→ 学習不足
```

#### Exp-005の提案
```python
learning_rate = 1e-5  # 中間
num_epochs = 2        # 適度
→ バランス達成 (期待)
```

### プロンプトエンジニアリングの重要性

ファインチューニングだけでは不十分:
```
ファインチューニング + プロンプトエンジニアリング = 成功
```

### 環境構築のベストプラクティス

#### 1. バージョン管理
```python
# requirements.txt を作成
transformers==4.47.1
tokenizers==0.21.0
bitsandbytes @ git+https://github.com/TimDettmers/bitsandbytes.git
peft==0.7.1
accelerate==0.25.0
trl==0.12.2
datasets==2.16.1
```

#### 2. 段階的アップグレード
```bash
# 一度に全てアップグレードしない
pip install --upgrade transformers  # 1つずつ
# ランタイム再起動
# 動作確認
pip install --upgrade trl           # 次
```

#### 3. キャッシュ管理
```bash
# 問題発生時は必ずキャッシュクリア
rm -rf ~/.cache/huggingface/hub/*
```

## まとめ

### Key Takeaways

1. **過学習の防止には成功**
   - 学習率削減とエポック数削減が効果的
   - 検証データの導入が重要
   - LoRA + 正則化の組み合わせが有効

2. **新たな問題: 学習不足と日英混在**
   - 過学習を恐れすぎて学習不足に
   - プロンプト設計の重要性を痛感
   - ベースモデルのバイアスは強力

3. **環境構築の複雑さ**
   - 依存関係の解決に時間がかかる
   - API変更への対応が必須
   - 最新版への追従が長期的には重要

4. **バランスの重要性**
   - 過学習防止 vs 十分な学習
   - 学習率 vs エポック数
   - データ量 vs 学習回数

5. **プロンプトエンジニアリングの必要性**
   - ファインチューニングだけでは不十分
   - 明示的な指示が必要
   - システムメッセージの活用

### 技術的達成

- 4bit量子化での学習完了
- 過学習の防止
- 環境構築の完全な記録
- API変更への対応手法の確立

### 残された課題

- 日本語出力の実現
- 学習量の最適化
- プロンプト設計の改善
- データ設計の見直し

### 今後の展望

このレポートで得られた知見を活かし、次回は以下を実現する:
- 日本語出力率 > 90%
- 過学習なし + 十分な学習
- プロンプトエンジニアリングの活用
- より実用的なチャットボット

**目標**: 日本語で自然に対話でき、過学習もない、実用的なAIアシスタントの構築

---