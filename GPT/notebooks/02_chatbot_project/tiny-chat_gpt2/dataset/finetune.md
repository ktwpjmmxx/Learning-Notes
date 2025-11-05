#### 必要なライブラリのインストール
!pip install datasets transformers torch -q

from datasets import load_dataset

#### 試してみるデータセット一覧
datasets_to_try = [
    ("Anthropic/hh-rlhf", "HH-RLHF (Anthropicの高品質対話)"),
    ("OpenAssistant/oasst1", "Open Assistant対話"),
    ("allenai/prosocial-dialog", "Prosocial Dialog"),
]

successful_dataset = None

for dataset_name, description in datasets_to_try:
    try:
        print(f"試行中: {description}...")
        dataset = load_dataset(dataset_name)
        print(f"✓ 成功: {description}")
        print(f"  データ数: {len(dataset['train'])} 件")
        successful_dataset = dataset
        break
    except Exception as e:
        print(f"✗ 失敗: {description}")
        print(f"  エラー: {str(e)[:100]}...")
        continue

if successful_dataset:
    print("\n=== データセットの例 ===")
    print(successful_dataset['train'][0])
else:
    print("\n全てのデータセットで失敗しました。自作データセットを使いましょう。")

#### データセットの詳細を確認

print("=== データセット情報 ===")
print(f"学習データ数: {len(dataset['train'])}")
if 'validation' in dataset:
    print(f"検証データ数: {len(dataset['validation'])}")
if 'test' in dataset:
    print(f"テストデータ数: {len(dataset['test'])}")

print("\n=== データの構造 ===")
print(dataset['train'].features)

print("\n=== 最初の3つの会話例 ===")
for i in range(min(3, len(dataset['train']))):
    print(f"\n--- 例 {i+1} ---")
    print(dataset['train'][i])

#### ファインチューニングの実施

from transformers import GPT2LMHeadModel, Trainer, TrainingArguments
import torch

# GPUが使えるか確認
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用デバイス: {device}")

# モデルの読み込み
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.to(device)

# 学習設定
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",           # 保存先
    num_train_epochs=3,                       # 学習回数（3回繰り返し）
    per_device_train_batch_size=4,            # バッチサイズ（小さめ）
    per_device_eval_batch_size=4,
    warmup_steps=100,                         # ウォームアップ
    weight_decay=0.01,                        # 重み減衰
    logging_dir='./logs',                     # ログ保存先
    logging_steps=50,                         # 50ステップごとにログ
    eval_strategy="steps",                    # 評価タイミング
    eval_steps=200,                           # 200ステップごとに評価
    save_steps=200,                           # 200ステップごとに保存（eval_stepsと同じに修正）
    save_total_limit=2,                       # 最新2つのみ保存
    load_best_model_at_end=True,              # 最良モデルを読み込む
    report_to="none",                         # 外部ツール連携なし
)

# Trainerの設定
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
)

# ファインチューニング開始！
print("\nファインチューニング開始！")
print("=" * 50)
trainer.train()

print("\nファインチューニング完了！")

# モデルを保存
model.save_pretrained("./gpt2-finetuned-final")
tokenizer.save_pretrained("./gpt2-finetuned-final")
print("モデルを保存しました: ./gpt2-finetuned-final")