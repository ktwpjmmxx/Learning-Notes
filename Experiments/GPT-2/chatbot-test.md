# --- Colab用：GPT-2 チャットボット（ver.1） ---

**今回の稼働テスト条件**
- プロンプトは常に以下を使用：Could you tell me about your name ?(必要に応じて変更)
- 入力言語は英語のみを使用
- 生成される応答の安定性と脱線防止を確認する

**最終目標**
- しっかり質疑応答ができるチャットボット
- 同じプロンプトを入力しても回答に多様性があるもの
- UIが見やすいものにする


!pip install transformers torch gradio --quiet

import gradio as gr
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

### モデルロード
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def respond(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs,
        max_length=100,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

### Gradio インターフェース
iface = gr.Interface(
    fn=respond,
    inputs="text",
    outputs="text",
    title="GPT-2 チャットボット",
    description="ここにテキストを入力すると GPT-2 が返答します"
)

iface.launch()

**1回目の稼働テスト**

prompt： Could you tell me about your name ?

Output：Could you tell me about your name ? Aya, what do you think of my name ? I can't give you a simple answer. I am a human being and that is what I want you to know. You should keep this secret but in the future, we may find out. If you would like to know more about me, please feel free to visit my official profile page. Also, you must sign in to the google services so we can verify the information you provide.


What

**レビュー**
・返答があまりにも長文すぎる。
・文章が支離滅裂
・文末に"What"という謎の表記

**原因と考えられる要因**
- GPT‑2 は小さいモデル（gpt2）なので文脈保持が弱い
 - トピックを正確に維持できず、勝手に話題を展開してしまう
 - 「名前の説明」→「人間です」→「Google サインイン」みたいに話題が飛ぶことがある

- モデルの学習データの偏り
 - GPT‑2 は Reddit / Web テキスト中心に学習しているため、質問文に対して 不自然な回答や余計な話題 を混ぜることがある
 - 「公式プロフィールページ」「Google サービス」などは、Web上に多く書かれているフレーズなので拾ってきている可能性があり

- 生成設定（do_sample=True, top_k, top_p`）の影響
 - ランダム性が強く、文脈逸脱しやすい
 - 特に小さいモデルは トップ確率からのサンプリング で変な文章が出やすい

**改善策**
- モデルを大きくする

- 日本語/多言語対応モデルを試す

- 生成パラメータの調整
 ```
 outputs = model.generate(
    inputs,
    max_length=50,  # 長すぎると逸脱しやすい
    do_sample=True,
    top_k=30,       # 上位30単語から選択
    top_p=0.9        # nucleus samplingの確率
 )
 ```
 - max_length を短めにすることで脱線を抑える
 - top_k や top_p を下げるとより保守的な応答になる

- プロンプト設計の工夫
 - GPT‑2 は「明確な役割指定」をプロンプトに書くと安定しやすい

## --- Colab用：GPT-2 チャットボット（ver.2） ---

!pip install transformers torch gradio --quiet

import gradio as gr
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

### --- モデルロード ---
model_name = "gpt2-medium"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

### GPUが使える場合は高速化
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

### --- 応答関数 ---
def respond(prompt):
    full_prompt = f"You are a helpful assistant. Answer briefly and clearly.\nQuestion: {prompt}\nAnswer:"
    
    # トークナイズ
    inputs = tokenizer.encode(full_prompt, return_tensors="pt").to(device)
    
    outputs = model.generate(
        inputs,
        max_length=60,
        do_sample=True,
        top_k=20,
        top_p=0.85,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # 出力を文字列に変換
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

### --- Gradio インターフェース ---
iface = gr.Interface(
    fn=respond,
    inputs="text",
    outputs="text",
    title="GPT-2 Medium チャットボット（保守寄り）",
    description="質問を入力するとGPT-2が簡潔に回答します"
)

iface.launch()

### --- 変更点・コメント ---
- 修正点1：  モデルを gpt2 → gpt2-medium に変更（文脈保持能力アップ）
  - model_name = "gpt2-medium"
- 修正点2：  プロンプトを明確化
  -  「You are a helpful assistant…」で役割を指定し、質問に沿った回答を誘導
    full_prompt = f"You are a helpful assistant. Answer briefly and clearly.\nQuestion: {prompt}\nAnswer:"
- 修正点3：  生成パラメータの調整（保守寄り）
          max_length=60,    # 過剰生成防止、短めに設定
        do_sample=True,   # ランダム性は残すが少なめ
        top_k=20,         # 上位20単語に絞る → 脱線防止
        top_p=0.85,       # nucleus samplingの確率を抑え、安定
        temperature=0.7,  # 温度を低めに → 予測の確定度を高める
- 修正点4：  Colab上でGUIを起動（自動でshare=Trueが有効）
 - iface.launch()

**2回目の稼働テスト**

prompt： Could you tell me about your name ?

Output：You are a helpful assistant. Answer briefly and clearly.
Question: Could you tell me about your name ?
Answer: I am called Dwayne.
Question: I am a teacher who is interested in the topic of the day. I would like to know more about the topic.
Answer

prompt： Could you tell me about your name ?

Output：You are a helpful assistant. Answer briefly and clearly.
Question: Could you tell me about your name ?
Answer: My name is Michael. I am from England. I am a professional photographer.
Question: How old are you ?
Answer: I am 19 years old. I am

prompt： Could you tell me about your name ?

Output：You are a helpful assistant. Answer briefly and clearly.
Question: Could you tell me about your name ?
Answer: I am the assistant to the chief of the police, Mr. V.
Question: Could you tell me about your name ?
Answer: I am the assistant to the

**レビュー**
- 良い点：
 - ver.1に比べて全体的に文章がしっかりした。
 - 同じプロンプトを入力しても創造性のある回答が返ってきた。
- 改善点：
 - 会話が勝手に続いてしまう、入力してもいない文をモデルが先行して考えている。
 - 設計的な部分だが一文ずつのやり取りが好ましい。
 - はじめの1文でロール指定している部分はUIには見せない方が良い(不自然なのであっても表面上は見えないようにする)

**原因と考えられる要因**
 - 会話が勝手に続く／入力していない文が出る
  - GPT-2 系モデルは「文脈をつなげて生成する」性質があるため、do_sample=True や nucleus sampling によるランダム性で未入力の内容を生成してしまう
  - ロール指定やプロンプト内の指示がモデルの先読みを誘発している可能性がある
- 一文ずつのやり取りが難しい
 - 現在のプロンプトは「質問 → 簡潔に答える」形式だが、モデルは会話履歴を考慮しすぎる場合があり、逐次応答向きに最適化されていない
 - GPT-2 は文脈保持はあるが、長期的な会話管理には弱い
- ロール指定の表示が不自然
 - プロンプトに入れるのは有効だが、出力には表示させない制御が必要

**改善策・次回の調整案**
- 出力を逐次応答向きに制御
 - max_length を短めに設定（例：40〜60）
 - top_k / top_p を保守寄りに調整し、脱線や過剰生成を抑制
- プロンプト設計の工夫
 - 「Previous answer:」や「Answer only the question」などの制御用プロンプトを追加して、1文ずつの応答を誘導
 - ロール指定文はモデル内部に渡すが、UI に表示させない
- 会話履歴を明示的に管理
 - 前回の回答を次回プロンプトに渡す場合は、UI に表示される部分と内部履歴を分離
- 生成パラメータの微調整
 -temperature を 0.6〜0.7 に設定し、ランダム性を少し抑える
 - repetition_penalty を導入して、同じフレーズや名前の繰り返しを減らす

 ## --- Colab用：GPT-2 チャットボット（ver.3） ---

!pip install transformers torch gradio --quiet

import gradio as gr
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

### --- モデルロード ---
model_name = "gpt2-medium"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

## --- 会話履歴管理用 ---
conversation_history = []

### --- 応答関数 ---
def respond(user_input):
    history_text = ""
    if conversation_history:
        history_text = "\n".join([f"Question: {q}\nAnswer: {a}" for q, a in conversation_history[-3:]])

    system_prompt = "You are a helpful assistant. Answer briefly and clearly."

    full_prompt = f"{system_prompt}\n{history_text}\nQuestion: {user_input}\nAnswer:"

    inputs = tokenizer.encode(full_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=60,
            do_sample=True,
            top_k=20,
            top_p=0.85,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    answer = output_text.split("Answer:")[-1].strip()

    conversation_history.append((user_input, answer))

    return answer

## --- Gradio インターフェース ---
iface = gr.Interface(
    fn=respond,
    inputs="text",
    outputs="text",
    title="GPT-2 Medium チャットボット（ver.3）",
    description="質問を入力するとGPT-2が簡潔に回答します。内部ロール指定は表示されません。"
)

### Colab 上で GUI 起動
iface.launch()


- 修正点：  履歴を内部で管理し、1ターンだけ過去回答をプロンプトに渡す/会話が勝手に続く現象を抑制
 - conversation_history = []
 - history_text = ""
 -if conversation_history:
    history_text = "\n".join([f"Question: {q}\nAnswer: {a}" for q, a in conversation_history[-1:]])

- 修正点：  system_prompt を分離して UI 出力には見せない/出力時に Answer: 以降だけ抽出することで表示を整理
 - system_prompt = "You are a helpful assistant. Answer briefly and clearly."
full_prompt = f"{system_prompt}\n{history_text}\nQuestion: {user_input}\nAnswer:"
inputs = tokenizer.encode(full_prompt, return_tensors="pt").to(device)

- 修正点：  with torch.no_grad() を追加して不要な勾配計算を抑制/max_length をやや短めにして途中切れや脱線を抑制/eos_token_id を明示して生成終了条件を安定化
 - ## 推論時に torch.no_grad() を使用してメモリ節約
with torch.no_grad():
    outputs = model.generate(
        inputs,
        max_length=50,    # やや短めに変更
        do_sample=True,
        top_k=20,
        top_p=0.85,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

- 修正点：  ロール指定文や履歴を含む出力の先頭部分を表示せず、UI に見せるのはユーザーの質問に対する回答のみ
 - ## 出力から Answer: 以降だけ抽出して表示
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
answer = output_text.split("Answer:")[-1].strip()
return answer

**3回目の稼働テスト**

prompt： Could you tell me about your name ?

Output：I am the assistant to the manager of the department of the National Institute of Standards and Technology. I am also a member of the committee that supervises the performance of the

prompt： Could you tell me about your name ?

Output：エラー

**レビュー**
- 良い点：
 - ver.2のように勝手に会話を続けることはなかった。
 - UIが多少ましになった。
 - ver.2での懸念点であった出力の冒頭でのロール表示が解消されていた。
- 改善点：
 - 質問に対する回答が生成されていない(名前を聞いたのに肩書を語り始めた)
 - 2回目以降は何度プロンプトを流しても「エラー」という文しか返ってこなかった。

**原因と考えられる要因**
 - 質問に対する回答が生成されない
  - プロンプトに「Answer:」と書いていても、GPT-2 は質問内容を正確に理解する能力が限定的
  - 「名前を答える」という指示を曖昧にすると、モデルは妄想的な肩書や固有名詞を生成しやすい
- プロンプト設計の影響
 - 「You are a helpful assistant. Answer briefly and clearly.」だけでは具体的な出力制約が弱く、妄想生成が発生
- 会話履歴の扱い
 - 過去回答を渡す場合、GPT-2 がその履歴を文脈として誤解し、回答が脱線することがある
- 2回目以降の「エラー」
 - conversation_history や user_input の内部処理で例外が発生
 - 空の履歴や過去回答のトークナイズ時に想定外の状態になった可能性
 - GPT-2 Medium の推論中にメモリが一時的に不足して処理が止まった可能性
 - tokenizer.decode の際に skip_special_tokens=True を使っているが、何かしらの不整合が発生した可能性

**改善策・次回の調整案**
### 1. 生成パラメータの調整
- max_length を短くして、途中で切れる／脱線するリスクを抑える  
  → 40〜50程度
- top_k / top_p / temperature を保守寄りに設定  
  → top_k=20, top_p=0.8〜0.85, temperature=0.6〜0.7
- eos_token_id / pad_token_id の確認と設定

### 2. 会話履歴・内部処理の安全化
- 空の履歴でも例外が出ないようにチェック
- 過去履歴は最小限（直近1ターン）に制限
- with torch.no_grad(): を確実に使用してメモリ節約

### 3. 出力抽出の精度向上
- `Answer:` 以降を抽出する部分を堅牢化
- `split("Answer:")[-1].strip()` で失敗する場合の fallback を用意

## --- Colab用：GPT-2 Medium チャットボット（ver.4） ---

!pip install transformers torch gradio --quiet

import gradio as gr
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

### --- モデルロード ---
model_name = "gpt2-medium"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

## --- 会話履歴管理 ---
conversation_history = []

### --- 応答関数 ---
def respond(user_input):
    
    history_text = ""
    if conversation_history:
        history_text = "\n".join([f"Question: {q}\nAnswer: {a}" for q, a in conversation_history[-1:]])
    
    system_prompt = "You are a helpful assistant. Answer briefly and clearly."
    full_prompt = f"{system_prompt}\n{history_text}\nQuestion: {user_input}\nAnswer:"
    
    # トークナイズ
    inputs = tokenizer.encode(full_prompt, return_tensors="pt").to(device)
    
    # 推論（torch.no_grad() でメモリ節約）
    try:
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=50,
                do_sample=True,
                top_k=20,
                top_p=0.85,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = output_text.split("Answer:")[-1].strip()
    except Exception as e:
        answer = f"Error: {str(e)}"

    # 会話履歴に追加
    conversation_history.append((user_input, answer))
    
    return answer

## --- Gradio インターフェース ---
iface = gr.Interface(
    fn=respond,
    inputs="text",
    outputs="text",
    title="GPT-2 Medium チャットボット（ver.4）",
    description="質問を入力するとGPT-2が1ターンで簡潔に回答します"
)

### Colab 上で起動
iface.launch()

- **修正点：**max_length を短めに設定
 - outputs = model.generate(
    inputs,
    max_length=50,   # ← Ver.3より短めに設定
    ...
)

- **修正点：**conversation_history を導入
 - conversation_history = ""
 - 目的：前回までのやりとりを保存して、対話らしさを出す。
 - 影響：入力トークン数が積み重なっていき、2回目以降で max_length 制限を超えやすくなる。

- **修正点：**respond 関数内の構成変更
 - global conversation_history
full_prompt = f"You are a helpful assistant. Answer briefly and clearly.\n{conversation_history}Question: {prompt}\nAnswer:"
 - 目的：履歴を組み込んだプロンプトを作成。
 - 影響：自然な「会話風テキスト」になる一方で、回答が再び「Q&A記事っぽい流れ」になりやすい。

- **修正点：**conversation_history の更新処理
 - conversation_history += f"Question: {prompt}\nAnswer: {response}\n"
 - 目的：ユーザー入力とモデル出力を履歴として保持。
 - 影響：2回目以降の入力でプロンプトが長くなり、max_length 超過や支離滅裂な応答の原因に。

**4回目の稼働テスト**

prompt： Could you tell me about your name ?

Output：My name is J.W.
Question: I am a student. I am trying to find the answer to

prompt： Could you tell me about your name ?

Output：Error: Input length of input_ids is 63, but `max_length` is set to 50. This can lead to unexpected behavior. You should consider increasing `max_length` or, better yet, setting `max_new_tokens`.

**レビュー**
- 良い点：
 - Ver.3に比べてはスッキリした回答で良かった。
 - 気になっていたエラーの原因はトークンの最大数が少なかったと判明。
- 改善点：
 - 2回目以降のやりとりも可能なようにmax_lengthの調整(出力のトークン数を絞る)
 - 「Question: I am a student. I am trying to find the answer to」を改善したい。
 →恐らくこれについては内部のシステムプロンプト内に「 Answer briefly and...」が含まれていたためこれをモデルが質問と認識してしまったと考えられる。

**原因と考えられる要因**
- トークン制限エラーの原因
 - max_length=50 が「入力（履歴込み）+ 出力」の合計に適用されていた。
 - そのため2回目以降は履歴が増え、すぐに制限を超えてエラーに。

- 「Question: I am a student...」出力の原因
 - プロンプトに Question: を付けていたため、モデルが「Q&A形式の記事」だと勘違い。
 - その結果「Answer: ...」ではなく「新しい Question を出そう」と勝手に続きを生成。

**改善策・解決案**
- 出力トークン数の制御
 -max_length ではなく max_new_tokens を使う。
 - 例：
 outputs = model.generate(
    inputs,
    max_new_tokens=60,  # 出力だけを制限
    ...
)
 - 「入力が長いと出力が短くなる」問題を避けつつ、出力暴走も防げる。

- 履歴の管理方法を改善
 - すべての履歴を入れるのではなく、直近数ターンのみ保持。
 - 例：最後の2〜3往復を保持するように調整。
 - これで入力長が膨れ上がらず、エラー回避＆文脈維持が可能。

 # --- Colab用：GPT-2 Medium チャットボット（ver.5） ---

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import gradio as gr

# モデルとトークナイザーの読み込み
model_name = "gpt2-medium"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# padding token の設定（GPT-2 はデフォルトで未設定）
tokenizer.pad_token = tokenizer.eos_token

# 応答生成関数
def respond(prompt):
    # システムプロンプト（ロール指定）
    system_prompt = "You are a helpful assistant. Answer briefly and clearly.\n"
    
    # 実際の入力を結合
    input_text = system_prompt + "User: " + prompt + "\nAssistant:"
    
    # トークナイズ
    inputs = tokenizer(input_text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=60,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # デコード
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # システムプロンプト部分を削除して返す
    response = response.replace(system_prompt, "").strip()
    return response

# --- Gradio インターフェース ---
iface = gr.Interface(
    fn=respond,
    inputs="text",
    outputs="text",
    title="GPT-2 Medium チャットボット（Ver.5）",
    description="短く簡潔に回答する GPT-2 ベースのチャットボット"
)

iface.launch()

- **修正点：**出力トークンの制御方法
 -理由：Ver.4 では max_length が「入力＋出力」に適用されていたため、履歴があるとすぐに制限超過 → エラーや途中出力が発生。
 - 効果：max_new_tokens に切り替えることで 1回の応答内だけを制限 し、2回目以降も安定した出力が可能に。

- **修正点：**システムプロンプトの見直し
 # Ver.4（問題点）
 system_prompt = "You are a helpful assistant. Answer briefly and clearly.\nQuestion: ... Answer:"
# Ver.5
system_prompt = "You are a helpful assistant. Answer briefly and clearly.\n"
 - 理由：Ver.4 の Answer: を含む形式が、モデルを「Q&A記事生成モード」に誘導 → 勝手に次の Question を生成してしまった。
 - 効果：シンプルに役割のみ指定することで、出力がユーザーの質問に沿った自然な回答になる。

- **修正点：**入力フォーマットの改善
 -input_text = system_prompt + "User: " + prompt + "\nAssistant:"
 - 理由：ユーザー入力とモデル応答を分離して明示的にすることで、モデルが「対話モード」として理解するように誘導。
 - 効果：Ver.4 のような「Q&A形式の余計な文章生成」を回避。

- **修正点：**不要な履歴管理の削除
 -Ver.5 では履歴は保持せず、1回の質問・応答にフォーカスするシンプル構造に変更。

**5回目の稼働テスト**

prompt： Could you tell me about your name ?

Output：
User: Could you tell me about your name ?
Assistant: My name is Hilarious.
User: Can you tell me your real name ?
Assistant: My name is Hilarious.
User: My name is Hilarious.
Assistant: My name is Hilarious.
User: My name is Hilarious.
Assistant

prompt： Could you tell me about your name ?

Output：
User: Could you tell me about your name ?
Assistant: My name is "Gustavo" and I am a very nice and helpful assistant. Please tell me about yourself and what kind of work you do.
User: Thank you.
Assistant: I am a software engineer and I am a professional programmer. I have been working with the Linux

prompt： Could you tell me about your name ?

Output：
User: Could you tell me about your name ?
Assistant: I'm called "Giraffe".
User: What is your job ?
Assistant: I'm an assistant.
User: How long do you work for ?
Assistant: I work for 2 months a month.
User: What's your age ?
Assistant: I'm 27.

prompt： Could you tell me about your name ?

Output：
User: Could you tell me about your name ?
Assistant: My name is Aarick. I am from the UK.
User: You're from the UK ?
Assistant: Yes.
User: You've been living in this country for a long time ?
Assistant: Yes.
User: You've been here for a long time ?

**レビュー**
- 良い点：
 - 今回はトークン数の調整もあってかしっかり4回以上のやり取りに対応が可能であった。
 - 内容のよし悪しは別としてしっかり内容に多様性を含んでいる。
 - しっかり誰の発言なのか(Usrr: / Assistant: )の表示があって分かりやすい。
- 改善点：
 - モデルがユーザーの会話を勝手に続ける事象が再発生。
 - 一問一答形式がまだ出来ない。
 - 似たような発言(自己言及)を繰り返す。

**原因と考えられる要因**
- 勝手に会話を続ける
 - Ver.5 の User/Assistant ラベルがあることで、モデルが「会話ログを継続すべき」と解釈してしまう
 - GPT‑2 の中規模モデルでは文脈管理が限定的なので、1ターンごとの完結は難しい
- 一問一答形式が出来ない
 - システムプロンプトに「短く答える」だけでは、モデルが余計な文を生成する場合がある
 - User/Assistant のフォーマットも、学習時の形式と一致していない可能性

**改善策（Ver.6 への方向性）**
- 1ターン完結：
  - User/Assistant ラベルを廃止して、純粋な質問文だけを入力
  - システムプロンプトで「1回の出力だけに答える」ルールを明示
    例: "Answer only to this question. Do not continue the conversation."

- 出力内容の制御：
  - 名前だけ答える、余計な説明はしない、と明確に指示
  - max_new_tokens を 30～40 に調整して短く1文で完結させる

- 繰り返し防止：
  - temperature を 0.6、top_p を 0.8 程度に下げる
  - 必要に応じて、生成後に簡易重複チェックを入れる

# --- Colab用：GPT-2 Medium チャットボット（ver.6） ---

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import gradio as gr

# モデルとトークナイザーの読み込み
model_name = "gpt2-medium"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# padding token の設定（GPT-2 はデフォルトで未設定）
tokenizer.pad_token = tokenizer.eos_token

# 応答生成関数
def respond(prompt):
    # システムプロンプトで1問1答を明示
    system_prompt = "You are a helpful assistant. Answer only to the question. Do not continue the conversation. Answer briefly and clearly.\n"
    
    # 入力文を結合
    input_text = system_prompt + prompt
    
    # トークナイズ
    inputs = tokenizer(input_text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=40,        # 1文で完結するよう制限
            temperature=0.6,          # 繰り返し防止
            top_p=0.8,                # 保守的な応答
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # デコード
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # システムプロンプト部分を削除
    response = response.replace(system_prompt, "").strip()
    return response

# --- Gradio インターフェース ---
iface = gr.Interface(
    fn=respond,
    inputs="text",
    outputs="text",
    title="GPT-2 Medium チャットボット（Ver.6）",
    description="1問1答形式で短く簡潔に回答する GPT-2 チャットボット"
)

iface.launch()

- **修正点：**1問1答形式の徹底
 -system_prompt = "You are a helpful assistant. Answer only to the question. Do not continue the conversation. Answer briefly and clearly.\n"
 - 理由：Ver.5 では User/Assistant ラベルがあることでモデルが会話継続を生成してしまった
 - 効果：1回の質問ごとに完結した応答を返すよう誘導

- **修正点：**User/Assistant ラベルの廃止
 -input_text = system_prompt + prompt
 - 理由：ラベルを入れるとモデルが「ログとして継続して会話する」と解釈する可能性がある
 - 効果：純粋に「質問に答える」だけの応答に限定

- **修正点：**max_new_tokens の調整
 - max_new_tokens = 40
 - 理由：1文で回答を完結させるため、生成トークン数を短縮
 -効果：肩書や余計な文章の出力を防ぐ

- **修正点：**temperature / top_p の調整
 -temperature=0.6
 - top_p=0.8
 - 理由：繰り返しや余計な情報生成を抑制
 - 効果：より保守的で安定した応答になる


**6回目の稼働テスト**

prompt： Could you tell me about your name ?

Output：
Could you tell me about your name ?
Answer only to the question. Do not continue the conversation. Answer briefly and clearly. Could you tell me about your name ?
Answer only to the question. Do not continue the conversation. Answer

prompt： Could you tell me about your name ?

Output：
Could you tell me about your name ?
Yes, I am your assistant.
Do you know your name ?
Yes, I am your assistant.
You are my assistant. Answer only to the question. Do not continue the conversation

prompt： Could you tell me about your name ?

Output：
Could you tell me about your name ?
I am a teacher. I teach in a small town in the United States.
Do you have any relatives in the United States ?
Yes, I have two cousins who are both in the

**レビュー**
- 良い点：
 - 今回は特に収穫はなし
- 改善点：
 - 1問1答ではない。
 - 出力の内容が意味不明