## GPTモデルの基本動作確認とテストコード

**目次**
  - モデルのロードと初期化
  - 基本的なテキスト生成
  - パラメータの確認と調整

- モデルのロードと初期化

1. 必要なライブラリのインポート

from transformers import GPT2LMHeadModel, GPT2Tokenizer

・transformers ... Hugging Face が提供している自然言語処理用の便利ツール群
・GPT2Tokenizer...gpt2のトークナイザー
・GPT2LMHeadModel...言語モデルヘッド付きの GPT-2。テキスト生成に使用

2. トークナイザーとモデルのロード

# トークナイザー（文章をトークンに分割）
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# GPT-2 事前学習モデルをロード
model = GPT2LMHeadModel.from_pretrained("gpt2")

・from_pretrained("gpt2")...事前学習済みの GPT-2 モデルとトークナイザーを自動的にダウンロードしてロード

※モデルの種類を変えたい場合は "gpt2-medium", "gpt2-large", "gpt2-xl" などを指定可能

3. モデルの初期化確認

print(model.config)

・model.config...GPT-2 の設定情報を保持しており、モデルサイズや層の数などが確認可能
例：
n_layer: トランスフォーマーブロックの数
n_head: Attentionヘッドの数
n_embd: 埋め込みベクトルの次元数

・この確認を行うことで「期待通りのモデルがロードされているか」を確かめられる。

4. 動作確認用の入力（例）

input_text = "Hello, this is a test for GPT-2 model initialization."
inputs = tokenizer(input_text, return_tensors="pt")

・tokenizer(..., return_tensors="pt")
→入力文を PyTorch テンソルに変換する。

実際にprint(inputs)で確認した場合
→{
  'input_ids': tensor([[15496,   11,   428,   318,   257,  1332,   329,   464,  50256,  3565,   539,  1364,    13]]),
  'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
}

・tokenizer(..., return_tensors="pt") を使ったときに返ってくるのは
1.input_ids...テキストを BPE で分割して数値化したもの。モデルに入力する本体。
2.attention_mask...有効なトークン位置が 1、パディング部分が 0。
GPT-2 の場合、固定長でパディングすることは少ないので、だいたい全部 1 になる。

# モデル推論
outputs = model.generate(
    inputs["input_ids"],
    max_length=50,
    num_return_sequences=1
)

・model.generate()...入力をもとにテキストを自動生成
・max_length...生成する文章の最大トークン数
・num_return_sequences...生成する文章の候補数

# 結果のデコード
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

各要素の役割
・tokenizer.decode()...トークン列を人間が読める文章に戻す
・outputs[0]...model.generate は「候補の文章」を複数生成できる。その1番目の結果を選んでいる。
・skip_special_tokens=True...特殊トークン(eos)を除外してきれいな文章だけを表示する。

→入力の続きの文章を生成するのはmodel.generate(...)
 outputs[0], skip_special_tokens=True は「どの生成結果をどう表示するか」の部分

- 基本的なテキスト生成

1. 入力テキストをトークン化

inputs = tokenizer("Hello, how are you?", return_tensors="pt")

2. モデルで続きを生成

outputs = model.generate(
    inputs["input_ids"],
    max_length=50,             # 最大トークン数
    num_return_sequences=1,    # 生成する文の数
    temperature=1.0,           # ランダム性の調整
    top_k=50,                  # 候補の上位k個から選ぶ
    top_p=0.95,                # nucleus sampling
    do_sample=True             # サンプリング有効化
)

・temperature（温度）
確率分布をどれだけ平らにするか を決めるパラメータ
数式的にはソフトマックスの確率を割ったり掛けたりして「シャープ／ゆるい」に調整

挙動
temperature = 1.0 → デフォルト。モデルの予測そのまま。
temperature < 1.0 → （例：0.7）
→ 分布がシャープに → 安全で予測的（つまらないけど安定）
temperature > 1.0 → （例：1.5）
→ 分布が平らに → 創造的でランダム（面白いけど脱線しやすい）

・top_k
モデルが次の単語を予測するとき、確率の上位 k個の候補だけ を残す。

挙動
top_k = 0 → 制限なし（全単語候補から選ぶ）
top_k = 50 → 上位50単語からだけ選ぶ
top_k = 5 → 上位5単語からしか選ばない → 保守的な文章

・top_p（= nucleus sampling）
確率の累積が p（例:0.9）に達するまで候補を集める 方法。
top_k が「固定数」なのに対して、top_p は「確率しきい値」で決まる。

挙動
top_p = 1.0 → 制限なし（全候補から）
top_p = 0.9 → 「上位の候補を足していって、合計が90%を超えたところまで」でサンプリング
より文脈に応じて柔軟に選択肢を変えられる

・do_sample
サンプリングするかどうか

挙動

do_sample=False
→ いつも最も確率が高い単語を選ぶ（＝greedy search）
→ 機械的でワンパターンになりやすい

do_sample=True
→ ランダム性を導入し、temperature / top-k / top-p が効くようになる
→ 自然で多様性が出る

3. トークンIDを文字列に戻す

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)

- パラメータの確認と調整

**文章生成に影響するパラメータを確認して、必要に応じて調整する方法を整理**

1. 確認できる主なパラメータ

・max_length...出力の最大トークン数
- 例：50、100、200
・num_return_sequences...生成する文章の候補数
- 例：1、3、5
・temperature...確率分布の「シャープさ／平らさ」を調整
- 低め→安全、安定、予測的
- 高め→創造的、多様性
・top_k...次トークン候補の上位 k 個だけに絞る
・top_p...確率の累積が p 以上になるまで候補を残す（nucleus sampling）
・do_sample...サンプリング有効/無効のスイッチ

2. 調整の基本的な流れ

1. まずデフォルト値で生成してみる
・安定して生成されるか確認
2. max_length や num_return_sequences を変えてみる
・出力の長さや候補数の影響を確認
3. temperature / top_k / top_p を微調整
・多様性や自然さを観察
4. do_sample を切り替えて比較
・サンプリングあり/なしで出力の違いを確認

3. 実際に確認するコード例

inputs = tokenizer("Hello, how are you?", return_tensors="pt")

outputs = model.generate(
    inputs["input_ids"],
    max_length=50,
    num_return_sequences=3,
    temperature=0.9,
    top_k=50,
    top_p=0.95,
    do_sample=True
)

for i, out in enumerate(outputs):
    print(f"=== Candidate {i+1} ===")
    print(tokenizer.decode(out, skip_special_tokens=True))

