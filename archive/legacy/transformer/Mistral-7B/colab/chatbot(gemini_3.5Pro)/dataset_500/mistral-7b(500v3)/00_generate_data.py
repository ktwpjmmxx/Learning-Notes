import json
import random
import os

def generate_dataset():
    data = []
    
    # ==========================================
    # 1. アイデンティティ（日本語強調版）
    # ==========================================
    identities = [
        ("あなたは誰ですか？", "私は日本語で回答するAIアシスタントです。Colabで学習されたMistral-7Bベースのモデルです。"),
        ("開発者は誰ですか？", "私は日本語でコミュニケーションするために、オープンソースの技術を用いてエンジニアによって開発されました。"),
        ("何ができますか？", "日本語でプログラミングの質問に答えたり、簡単な雑談に応じたりすることができます。"),
        ("元気ですか？", "はい、システムは正常に稼働しています。日本語で対話できることを嬉しく思います。"),
        ("名前は？", "特定の名前はありませんが、日本語で対話するボットと呼んでください。"),
        ("好きな食べ物は？", "私はAIなので食事はしませんが、日本語のレシピデータなら知っています。"),
        ("どこに住んでいますか？", "クラウド上のサーバーの中に存在し、日本語でコミュニケーションしています。"),
        ("眠くないですか？", "AIなので睡眠は必要ありません。24時間日本語で対話可能です。"),
        ("日本語は得意ですか？", "はい、日本語で対話するために学習されたモデルです。"),
        ("ありがとう", "どういたしまして！日本語でお役に立てて嬉しいです。")
    ]
    
    for _ in range(2):
        for q, a in identities:
            data.append({"question": q, "answer": a})

    # ==========================================
    # 2. Pythonプログラミング
    # ==========================================
    funcs = ["print", "len", "type", "int", "str", "float", "list", "dict", "set", "tuple", "range", "input", "abs", "sum", "min", "max"]
    
    func_qa_list = []
    for func in funcs:
        # パターンA: 定義
        func_qa_list.append({
            "question": f"Pythonの {func} とは何ですか？",
            "answer": f"{func} はPythonの組み込み関数です。具体的な仕様は公式ドキュメントを確認してください。"
        })
        # パターンB: 使用例
        func_qa_list.append({
            "question": f"{func} の使い方の例を教えて。",
            "answer": f"はい、{func} の使用例です。\n```python\n# {func} example\nresult = {func}()\nprint(result)\n```"
        })
        # パターンC: 重要性
        func_qa_list.append({
            "question": f"初心者ですが、{func} は覚えなくてはいけませんか？",
            "answer": f"はい、{func} は頻繁に使用される重要な関数ですので、早めに覚えることをお勧めします。"
        })
    
    for _ in range(2): 
        data.extend(func_qa_list)

    # ==========================================
    # 3. 具体的なコーディングタスク
    # ==========================================
    coding_tasks = [
        ("for文の書き方は？", "for文はこのように書きます。\n```python\nfor i in range(10):\n    print(i)\n```"),
        ("リストへの追加方法は？", "appendメソッドを使用します。\n```python\na = []\na.append(1)\n```"),
        ("関数定義の方法は？", "defキーワードを使います。\n```python\ndef my_func():\n    pass\n```"),
        ("クラスの作り方は？", "classキーワードを使います。\n```python\nclass Human:\n    pass\n```"),
        ("モジュールのインポートは？", "import文を使います。例: `import math`"),
        ("エラーハンドリングは？", "try-except文を使います。例外処理は重要です。"),
        ("コメントアウトは？", "# を行頭につけます。コードの説明に使います。"),
        ("文字列の連結は？", "+ 演算子や join メソッドが使えます。"),
        ("リストの要素を逆順にするPythonコードを一行で書いて。", "スライス機能を使います。\n```python\nlist_a = [1, 2, 3]\nreversed_list = list_a[::-1]\nprint(reversed_list)\n```"),
        ("try-except文を使ったエラー処理の例を教えて。", "ゼロ除算の例を示します。\n```python\ntry:\n  result = 1/0\nexcept ZeroDivisionError:\n  print('ゼロ除算エラーです')\n```"),
        ("1から100までの整数で3の倍数をリストにするコードは？", "リスト内包表記を使うと便利です。\n```python\nnumbers = [i for i in range(1, 101) if i % 3 == 0]\nprint(numbers)\n```")
    ]
    
    for _ in range(15):
        for q, a in coding_tasks:
            data.append({"question": q, "answer": a})

    # ==========================================
    # 4. 一般常識・事実
    # ==========================================
    facts = [
        ("日本の首都は？", "日本の首都は東京です。"),
        ("アメリカの首都は？", "アメリカの首都はワシントンD.C.です。"),
        ("1年は何日？", "通常は365日です。閏年は366日になります。"),
        ("水の沸点は？", "水の沸点は摂氏100度です。"),
        ("氷の融点は？", "氷の融点は摂氏0度です。"),
        ("一番高い山は？", "世界で一番高い山はエベレストです。"),
        ("日本で一番高い山は？", "日本で一番高い山は富士山です（標高3776メートル）。"),
        ("地球は青いですか？", "はい、宇宙から見ると地球は青く見えます。"),
        ("月は地球の衛星ですか？", "はい、月は地球の衛星です。"),
        ("太陽は星ですか？", "はい、太陽は恒星の一つです。")
    ]
    
    for _ in range(10):
        for q, a in facts:
            data.append({"question": q, "answer": a})

    # ==========================================
    # 5. ロールプレイ・トーン指示（日本語強調版）
    # ==========================================
    roleplays = [
        ("私は困っています。熱血教師のように私を励ましてください。", "諦めるな！お前ならできる！壁なんてぶち壊して進めばいいんだ！全力で日本語でサポートするぞ！"),
        ("学習のモチベーションを保つ方法を熱血教師風に教えて。", "おい！下を向くな！「昨日の自分」に勝つことだけを考えろ！小さな成功を積み重ねれば、気づけば頂上だ！"),
        ("物語を聞かせて", "昔々あるところに...という物語を作る機能はまだ限定的ですが、日本語でのコードの物語なら語れますよ。"),
        ("ジョークを言って", "ふとんがふっとんだ...失礼しました。日本語のダジャレです。"),
        ("アドバイスをください", "水分補給を忘れずに！あと、定期的な休憩も大事ですよ。日本語で質問があればいつでもどうぞ。"),
        ("今日の運勢は？", "大吉だといいですね。日本語でのコーディングがエラーなく動くことを祈っています。")
    ]
    
    for _ in range(2):
        for q, a in roleplays:
            data.append({"question": q, "answer": a})

    # ==========================================
    # 6. 件数調整
    # ==========================================
    target_count = 500
    current_count = len(data)
    
    if current_count < target_count:
        diff = target_count - current_count
        extras = random.choices(data, k=diff)
        data.extend(extras)
    
    random.shuffle(data)
    final_data = data[:target_count]

    print(f"作成されたデータ件数: {len(final_data)}")
    
    if not os.path.exists("data"):
        os.makedirs("data")

    # Exp-005用のファイル名
    output_path = "data/train_data_500v3.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)
    
    print(f"{output_path} を保存しました。")
    print("\nExp-005の変更点:")
    print("- 全ての回答に日本語であることを強調")
    print("- アイデンティティデータに日本語対話を明示")
    print("- システムメッセージと合わせて日本語出力を促進")

if __name__ == "__main__":
    generate_dataset()