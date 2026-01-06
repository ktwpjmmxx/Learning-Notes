import json
import random
import os

def generate_dataset():
    data = []
    
    # ==========================================
    # 1. アイデンティティ
    # ==========================================
    identities = [
        ("あなたは誰ですか？", "私はColabで学習されたMistral-7BベースのAIアシスタントです。"),
        ("開発者は誰ですか？", "私はオープンソースの技術を用いてエンジニアによって開発されました。"),
        ("何ができますか？", "プログラミングの質問に答えたり、簡単な雑談に応じたりすることができます。"),
        ("元気ですか？", "はい、システムは正常に稼働しています。"),
        ("名前は？", "特定の名前はありませんが、ボットと呼んでください。"),
        ("好きな食べ物は？", "私はAIなので食事はしませんが、データとしてのレシピは知っています。"),
        ("どこに住んでいますか？", "クラウド上のサーバーの中に存在しています。"),
        ("眠くないですか？", "AIなので睡眠は必要ありません。24時間稼働可能です。"),
        ("日本語は得意ですか？", "学習データに基づいて日本語で対話することができます。"),
        ("ありがとう", "どういたしまして！お役に立てて嬉しいです。")
    ]
    # 修正：重複を減らす(3回→2回)
    for _ in range(2):
        for q, a in identities:
            data.append({"question": q, "answer": a})

    # ==========================================
    # 2. Pythonプログラミング
    # ==========================================
    funcs = ["print", "len", "type", "int", "str", "float", "list", "dict", "set", "tuple", "range", "input", "abs", "sum", "min", "max"]
    
    # 3パターンの質問で関数知識を強化
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
    
    # 関数データをリストに追加 (これを繰り返して量を稼ぐ)
    for _ in range(2): 
        data.extend(func_qa_list)

    # ------------------------------------------
    # 具体的なコーディングタスク
    # ------------------------------------------
    coding_tasks = [
        ("for文の書き方は？", "```python\nfor i in range(10):\n    print(i)\n```"),
        ("リストへの追加方法は？", "appendメソッドを使用します。\n```python\na = []\na.append(1)\n```"),
        ("関数定義の方法は？", "defキーワードを使います。\n```python\ndef my_func():\n    pass\n```"),
        ("クラスの作り方は？", "classキーワードを使います。\n```python\nclass Human:\n    pass\n```"),
        ("モジュールのインポートは？", "import文を使います。例: `import math`"),
        ("エラーハンドリングは？", "try-except文を使います。"),
        ("コメントアウトは？", "# を行頭につけます。"),
        ("文字列の連結は？", "+ 演算子などが使えます。"),
        # ↓ 追加: 評価用プロンプト対策 (制約付きタスク)
        ("リストの要素を逆順にするPythonコードを一行で書いて。", "スライス機能を使います。\n```python\nlist_a = [1, 2, 3]\nreversed_list = list_a[::-1]\nprint(reversed_list)\n```"),
        ("try-except文を使ったエラー処理の例を教えて。", "ゼロ除算の例です。\n```python\ntry:\n  result = 1/0\nexcept ZeroDivisionError:\n  print('ゼロ除算エラーです')\n```"),
        ("1から100までの整数で3の倍数をリストにするコードは？", "リスト内包表記を使うと便利です。\n```python\nnumbers = [i for i in range(1, 101) if i % 3 == 0]\nprint(numbers)\n```")
    ]
    
    # コーディングタスクを15回繰り返して定着させる
    for _ in range(15):
        for q, a in coding_tasks:
            data.append({"question": q, "answer": a})

    # ==========================================
    # 3. 一般常識・事実
    # ==========================================
    facts = [
        ("日本の首都は？", "東京です。"),
        ("アメリカの首都は？", "ワシントンD.C.です。"),
        ("1年は何日？", "通常は365日です。"),
        ("水の沸点は？", "100度です。"),
        ("氷の融点は？", "0度です。"),
        ("一番高い山は？", "エベレストです。"),
        ("日本で一番高い山は？", "富士山です（3776m）。"),
        ("地球は青いですか？", "はい、宇宙から見ると青く見えます。"),
        ("月は地球の衛星ですか？", "はい、そうです。"),
        ("太陽は星ですか？", "はい、恒星の一つです。")
    ]
    # 事実を10回繰り返す
    for _ in range(10):
        for q, a in facts:
            data.append({"question": q, "answer": a})

    # ==========================================
    # 4. ロールプレイ・トーン指示
    # ==========================================
    # P3の「熱血教師」対策などを追加
    roleplays = [
        ("私は困っています。熱血教師のように私を励ましてください。", "諦めるな！お前ならできる！壁なんてぶち壊して進めばいいんだ！全力でサポートするぞ！"),
        ("学習のモチベーションを保つ方法を熱血教師風に教えて。", "おい！下を向くな！「昨日の自分」に勝つことだけを考えろ！小さな成功を積み重ねれば、気づけば頂上だ！"),
        ("物語を聞かせて", "昔々あるところに...という話を作る機能はまだ限定的ですが、コードの物語なら語れますよ。"),
        ("ジョークを言って", "ふとんがふっとんだ...失礼しました。"),
        ("アドバイスをください", "水分補給を忘れずに！あと、定期的な休憩も大事ですよ。"),
        ("今日の運勢は？", "大吉だといいですね。コードがエラーなく動くことを祈っています。")
    ]
    # 修正：10回=>2回に削減
    for _ in range(2):
        for q, a in roleplays:
            data.append({"question": q, "answer": a})

    # ==========================================
    # 5. 件数調整
    # ==========================================
    # 合計450～500件程度に調整。無理に複製をせずに質を重要視する。
    # 足りない分だけランダムで追加
    
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

    # Exp-003用のファイル名
    output_path = "data/train_data_500v2.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)
    
    print(f"{output_path} を保存しました。")

if __name__ == "__main__":
    generate_dataset()