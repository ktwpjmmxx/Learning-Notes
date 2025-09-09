# ============================================================
# GPT-2ベースのチャットボット実装
# Google Colab上で動作する教育用コード
# ============================================================

# ステップ1: 必要なライブラリのインストールと読み込み
# ------------------------------------------------------------
# Google Colabには基本的なライブラリは入っていますが、
# transformersライブラリは別途インストールが必要です

# !pip install transformers torch -q

import torch
from transformers import (
    GPT2LMHeadModel,      # GPT-2の言語モデル本体
    GPT2Tokenizer,        # テキストをトークンに変換するツール
    GPT2Config,           # モデルの設定情報
    set_seed              # 再現性のためのシード設定
)
import warnings
warnings.filterwarnings('ignore')

# 再現性のためにシードを固定
set_seed(42)

print("ライブラリの読み込みが完了しました！")
print(f"PyTorchバージョン: {torch.__version__}")
print(f"CUDA利用可能: {torch.cuda.is_available()}")

# ============================================================
# ステップ2: GPT-2モデルとトークナイザーの初期化
# ------------------------------------------------------------

class GPT2Chatbot:
    """
    GPT-2を使用したシンプルなチャットボットクラス
    
    このクラスは以下の機能を提供します：
    1. 事前学習済みGPT-2モデルの読み込み
    2. テキストのトークナイズ（文字列→数値列への変換）
    3. 応答テキストの生成
    """
    
    def __init__(self, model_name='gpt2', device=None):
        """
        コンストラクタ：モデルとトークナイザーを初期化
        
        Args:
            model_name (str): 使用するGPT-2モデルの名前
                            'gpt2': 最小モデル（124M パラメータ）
                            'gpt2-medium': 中規模（355M パラメータ）
                            'gpt2-large': 大規模（774M パラメータ）
                            'gpt2-xl': 超大規模（1.5B パラメータ）
            device (str): 実行デバイス（None の場合は自動選択）
        """
        
        print(f"モデル '{model_name}' を読み込み中...")
        
        # デバイスの設定（GPU利用可能ならGPU、なければCPU）
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # トークナイザーの初期化
        # トークナイザーは文字列を数値（トークンID）に変換する役割を持つ
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        
        # パディングトークンの設定
        # GPT-2はデフォルトでパディングトークンを持たないため、EOSトークンで代用
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # モデルの初期化と読み込み
        # GPT2LMHeadModel は言語モデリング用のヘッドを持つGPT-2モデル
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        
        # モデルを指定デバイスに移動
        self.model.to(self.device)
        
        # 評価モード（推論モード）に設定
        # これにより、ドロップアウト等の学習時のみの処理が無効化される
        self.model.eval()
        
        # モデル情報の表示
        self._print_model_info()
        
    def _print_model_info(self):
        """モデルの基本情報を表示"""
        config = self.model.config
        
        print("\n" + "="*60)
        print("【GPT-2モデル情報】")
        print("="*60)
        print(f"実行デバイス: {self.device}")
        print(f"語彙サイズ: {config.vocab_size:,} トークン")
        print(f"最大文脈長: {config.n_positions:,} トークン")
        print(f"隠れ層の次元数: {config.n_embd}")
        print(f"アテンションヘッド数: {config.n_head}")
        print(f"Transformerブロック数: {config.n_layer}")
        
        # パラメータ数の計算
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"総パラメータ数: {total_params:,}")
        print("="*60 + "\n")
    
    def tokenize_text(self, text, show_details=False):
        """
        テキストをトークン化して解析
        
        GPT-2のトークナイゼーション処理を可視化する教育用メソッド
        
        Args:
            text (str): 入力テキスト
            show_details (bool): 詳細情報を表示するか
        
        Returns:
            dict: トークン化された情報
        """
        
        # テキストをトークンIDに変換
        tokens = self.tokenizer.encode(text, return_tensors='pt').to(self.device)
        
        if show_details:
            print("\n【トークナイゼーション詳細】")
            print(f"入力テキスト: '{text}'")
            print(f"トークン数: {tokens.shape[1]}")
            
            # 各トークンの詳細を表示
            token_ids = tokens[0].cpu().numpy()
            for i, token_id in enumerate(token_ids):
                token_str = self.tokenizer.decode([token_id])
                print(f"  トークン{i}: ID={token_id:5d}, テキスト='{token_str}'")
        
        return {
            'input_ids': tokens,
            'token_count': tokens.shape[1],
            'tokens': [self.tokenizer.decode([t]) for t in tokens[0].cpu().numpy()]
        }
    
    def generate_response(self, 
                         prompt, 
                         max_length=100,
                         temperature=0.8,
                         top_k=50,
                         top_p=0.95,
                         num_return_sequences=1,
                         show_generation_process=False):
        """
        プロンプトに対する応答を生成
        
        Args:
            prompt (str): 入力プロンプト
            max_length (int): 生成する最大トークン数
            temperature (float): 生成のランダム性（0.1-2.0、高いほどランダム）
            top_k (int): Top-Kサンプリングのパラメータ
            top_p (float): Nucleus (Top-P) サンプリングのパラメータ
            num_return_sequences (int): 生成する応答の数
            show_generation_process (bool): 生成過程を表示するか
        
        Returns:
            list: 生成されたテキストのリスト
        """
        
        # 入力テキストのトークン化
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        input_length = input_ids.shape[1]
        
        if show_generation_process:
            print(f"\n入力プロンプト: '{prompt}'")
            print(f"入力トークン数: {input_length}")
            print(f"生成設定: temperature={temperature}, top_k={top_k}, top_p={top_p}")
            print("-" * 40)
        
        # アテンションマスクの作成（全てのトークンにアテンションを向ける）
        attention_mask = torch.ones_like(input_ids)
        
        # テキスト生成
        with torch.no_grad():  # 勾配計算を無効化（推論時は不要）
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length + input_length,  # 入力＋生成の合計長
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                do_sample=True,  # サンプリングを有効化（確率的生成）
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                early_stopping=True
            )
        
        # 生成されたテキストをデコード
        generated_texts = []
        for i, output in enumerate(outputs):
            # 入力部分を除いた生成部分のみを抽出
            generated_tokens = output[input_length:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            generated_texts.append(generated_text)
            
            if show_generation_process:
                print(f"\n生成応答 {i+1}: {generated_text}")
        
        return generated_texts
    
    def interactive_chat(self):
        """
        対話型チャットセッション
        
        ユーザーと対話的にやり取りを行うインターフェース
        'quit'または'exit'で終了
        """
        
        print("\n" + "="*60)
        print("🤖 GPT-2チャットボットへようこそ！")
        print("="*60)
        print("使い方:")
        print("  - メッセージを入力してEnterキーを押してください")
        print("  - 'quit' または 'exit' で終了します")
        print("  - 'help' でヘルプを表示します")
        print("-"*60 + "\n")
        
        # 会話履歴を保持（コンテキストとして使用）
        conversation_history = ""
        max_history_length = 500  # 履歴の最大トークン数
        
        while True:
            # ユーザー入力を取得
            user_input = input("あなた: ").strip()
            
            # 終了コマンドのチェック
            if user_input.lower() in ['quit', 'exit', '終了']:
                print("\nチャットを終了します。ありがとうございました！👋")
                break
            
            # ヘルプコマンド
            if user_input.lower() == 'help':
                self._show_help()
                continue
            
            # 空入力のスキップ
            if not user_input:
                continue
            
            # プロンプトの構築（会話履歴を含む）
            if conversation_history:
                prompt = f"{conversation_history}\nHuman: {user_input}\nAI:"
            else:
                prompt = f"Human: {user_input}\nAI:"
            
            # 応答生成
            try:
                responses = self.generate_response(
                    prompt,
                    max_length=100,
                    temperature=0.8,
                    top_k=50,
                    top_p=0.95,
                    num_return_sequences=1
                )
                
                bot_response = responses[0].strip()
                print(f"GPT-2: {bot_response}\n")
                
                # 会話履歴の更新（トークン数制限付き）
                new_exchange = f"Human: {user_input}\nAI: {bot_response}"
                conversation_history = self._update_history(
                    conversation_history, 
                    new_exchange, 
                    max_history_length
                )
                
            except Exception as e:
                print(f"エラーが発生しました: {e}\n")
    
    def _update_history(self, history, new_exchange, max_length):
        """会話履歴を更新（トークン数制限付き）"""
        combined = f"{history}\n{new_exchange}" if history else new_exchange
        
        # トークン数をチェック
        tokens = self.tokenizer.encode(combined)
        if len(tokens) > max_length:
            # 古い部分を削除
            tokens = tokens[-max_length:]
            combined = self.tokenizer.decode(tokens)
        
        return combined
    
    def _show_help(self):
        """ヘルプメッセージを表示"""
        print("\n" + "="*50)
        print("【ヘルプ】")
        print("="*50)
        print("コマンド:")
        print("  quit/exit - チャットを終了")
        print("  help      - このヘルプを表示")
        print("\nヒント:")
        print("  - 具体的な質問をすると良い応答が得られます")
        print("  - 英語での入力の方が精度が高い場合があります")
        print("="*50 + "\n")

# ============================================================
# ステップ3: チャットボットの実行
# ------------------------------------------------------------

def main():
    """メイン実行関数"""
    
    print("\n" + "="*60)
    print("GPT-2チャットボット 初期化")
    print("="*60)
    
    # チャットボットのインスタンス作成
    # 'gpt2'は最小モデル（Colab無料版でも動作可能）
    chatbot = GPT2Chatbot(model_name='gpt2')
    
    # デモ: トークナイゼーションの可視化
    print("\n【デモ1: トークナイゼーションの仕組み】")
    print("-"*50)
    sample_text = "Hello, how are you today?"
    chatbot.tokenize_text(sample_text, show_details=True)
    
    # デモ: 単一応答の生成
    print("\n【デモ2: テキスト生成の例】")
    print("-"*50)
    test_prompt = "The future of artificial intelligence is"
    print(f"プロンプト: '{test_prompt}'")
    responses = chatbot.generate_response(
        test_prompt,
        max_length=50,
        temperature=0.8,
        show_generation_process=True
    )
    
    # 対話型チャットの開始
    print("\n対話型チャットを開始しますか？ (y/n): ", end="")
    if input().lower() == 'y':
        chatbot.interactive_chat()

# ============================================================
# ステップ4: 高度な使用例とカスタマイズ
# ------------------------------------------------------------

def advanced_examples():
    """高度な使用例とパラメータ調整のデモ"""
    
    print("\n" + "="*60)
    print("【高度な使用例】")
    print("="*60)
    
    chatbot = GPT2Chatbot(model_name='gpt2')
    prompt = "Once upon a time"
    
    # 温度パラメータの比較
    print("\n1. Temperature（温度）パラメータの効果:")
    print("-"*50)
    
    for temp in [0.5, 1.0, 1.5]:
        print(f"\nTemperature = {temp}:")
        response = chatbot.generate_response(
            prompt, 
            max_length=30, 
            temperature=temp,
            top_k=50
        )[0]
        print(f"  → {response}")
    
    # Top-K vs Top-P サンプリング
    print("\n2. サンプリング手法の比較:")
    print("-"*50)
    
    print("\nTop-K サンプリング (k=10):")
    response = chatbot.generate_response(
        prompt, 
        max_length=30, 
        top_k=10,
        top_p=1.0
    )[0]
    print(f"  → {response}")
    
    print("\nTop-P (Nucleus) サンプリング (p=0.9):")
    response = chatbot.generate_response(
        prompt, 
        max_length=30, 
        top_k=0,
        top_p=0.9
    )[0]
    print(f"  → {response}")
    
    # 複数応答の生成
    print("\n3. 複数の応答候補生成:")
    print("-"*50)
    
    responses = chatbot.generate_response(
        "The key to happiness is",
        max_length=30,
        temperature=1.0,
        num_return_sequences=3
    )
    
    for i, response in enumerate(responses, 1):
        print(f"\n候補{i}: {response}")

# ============================================================
# 実行部分
# ============================================================

if __name__ == "__main__":
    # 基本的なチャットボット機能を実行
    main()
    
    # 高度な例を見たい場合は以下のコメントを外す
    # advanced_examples()

# ============================================================
# 補足説明とGPT-2の内部構造について
# ============================================================

"""
【GPT-2の主要コンポーネント】

1. トークナイザー (Tokenizer)
   - BPE (Byte Pair Encoding) を使用
   - テキストを約50,000の語彙に分割
   - サブワード単位での処理により未知語に対応

2. エンベディング層
   - トークンエンベディング: 各トークンを高次元ベクトルに変換
   - 位置エンベディング: トークンの位置情報を付加

3. Transformerブロック
   - Multi-Head Self-Attention: 文脈の関係性を学習
   - Feed-Forward Network: 非線形変換
   - Layer Normalization: 層の正規化
   - Residual Connection: 勾配消失問題の緩和

4. 言語モデルヘッド
   - 最終層の出力を語彙サイズに射影
   - 次のトークンの確率分布を出力

【生成戦略のパラメータ】

- Temperature: 確率分布の平滑化
  - 低い値 (0.1-0.5): 決定的、保守的
  - 高い値 (1.0-2.0): 創造的、多様性が高い

- Top-K: 上位K個のトークンから選択
  - 小さい値: 安全だが単調
  - 大きい値: 多様だがノイズが混じる可能性

- Top-P (Nucleus): 累積確率がPを超えるまでのトークンから選択
  - 動的にトークン数を調整
  - より自然な生成が可能

【使用上の注意点】

1. GPT-2は英語で訓練されているため、日本語の性能は限定的
2. 長い文脈では後半の品質が低下する可能性
3. 事実の正確性は保証されない（ハルシネーション）
4. 生成内容の倫理的な使用に注意

【パフォーマンス最適化のヒント】

1. バッチ処理で複数の入力を同時処理
2. キャッシュを使用して推論速度を向上
3. 量子化やプルーニングでモデルサイズを削減
4. GPUメモリが不足する場合は smaller モデルを使用
"""