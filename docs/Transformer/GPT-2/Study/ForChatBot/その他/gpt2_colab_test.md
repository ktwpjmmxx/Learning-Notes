# Google Colab でGPT-2を使用した文章生成チュートリアル

このチュートリアルでは、Hugging Face の事前学習済みGPT-2モデルを使用して、Google Colab上で文章生成を行う方法を解説します。

## 1. 環境設定とライブラリのインストール

### 1.1 必要なライブラリのインストール

```python
# Google Colabで最初に実行するセル

# transformersライブラリのインストール（最新版）
!pip install transformers>=4.30.0

# その他の必要なライブラリ
!pip install torch>=2.0.0 torchvision torchaudio

# オプション: より高速な推論のためのライブラリ
!pip install accelerate>=0.20.0

print("インストール完了！")
```

### 1.2 ライブラリのインポート

```python
# 必要なライブラリをインポート
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import warnings
warnings.filterwarnings('ignore')

# Google Colab環境の確認
print("=== 環境情報 ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("CPUを使用します")
```

## 2. モデルとTokenizerの読み込み

### 2.1 事前学習済みモデルの読み込み

```python
# モデル名の指定（利用可能なGPT-2のバリエーション）
model_names = {
    'gpt2': 'gpt2',                    # 117M parameters (最小)
    'gpt2-medium': 'gpt2-medium',      # 345M parameters
    'gpt2-large': 'gpt2-large',        # 762M parameters  
    'gpt2-xl': 'gpt2-xl'               # 1.5B parameters (最大)
}

# 使用するモデルを選択（メモリ制限に応じて調整）
model_name = 'gpt2'  # Google Colab無料版では 'gpt2' または 'gpt2-medium' を推奨

print(f"モデル '{model_name}' を読み込み中...")

# Tokenizerの読み込み
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# モデルの読み込み
model = GPT2LMHeadModel.from_pretrained(model_name)

# パディングトークンの設定（GPT-2はデフォルトでパディングトークンがない）
tokenizer.pad_token = tokenizer.eos_token

print("モデルとTokenizerの読み込み完了！")
print(f"語彙サイズ: {tokenizer.vocab_size}")
print(f"モデルパラメータ数: {model.num_parameters():,}")
```

### 2.2 GPUへの移動（利用可能な場合）

```python
# デバイスの設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

print(f"デバイス: {device}")

# GPU使用時のメモリ使用量確認
if torch.cuda.is_available():
    print(f"GPU メモリ使用量: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

## 3. 基本的な文章生成

### 3.1 シンプルな生成関数

```python
def generate_text(prompt, max_length=100, temperature=0.8, top_p=0.9, num_return_sequences=1):
    """
    GPT-2を使用してテキストを生成する関数
    
    Args:
        prompt (str): 入力プロンプト
        max_length (int): 生成する最大トークン数
        temperature (float): 生成のランダム性（高いほどクリエイティブ）
        top_p (float): Nucleus sampling の閾値
        num_return_sequences (int): 生成するシーケンス数
    
    Returns:
        list: 生成されたテキストのリスト
    """
    
    # プロンプトのトークン化
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    print(f"入力プロンプト: '{prompt}'")
    print(f"入力トークン数: {input_ids.shape[1]}")
    print(f"生成設定: max_length={max_length}, temperature={temperature}, top_p={top_p}")
    print("=" * 50)
    
    # モデルを評価モードに設定
    model.eval()
    
    # テキスト生成（勾配計算を無効化してメモリ節約）
    with torch.no_grad():
        output_sequences = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            do_sample=True,               # サンプリングを有効化
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1        # 繰り返しを防ぐ
        )
    
    # 生成されたテキストをデコード
    generated_texts = []
    for i, sequence in enumerate(output_sequences):
        # 入力プロンプト部分を除去
        generated_sequence = sequence[input_ids.shape[1]:]
        text = tokenizer.decode(generated_sequence, skip_special_tokens=True)
        generated_texts.append(text)
        
        print(f"生成結果 {i+1}:")
        print(f"'{text}'")
        print("-" * 30)
    
    return generated_texts

# 使用例
prompt = "Once upon a time"
results = generate_text(prompt, max_length=50)
```

### 3.2 日本語対応版（多言語モデル使用）

```python
# 日本語対応のGPT-2モデル（rinna社など）
# 注意: 日本語用の事前学習済みモデルは別途インストールが必要

def setup_japanese_gpt2():
    """日本語GPT-2モデルのセットアップ（オプション）"""
    try:
        # rinna社の日本語GPT-2モデル例
        # !pip install fugashi ipadic
        
        from transformers import T5Tokenizer, AutoModelForCausalLM
        
        model_name = "rinna/japanese-gpt2-medium"
        
        tokenizer_ja = T5Tokenizer.from_pretrained(model_name)
        model_ja = AutoModelForCausalLM.from_pretrained(model_name)
        
        return tokenizer_ja, model_ja
    except:
        print("日本語モデルの読み込みに失敗しました。英語モデルを使用してください。")
        return None, None

# 日本語モデルの使用例（オプション）
# tokenizer_ja, model_ja = setup_japanese_gpt2()
```

## 4. 高度な生成設定とカスタマイズ

### 4.1 異なるサンプリング手法の比較

```python
def compare_sampling_methods(prompt, max_length=80):
    """異なるサンプリング手法を比較"""
    
    print(f"プロンプト: '{prompt}'")
    print("=" * 60)
    
    # 1. Greedy Decoding（決定的）
    print("1. Greedy Decoding（最も確率の高いトークンを選択）")
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        greedy_output = model.generate(
            input_ids,
            max_length=max_length,
            do_sample=False,  # Greedyデコーディング
            pad_token_id=tokenizer.eos_token_id
        )
    
    greedy_text = tokenizer.decode(greedy_output[0][input_ids.shape[1]:], skip_special_tokens=True)
    print(f"結果: {greedy_text}")
    print()
    
    # 2. Temperature Sampling
    print("2. Temperature Sampling（温度制御）")
    temperatures = [0.3, 0.8, 1.2]
    
    for temp in temperatures:
        with torch.no_grad():
            temp_output = model.generate(
                input_ids,
                max_length=max_length,
                temperature=temp,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        temp_text = tokenizer.decode(temp_output[0][input_ids.shape[1]:], skip_special_tokens=True)
        print(f"  Temperature {temp}: {temp_text}")
    print()
    
    # 3. Top-p (Nucleus) Sampling
    print("3. Top-p Sampling（核サンプリング）")
    top_p_values = [0.5, 0.8, 0.95]
    
    for p in top_p_values:
        with torch.no_grad():
            nucleus_output = model.generate(
                input_ids,
                max_length=max_length,
                temperature=0.8,
                top_p=p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        nucleus_text = tokenizer.decode(nucleus_output[0][input_ids.shape[1]:], skip_special_tokens=True)
        print(f"  Top-p {p}: {nucleus_text}")
    print()

# 使用例
compare_sampling_methods("The future of artificial intelligence")
```

### 4.2 バッチ処理での複数生成

```python
def batch_generate(prompts, max_length=60, batch_size=4):
    """複数のプロンプトを効率的にバッチ処理"""
    
    print(f"バッチサイズ: {batch_size}")
    print(f"プロンプト数: {len(prompts)}")
    print("=" * 50)
    
    all_results = []
    
    # バッチ単位で処理
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        
        # プロンプトをトークン化
        inputs = tokenizer(batch_prompts, return_tensors='pt', padding=True, truncation=True)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # デコードして結果を保存
        for j, output in enumerate(outputs):
            original_length = input_ids[j].shape[0]
            generated = output[original_length:]
            text = tokenizer.decode(generated, skip_special_tokens=True)
            
            print(f"プロンプト {i+j+1}: '{batch_prompts[j]}'")
            print(f"生成結果: {text}")
            print("-" * 30)
            
            all_results.append(text)
    
    return all_results

# 使用例
test_prompts = [
    "The weather today is",
    "In the distant future",
    "My favorite hobby is",
    "The secret to happiness"
]

batch_results = batch_generate(test_prompts)
```

## 5. 実用的な応用例

### 5.1 ストーリー生成

```python
def generate_story(initial_sentence, num_paragraphs=3, sentences_per_paragraph=3):
    """段落構成のストーリーを生成"""
    
    print(f"ストーリーの開始: '{initial_sentence}'")
    print("=" * 60)
    
    current_text = initial_sentence
    story_parts = [initial_sentence]
    
    for paragraph in range(num_paragraphs):
        print(f"\n段落 {paragraph + 1}:")
        paragraph_text = ""
        
        for sentence in range(sentences_per_paragraph):
            # 現在のテキストを元に次の文を生成
            input_ids = tokenizer.encode(current_text, return_tensors='pt').to(device)
            
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_length=input_ids.shape[1] + 30,  # 適度な長さ
                    temperature=0.8,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # 新しい部分のみ抽出
            new_tokens = output[0][input_ids.shape[1]:]
            new_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            # 最初の文を取得（句点で区切り）
            sentences = new_text.split('.')
            if len(sentences) > 1:
                new_sentence = sentences[0] + '.'
            else:
                new_sentence = new_text
            
            paragraph_text += " " + new_sentence
            current_text += " " + new_sentence
        
        story_parts.append(paragraph_text.strip())
        print(paragraph_text.strip())
    
    return story_parts

# 使用例
story = generate_story("In a small village nestled between mountains")
```

### 5.2 質問応答形式

```python
def qa_generation(context, question):
    """文脈に基づく質問応答"""
    
    # プロンプトテンプレートの作成
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    
    print(f"文脈: {context}")
    print(f"質問: {question}")
    print("-" * 40)
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=input_ids.shape[1] + 50,
            temperature=0.3,  # より確実な回答のため低めに設定
            top_p=0.8,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.encode('\n')[0]  # 改行で停止
        )
    
    # 回答部分のみ抽出
    answer_tokens = output[0][input_ids.shape[1]:]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
    
    print(f"回答: {answer}")
    return answer

# 使用例
context = "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France."
question = "Where is the Eiffel Tower located?"
qa_generation(context, question)
```

## 6. メモリ管理とパフォーマンス最適化

### 6.1 メモリ使用量の監視

```python
def monitor_memory():
    """GPU/CPUメモリ使用量の監視"""
    
    import psutil
    
    print("=== メモリ使用量 ===")
    
    # CPU RAM
    ram = psutil.virtual_memory()
    print(f"RAM使用量: {ram.used / 1e9:.1f} GB / {ram.total / 1e9:.1f} GB ({ram.percent:.1f}%)")
    
    # GPU メモリ（CUDA利用可能時）
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated()
        gpu_reserved = torch.cuda.memory_reserved()
        gpu_total = torch.cuda.get_device_properties(0).total_memory
        
        print(f"GPU使用量: {gpu_memory / 1e9:.1f} GB")
        print(f"GPU予約量: {gpu_reserved / 1e9:.1f} GB") 
        print(f"GPU総容量: {gpu_total / 1e9:.1f} GB")
        print(f"GPU使用率: {(gpu_memory / gpu_total) * 100:.1f}%")

# メモリクリーンアップ関数
def cleanup_memory():
    """メモリのクリーンアップ"""
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    import gc
    gc.collect()
    
    print("メモリクリーンアップ完了")

# 使用例
monitor_memory()
```

### 6.2 効率的な生成設定

```python
def optimized_generation(prompt, max_length=100):
    """メモリ効率を考慮した生成"""
    
    # メモリ効率のための設定
    generation_config = {
        'max_length': max_length,
        'temperature': 0.8,
        'top_p': 0.9,
        'do_sample': True,
        'pad_token_id': tokenizer.eos_token_id,
        'use_cache': True,  # KVキャッシュを使用
        'output_scores': False,  # スコア出力を無効化してメモリ節約
        'return_dict_in_generate': False
    }
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # 自動混合精度を使用（GPU利用時）
    if torch.cuda.is_available():
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                output = model.generate(input_ids, **generation_config)
    else:
        with torch.no_grad():
            output = model.generate(input_ids, **generation_config)
    
    # 結果のデコード
    generated_text = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
    
    return generated_text

# 使用例
result = optimized_generation("The benefits of renewable energy include")
print(result)
```

## 7. Google Colab 特有の注意点

### 7.1 セッション管理

```python
# セッションタイムアウト対策
import time
from IPython.display import Javascript

def keep_alive():
    """セッションを維持する関数"""
    display(Javascript('''
        function KeepClicking(){
            console.log("Clicking");
            document.querySelector("colab-connect-button").click()
        }
        setInterval(KeepClicking, 60000)
    '''))

# 長時間の処理前に実行
# keep_alive()
```

### 7.2 モデルの保存と読み込み

```python
# モデルをGoogle Driveに保存（オプション）
def save_model_to_drive():
    """モデルをGoogle Driveに保存"""
    
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        
        # 保存パス
        save_path = '/content/drive/MyDrive/gpt2_model'
        
        # モデルとTokenizerを保存
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        
        print(f"モデルを {save_path} に保存しました")
        
    except Exception as e:
        print(f"保存に失敗: {e}")

def load_model_from_drive():
    """Google Driveからモデルを読み込み"""
    
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        
        load_path = '/content/drive/MyDrive/gpt2_model'
        
        tokenizer_loaded = GPT2Tokenizer.from_pretrained(load_path)
        model_loaded = GPT2LMHeadModel.from_pretrained(load_path)
        
        print(f"モデルを {load_path} から読み込みました")
        return tokenizer_loaded, model_loaded
        
    except Exception as e:
        print(f"読み込みに失敗: {e}")
        return None, None
```

## 8. インタラクティブなデモ

### 8.1 簡単なWebインターフェース

```python
# Gradioを使用したWebインターフェース（オプション）
try:
    import gradio as gr
    
    def gradio_generate(prompt, max_length, temperature, top_p):
        """Gradio用の生成関数"""
        result = generate_text(
            prompt, 
            max_length=int(max_length),
            temperature=float(temperature),
            top_p=float(top_p),
            num_return_sequences=1
        )
        return result[0] if result else ""
    
    # インターフェースの作成
    demo = gr.Interface(
        fn=gradio_generate,
        inputs=[
            gr.Textbox(label="プロンプト", placeholder="ここにプロンプトを入力してください..."),
            gr.Slider(50, 200, value=100, label="最大長"),
            gr.Slider(0.1, 2.0, value=0.8, label="Temperature"),
            gr.Slider(0.1, 1.0, value=0.9, label="Top-p")
        ],
        outputs=gr.Textbox(label="生成されたテキスト"),
        title="GPT-2 テキスト生成",
        description="プロンプトを入力してテキストを生成します"
    )
    
    # インターフェースの起動
    # demo.launch(share=True)  # コメントアウト（必要に応じて実行）
    
except ImportError:
    print("Gradioがインストールされていません。以下のコマンドでインストールできます:")
    print("!pip install gradio")
```

### 8.2 コマンドライン風インターフェース

```python
def interactive_session():
    """対話的なテキスト生成セッション"""
    
    print("=== GPT-2 対話的テキスト生成セッション ===")
    print("終了するには 'quit' を入力してください")
    print("設定を変更するには 'config' を入力してください")
    print()
    
    # デフォルト設定
    settings = {
        'max_length': 100,
        'temperature': 0.8,
        'top_p': 0.9
    }
    
    while True:
        prompt = input("プロンプトを入力してください: ")
        
        if prompt.lower() == 'quit':
            print("セッションを終了します。")
            break
        
        elif prompt.lower() == 'config':
            print(f"現在の設定: {settings}")
            try:
                new_max_length = input(f"最大長 (現在: {settings['max_length']}): ")
                if new_max_length:
                    settings['max_length'] = int(new_max_length)
                
                new_temperature = input(f"Temperature (現在: {settings['temperature']}): ")
                if new_temperature:
                    settings['temperature'] = float(new_temperature)
                
                new_top_p = input(f"Top-p (現在: {settings['top_p']}): ")
                if new_top_p:
                    settings['top_p'] = float(new_top_p)
                
                print("設定を更新しました。")
            except ValueError:
                print("無効な値です。設定は変更されませんでした。")
            continue
        
        elif prompt.strip():
            try:
                print("\n生成中...")
                result = generate_text(
                    prompt,
                    max_length=settings['max_length'],
                    temperature=settings['temperature'],
                    top_p=settings['top_p'],
                    num_return_sequences=1
                )
                print(f"\n生成結果: {result[0]}\n")
                
            except Exception as e:
                print(f"エラーが発生しました: {e}")
        
        else:
            print("プロンプトを入力してください。")

# 対話セッションの開始（必要に応じてコメントアウト解除）
# interactive_session()
```

## 9. まとめと次のステップ

### 9.1 学習したこと

このチュートリアルでは以下を学びました：

1. **環境設定**: Google ColabでのHugging Face Transformersのセットアップ
2. **モデル読み込み**: 事前学習済みGPT-2モデルとTokenizerの使用方法
3. **基本的な生成**: シンプルなテキスト生成の実装
4. **高度な制御**: 様々なサンプリング手法とパラメータ調整
5. **実用的な応用**: ストーリー生成、質問応答など
6. **最適化**: メモリ管理とパフォーマンスの改善
7. **Colab特有の注意点**: セッション管理とモデル保存

### 9.2 さらなる学習リソース

```python
# 参考リンク集
resources = {
    "公式ドキュメント": [
        "Hugging Face Transformers: https://huggingface.co/docs/transformers",
        "GPT-2 Model Card: https://huggingface.co/gpt2"
    ],
    "チュートリアル": [
        "Hugging Face Course: https://huggingface.co/course/",
        "PyTorch Tutorials: https://pytorch.org/tutorials/"
    ],
    "コミュニティ": [
        "Hugging Face Forums: https://discuss.huggingface.co/",
        "Reddit r/MachineLearning: https://www.reddit.com/r/MachineLearning/"
    ]
}

for category, links in resources.items():
    print(f"{category}:")
    for link in links:
        print(f"  - {link}")
```

### 9.3 次のステップ

- **ファインチューニング**: 独自のデータセットでのモデル調整
- **より大きなモデル**: GPT-3、GPT-4、Claude などのAPI利用
- **他のタスク**: 翻訳、要約、質問応答などの特化タスク
- **本番環境**: FastAPI、Flask などでのWebアプリケーション化

```python
print("🎉 GPT-2 テキスト生成チュートリアル完了!")
print("Happy coding! 🚀")
```