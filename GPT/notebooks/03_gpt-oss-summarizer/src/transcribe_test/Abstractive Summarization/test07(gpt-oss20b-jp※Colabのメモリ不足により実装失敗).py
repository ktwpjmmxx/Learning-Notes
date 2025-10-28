# Google Colab 生成型テキスト要約システム v7 (GPT-OSS-20B版)

# 必要なライブラリのインストール
print("必要なライブラリをインストール中...")
!pip install -q transformers torch accelerate sentencepiece

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import ipywidgets as widgets
from IPython.display import display, HTML
import warnings
import gc
warnings.filterwarnings('ignore')

print("✓ ライブラリのインポート完了!")

# メモリクリア
gc.collect()
torch.cuda.empty_cache()

print("\n" + "="*60)
print("GPT-OSS-20B モデルを読み込んでいます...")
print("="*60)
print(" 初回は10-15分かかります(約8GB ダウンロード)")
print("MXFP4量子化済み(16GB以下で動作)")
print("="*60 + "\n")

# グローバル変数として定義
tokenizer = None
model = None
device = None

try:
    model_name = "openai/gpt-oss-20b"
    
    print(f"📥 Step 1/3: トークナイザーをダウンロード中...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # パディングトークンの設定
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"📥 Step 2/3: モデルをダウンロード中(8-10分)...")
    
    # MXFP4で量子化済みのモデルをそのまま読み込む
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16  # メモリ効率化
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"✓ Step 3/3: モデルの読み込み完了!")
    print(f"✓ デバイス: {device}")
    print(f"✓ 量子化: MXFP4 (OpenAI公式)")
    print(f"✓ モデル: {model_name}")
    
    # メモリ使用量を表示
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU使用量: {memory_allocated:.2f}GB (予約: {memory_reserved:.2f}GB)")
    
except Exception as e:
    print(f"\n❌ エラー: {e}")
    print("\n💡 トラブルシューティング:")
    print("1. ランタイムを再起動してください")
    print("2. GPU有効化を確認: ランタイム > ランタイムのタイプを変更 > T4 GPU")
    print("3. メモリ不足の場合: ランタイム > セッションの管理 > すべて終了")
    import traceback
    print("\n詳細:")
    print(traceback.format_exc())

def generate_summary_gpt(text, max_new_tokens=150, temperature=0.7):
    """
    GPT-OSS-20Bで要約を生成
    
    Parameters:
    - text: 入力テキスト
    - max_new_tokens: 生成する最大トークン数
    - temperature: 生成の多様性(0.1-1.0)
    """
    
    # 日本語要約用プロンプト(Harmony形式を参考)
    prompt = f"""以下のテキストを簡潔に要約してください。

【入力テキスト】
{text}

【要約】
"""
    
    # トークナイズ
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=1024,
        truncation=True,
        padding=True
    )
    
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # デコード
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # プロンプト部分を削除して要約のみ抽出
    if "【要約】" in generated_text:
        summary = generated_text.split("【要約】")[-1].strip()
    else:
        # フォールバック: 生成されたテキスト全体から元のテキストを除去
        summary = generated_text.replace(prompt, "").strip()
    
    # クリーニング
    summary = summary.strip()
    
    # 空の場合や短すぎる場合
    if not summary or len(summary) < 10:
        return "(要約生成に失敗しました。パラメータを調整してください。)"
    
    return summary

# UIコンポーネントの作成
input_text = widgets.Textarea(
    value='',
    placeholder='要約したいテキストをここに入力してください...\nGPT-OSS-20Bが日本語要約を生成します。',
    description='入力:',
    disabled=False,
    layout=widgets.Layout(width='95%', height='200px')
)

summarize_button = widgets.Button(
    description='要約生成',
    button_style='primary',
    tooltip='クリックして要約を生成(20-60秒)',
    icon='rocket',
    layout=widgets.Layout(width='200px', height='40px')
)

output_text = widgets.Textarea(
    value='',
    placeholder='生成された要約文がここに表示されます...',
    description='要約:',
    disabled=True,
    layout=widgets.Layout(width='95%', height='150px')
)

status_label = widgets.HTML(
    value='<p style="color: #666;">テキストを入力して「要約生成」ボタンをクリックしてください。</p>'
)

# 生成パラメータ
max_tokens_slider = widgets.IntSlider(
    value=150,
    min=50,
    max=300,
    step=25,
    description='生成長:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d'
)

temperature_slider = widgets.FloatSlider(
    value=0.7,
    min=0.1,
    max=1.0,
    step=0.1,
    description='多様性:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.1f'
)

# 要約処理の関数
def on_summarize_click(b):
    global tokenizer, model
    
    if tokenizer is None or model is None:
        status_label.value = '<p style="color: red;">❌ モデルが読み込まれていません。上のセルを実行してください。</p>'
        output_text.value = ''
        return
    
    text = input_text.value.strip()
    
    if not text:
        status_label.value = '<p style="color: red;">⚠ テキストを入力してください。</p>'
        output_text.value = ''
        return
    
    if len(text) < 50:
        status_label.value = '<p style="color: orange;">⚠ より長いテキスト(50文字以上)を推奨します。</p>'
    
    try:
        status_label.value = '<p style="color: blue;">⏳ GPT-OSS-20Bで要約生成中... (20-60秒かかります)</p>'
        
        # パラメータの取得
        max_tokens = max_tokens_slider.value
        temp = temperature_slider.value
        
        # 要約生成
        summary = generate_summary_gpt(
            text,
            max_new_tokens=max_tokens,
            temperature=temp
        )
        
        output_text.value = summary
        
        if "(要約生成に失敗しました" in summary:
            status_label.value = '<p style="color: red;">⚠ 要約生成に失敗しました。パラメータを調整してください。</p>'
        else:
            char_reduction = round((1 - len(summary) / len(text)) * 100, 1) if len(summary) < len(text) else 0
            status_label.value = f'''
            <p style="color: green;">✓ 要約が生成されました!</p>
            <p style="color: #666; font-size: 12px;">
            入力: {len(text)}文字 → 出力: {len(summary)}文字 | 圧縮率: {char_reduction}%
            </p>
            '''
        
    except Exception as e:
        status_label.value = f'<p style="color: red;">❌ エラー: {str(e)}</p>'
        output_text.value = ''
        import traceback
        print(traceback.format_exc())
        
        # メモリクリア
        gc.collect()
        torch.cuda.empty_cache()

summarize_button.on_click(on_summarize_click)

# UIの表示
display(HTML("""
<h2>🚀 生成型テキスト要約システム v7 (GPT-OSS-20B 修正版)</h2>
<p><strong>モデル:</strong> OpenAI GPT-OSS-20B (21Bパラメータ、3.6Bアクティブ)</p>
<p><strong>特徴:</strong> MXFP4量子化済み、16GB以下で動作</p>
<p><strong>ライセンス:</strong> Apache 2.0 (商用利用可能)</p>
<hr>
"""))

display(input_text)

display(HTML("<h4>⚙️ 生成パラメータ</h4>"))
display(max_tokens_slider)
display(HTML("<p style='color: #666; font-size: 12px; margin-top: -10px;'>生成する最大トークン数(約50-200トークン = 100-400文字)</p>"))

display(temperature_slider)
display(HTML("<p style='color: #666; font-size: 12px; margin-top: -10px;'>0.1=決定的、1.0=創造的</p>"))

display(summarize_button)
display(status_label)
display(HTML("<hr>"))
display(output_text)

print("\n" + "="*60)
print("✓ セットアップ完了！GPT-OSS-20B要約システムが起動しました")
print("="*60)
print("\n🎯 推奨設定:")
print("- 生成長: 150トークン")
print("- 多様性: 0.7")
print("\n⏱️  処理時間: 20-60秒/回")
print("\n💡 ヒント:")
print("- 初回生成は時間がかかります")
print("- 日本語は英語より長くなる傾向があります")
print("- エラーが出たら、ランタイムを再起動してください")
print("\n⚠️  注意:")
print("- GPT-OSS-20Bは主に英語で訓練されています")
print("- 日本語要約の品質は限定的な可能性があります")
print("- メモリ不足の場合は、他のセルを閉じてください")