# Google Colab 生成型テキスト要約システム v6

# 必要なライブラリのインストール
!pip install -q transformers torch sentencepiece accelerate

import torch
from transformers import pipeline
import ipywidgets as widgets
from IPython.display import display, HTML
import warnings
warnings.filterwarnings('ignore')

print("ライブラリのインポート完了!")

# 生成型要約モデルの初期化
print("モデルを読み込んでいます... (初回は数分かかります)")

model = None
summarizer = None

try:
    # GPU が利用可能か確認
    device = 0 if torch.cuda.is_available() else -1
    
    # 英語の高品質要約モデルを使用
    model_name = "facebook/bart-large-cnn"
    
    # 要約パイプラインの作成
    summarizer = pipeline(
        "summarization",
        model=model_name,
        device=device
    )
    
    print(f"✓ モデルの読み込み完了! (モデル: {model_name})")
    print(f"✓ デバイス: {'GPU' if device == 0 else 'CPU'}")
    print("✓ 生成型要約 - 新しい言葉で要約を生成します")
    
except Exception as e:
    print(f"❌ エラー: {e}")
    print("モデルの読み込みに失敗しました。")
    import traceback
    print(traceback.format_exc())

# UIコンポーネントの作成
input_text = widgets.Textarea(
    value='',
    placeholder='Enter English text to summarize...\n(Longer texts with multiple sentences work best)',
    description='Input:',
    disabled=False,
    layout=widgets.Layout(width='95%', height='200px')
)

summarize_button = widgets.Button(
    description='Generate Summary',
    button_style='primary',
    tooltip='Click to generate a new summary',
    icon='magic',
    layout=widgets.Layout(width='220px', height='40px')
)

output_text = widgets.Textarea(
    value='',
    placeholder='Generated summary will appear here...',
    description='Summary:',
    disabled=True,
    layout=widgets.Layout(width='95%', height='150px')
)

status_label = widgets.HTML(
    value='<p style="color: #666;">Enter text and click "Generate Summary" to create a new summary.</p>'
)

# 要約の長さ設定
max_length_slider = widgets.IntSlider(
    value=130,
    min=30,
    max=200,
    step=10,
    description='Max Length:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d'
)

min_length_slider = widgets.IntSlider(
    value=30,
    min=10,
    max=100,
    step=10,
    description='Min Length:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d'
)

# 生成の多様性設定
creativity_slider = widgets.FloatSlider(
    value=0.0,
    min=0.0,
    max=1.0,
    step=0.1,
    description='Creativity:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.1f'
)

# 要約生成の関数
def on_summarize_click(b):
    global summarizer
    
    # モデルが読み込まれているかチェック
    if summarizer is None:
        status_label.value = '<p style="color: red;">❌ Model not loaded. Please restart the cell.</p>'
        output_text.value = ''
        return
    
    text = input_text.value.strip()
    
    if not text:
        status_label.value = '<p style="color: red;">⚠ Please enter text.</p>'
        output_text.value = ''
        return
    
    # 最小文字数チェック
    if len(text) < 100:
        status_label.value = '<p style="color: orange;">⚠ Text is too short. Please enter at least 100 characters.</p>'
        output_text.value = ''
        return
    
    try:
        status_label.value = '<p style="color: blue;">⏳ Generating summary...</p>'
        
        # 入力の長さを取得
        input_tokens = len(summarizer.tokenizer.encode(text))
        
        # max_lengthを入力より短く自動調整
        adjusted_max = min(max_length_slider.value, int(input_tokens * 0.7))
        adjusted_min = min(min_length_slider.value, int(input_tokens * 0.3))
        adjusted_max = max(adjusted_max, adjusted_min + 10)
        
        # 生成パラメータ
        do_sample = creativity_slider.value > 0.0
        
        # 要約生成
        summary = summarizer(
            text,
            max_length=adjusted_max,
            min_length=adjusted_min,
            do_sample=do_sample,
            temperature=creativity_slider.value if do_sample else 1.0,
            num_beams=4 if not do_sample else 1,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
        
        output_text.value = summary[0]['summary_text']
        
        # 統計情報
        output_tokens = len(summarizer.tokenizer.encode(summary[0]['summary_text']))
        compression = round((1 - output_tokens / input_tokens) * 100, 1)
        
        status_label.value = f'''
        <p style="color: green;">✓ Summary generated!</p>
        <p style="color: #666; font-size: 12px;">
        Input: {input_tokens} tokens ({len(text)} chars) → Output: {output_tokens} tokens ({len(summary[0]['summary_text'])} chars) | Compression: {compression}%
        </p>
        '''
        
    except Exception as e:
        status_label.value = f'<p style="color: red;">Error: {str(e)}</p>'
        output_text.value = ''
        import traceback
        print(traceback.format_exc())

# ボタンにイベントハンドラを設定
summarize_button.on_click(on_summarize_click)

# UIの表示
display(HTML("""
<h2>✨ Generative Text Summarization System v6</h2>
<p><strong>Feature:</strong> Creates new summaries in different words (English only)</p>
<p style="color: #666; font-size: 14px;">⚠️ Note: Generative models may occasionally produce inaccurate information</p>
<hr>
"""))

display(input_text)
display(HTML("<h4>⚙️ Settings</h4>"))
display(min_length_slider)
display(max_length_slider)
display(HTML("<h4>🎨 Creativity (Advanced)</h4>"))
display(creativity_slider)
display(HTML("<p style='color: #666; font-size: 12px;'>0.0 = Deterministic (same result), 0.5-1.0 = More creative (different results)</p>"))
display(summarize_button)
display(status_label)
display(HTML("<hr>"))
display(output_text)

print("\n" + "="*60)
print("✓ Setup complete! Generative summarization system ready")
print("="*60)
print("\n💡 Tips:")
print("- Works best with English text (100+ characters)")
print("- Generates NEW words based on understanding")
print("- Results may vary with creativity settings")
print("- Compare with extractive method (v5) for accuracy")