# Google Colab テキスト要約システム

!pip install -q transformers torch sentencepiece accelerate

import torch
from transformers import pipeline
import ipywidgets as widgets
from IPython.display import display, HTML

print("ライブラリのインポート完了!")

print("モデルを読み込んでいます... (初回は数分かかる場合があります)")

try:
    
    device = 0 if torch.cuda.is_available() else -1
    
    model_name = "csebuetnlp/mT5_multilingual_XLSum"
    
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    if device == 0:
        model = model.to('cuda')
    
    print(f"✓ モデルの読み込み完了! (モデル: {model_name}, デバイス: {'GPU' if device == 0 else 'CPU'})")
    
except Exception as e:
    print(f"エラー: {e}")
    print("モデルの読み込みに失敗しました。")

# UIコンポーネント
input_text = widgets.Textarea(
    value='',
    placeholder='要約したいテキストをここに入力してください...',
    description='入力:',
    disabled=False,
    layout=widgets.Layout(width='95%', height='200px')
)

summarize_button = widgets.Button(
    description='要約する',
    button_style='primary',
    tooltip='クリックして要約を生成',
    icon='check',
    layout=widgets.Layout(width='200px', height='40px')
)

output_text = widgets.Textarea(
    value='',
    placeholder='要約結果がここに表示されます...',
    description='要約:',
    disabled=True,
    layout=widgets.Layout(width='95%', height='150px')
)

status_label = widgets.HTML(
    value='<p style="color: #666;">テキストを入力して「要約する」ボタンをクリックしてください。</p>'
)

# 長さ設定のスライダー
max_length_slider = widgets.IntSlider(
    value=130,
    min=50,
    max=300,
    step=10,
    description='最大長:',
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
    description='最小長:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d'
)

def on_summarize_click(b):
    text = input_text.value.strip()
    
    if not text:
        status_label.value = '<p style="color: red;">⚠ テキストを入力してください。</p>'
        output_text.value = ''
        return
    
    if len(text) < 50:
        status_label.value = '<p style="color: orange;">⚠ テキストが短すぎます。より長いテキストを入力してください。</p>'
        output_text.value = ''
        return
    
    try:
        status_label.value = '<p style="color: blue;">⏳ 要約を生成中...</p>'
        
        # テキストをトークン化
        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        
        if device == 0:
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
        
        # 入力の長さを取得
        input_length = inputs['input_ids'].shape[1]
        
        # 要約の長さを計算
        adjusted_max_length = min(int(input_length * 0.6), max_length_slider.value)
        adjusted_min_length = min(int(input_length * 0.2), min_length_slider.value)
        adjusted_max_length = max(adjusted_max_length, adjusted_min_length + 10)
        
        # 要約生成
        summary_ids = model.generate(
            inputs['input_ids'],
            max_length=adjusted_max_length,
            min_length=adjusted_min_length,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
        
        # デコード
        summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        output_text.value = summary_text
        status_label.value = f'<p style="color: green;">✓ 要約が完成しました! (入力: {input_length}トークン → 出力: {len(summary_ids[0])}トークン)</p>'
        
    except Exception as e:
        status_label.value = f'<p style="color: red;">エラーが発生しました: {str(e)}</p>'
        output_text.value = ''
        import traceback
        print(traceback.format_exc())

# ボタンにイベントハンドラを設定
summarize_button.on_click(on_summarize_click)

# UI
display(HTML("<h2>テキスト要約システム</h2>"))
display(HTML("<p>テキストを入力して要約ボタンをクリックしてください。</p>"))
display(HTML("<hr>"))

display(input_text)
display(HTML("<h4>要約の長さ設定</h4>"))
display(min_length_slider)
display(max_length_slider)
display(summarize_button)
display(status_label)
display(HTML("<hr>"))
display(output_text)

print("\n" + "="*50)
print("セットアップ完了!")
print("="*50)