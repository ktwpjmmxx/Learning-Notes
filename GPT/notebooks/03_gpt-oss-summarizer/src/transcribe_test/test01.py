
# Google Colab テキスト要約システム

!pip install -q transformers torch sentencepiece accelerate

import torch
from transformers import pipeline
import ipywidgets as widgets
from IPython.display import display, HTML

print("ライブラリのインポート完了!")

print("モデルを読み込んでいます... (初回は数分かかる場合があります)")

# 日本語対応の軽量な要約モデルを使用
try:
    device = 0 if torch.cuda.is_available() else -1
    
    summarizer = pipeline(
        "summarization",
        model="sonoisa/t5-base-japanese",
        device=device
    )
    print(f"✓ モデルの読み込み完了! (デバイス: {'GPU' if device == 0 else 'CPU'})")
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

# 要約処理
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
        
        # 入力テキストのトークン数を推定（おおよそ文字数÷4）
        input_length = len(summarizer.tokenizer.encode(text))
        
        # max_lengthを入力の70%程度に自動調整
        adjusted_max_length = max(min_length_slider.value + 10, int(input_length * 0.7))
        adjusted_max_length = min(adjusted_max_length, max_length_slider.value)
        
        # min_lengthも入力より短くする
        adjusted_min_length = min(min_length_slider.value, int(input_length * 0.3))
        
        # 要約の生成
        summary = summarizer(
            text,
            max_length=adjusted_max_length,
            min_length=adjusted_min_length,
            do_sample=False
        )
        
        output_text.value = summary[0]['summary_text']
        status_label.value = f'<p style="color: green;">✓ 要約が完成しました! (入力: {input_length}トークン → 出力設定: 最大{adjusted_max_length}トークン)</p>'
        
    except Exception as e:
        status_label.value = f'<p style="color: red;">エラーが発生しました: {str(e)}</p>'
        output_text.value = ''

# ボタンにイベントハンドラを設定
summarize_button.on_click(on_summarize_click)

# UIの表示
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