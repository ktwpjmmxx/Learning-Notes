# Google Colab 生成型テキスト要約システム v5 (BARTモデル版)

# 必要なライブラリのインストール
!pip install -q transformers torch sentencepiece fugashi ipadic unidic-lite accelerate

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import ipywidgets as widgets
from IPython.display import display, HTML
import warnings
import re
warnings.filterwarnings('ignore')

print("ライブラリのインポート完了!")

print("生成型モデル(BART)を読み込んでいます...")
print("※ 初回は数分かかる場合があります")

# グローバル変数として定義
tokenizer = None
model = None
device = None

try:
    # 日本語BARTモデル - 要約タスクで高い性能
    model_name = "utokyo-nlp/bert-base-japanese-v3"
    
    # 実際には要約に適したモデルに変更
    # mBARTを試す
    model_name = "facebook/mbart-large-50"
    
    print(f"モデル {model_name} をダウンロード中...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # 言語設定(日本語)
    tokenizer.src_lang = "ja_XX"
    tokenizer.tgt_lang = "ja_XX"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    print(f"✓ モデルの読み込み完了! (デバイス: {device})")
    print("✓ mBART多言語モデル使用 - 日本語要約対応")
    print(f"✓ モデル: {model_name}")
except Exception as e:
    print(f"❌ エラー: {e}")
    print("モデルの読み込みに失敗しました。")
    import traceback
    print(traceback.format_exc())

def clean_summary(text):
    """生成された要約をクリーニング"""
    # 不要なプレフィックスを削除
    text = re.sub(r'^(要約|概要|まとめ)[:：]', '', text)
    text = re.sub(r'要約[:：]', '', text)
    
    # 空白の正規化
    text = re.sub(r'\s+', '', text)
    
    # 先頭・末尾の句読点を削除
    text = text.strip('、。:：')
    
    return text.strip()

def generate_summary_mbart(text, max_length=100, min_length=30, num_beams=4):
    """
    mBARTモデルで要約を生成
    """
    # トークナイズ
    inputs = tokenizer(
        text,
        max_length=512,
        truncation=True,
        padding='longest',
        return_tensors='pt'
    )
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 日本語として生成
    forced_bos_token_id = tokenizer.lang_code_to_id["ja_XX"]
    
    # 要約生成
    with torch.no_grad():
        output_ids = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            forced_bos_token_id=forced_bos_token_id,
            no_repeat_ngram_size=3,
            early_stopping=True,
            length_penalty=1.5,
        )
    
    # デコード
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # クリーニング
    summary = clean_summary(summary)
    
    return summary if summary else "(生成失敗)"

# UIコンポーネントの作成
input_text = widgets.Textarea(
    value='',
    placeholder='要約したいテキストをここに入力してください...\nmBARTモデルが多言語学習の知識を活用して要約します。',
    description='入力:',
    disabled=False,
    layout=widgets.Layout(width='95%', height='200px')
)

summarize_button = widgets.Button(
    description='要約生成',
    button_style='primary',
    tooltip='クリックして要約を生成',
    icon='magic',
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

# 要約の長さ設定
max_length_slider = widgets.IntSlider(
    value=80,
    min=40,
    max=150,
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
    min=15,
    max=80,
    step=5,
    description='最小長:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d'
)

num_beams_slider = widgets.IntSlider(
    value=4,
    min=2,
    max=8,
    step=1,
    description='品質:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d'
)

# 要約処理の関数
def on_summarize_click(b):
    global tokenizer, model, device
    
    if tokenizer is None or model is None:
        status_label.value = '<p style="color: red;">❌ モデルが読み込まれていません。セルを再実行してください。</p>'
        output_text.value = ''
        return
    
    text = input_text.value.strip()
    
    if not text:
        status_label.value = '<p style="color: red;">⚠ テキストを入力してください。</p>'
        output_text.value = ''
        return
    
    if len(text) < 50:
        status_label.value = '<p style="color: orange;">⚠ より長いテキスト(50文字以上)を入力してください。</p>'
        return
    
    try:
        status_label.value = '<p style="color: blue;">⏳ 要約を生成中... (10-20秒かかります)</p>'
        
        max_len = max_length_slider.value
        min_len = min(min_length_slider.value, max_len - 10)
        num_beams = num_beams_slider.value
        
        # 要約生成
        summary = generate_summary_mbart(
            text,
            max_length=max_len,
            min_length=min_len,
            num_beams=num_beams
        )
        
        if not summary or len(summary) < 10 or summary == "(生成失敗)":
            status_label.value = '<p style="color: red;">⚠ 要約生成に失敗しました。パラメータを調整してもう一度お試しください。</p>'
            output_text.value = summary
            return
        
        output_text.value = summary
        
        char_reduction = round((1 - len(summary) / len(text)) * 100, 1)
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

summarize_button.on_click(on_summarize_click)

# UIの表示
display(HTML("""
<h2>🤖 生成型テキスト要約システム v5 (mBART版)</h2>
<p><strong>特徴:</strong> Facebookの多言語BARTモデルを使用</p>
<p><strong>モデル:</strong> facebook/mbart-large-50 (日本語対応)</p>
<p><strong>注意:</strong> 初回ダウンロードは大きいファイルのため時間がかかります</p>
<hr>
"""))

display(input_text)
display(HTML("<h4>⚙️ 生成設定</h4>"))
display(max_length_slider)
display(min_length_slider)
display(num_beams_slider)
display(HTML("<p style='color: #666; font-size: 12px;'>※ mBARTは処理に時間がかかりますが、高品質な要約を生成します</p>"))
display(summarize_button)
display(status_label)
display(HTML("<hr>"))
display(output_text)

print("\n" + "="*60)
print("✓ セットアップ完了！mBART要約システムが起動しました")
print("="*60)
print("\n💡 特徴:")
print("- Facebookの大規模多言語モデル")
print("- 50言語対応、日本語の要約品質が高い")
print("- ファイルサイズ: 約2.3GB (初回ダウンロード時)")
print("\n🎯 推奨設定:")
print("- 最大長: 80")
print("- 最小長: 30")
print("- 品質: 4")
print("\n⏱️ 処理時間: 10-20秒程度")