# Google Colab 生成型テキスト要約システム v2 (修正版)

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

print("生成型モデルを読み込んでいます...")
print("※ 初回は数分かかる場合があります")

# グローバル変数として定義
tokenizer = None
model = None
device = None

try:
    # 日本語要約に特化したモデルに変更
    model_name = "sonoisa/t5-base-japanese"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    print(f"✓ モデルの読み込み完了! (デバイス: {device})")
    print("✓ T5日本語モデル使用 - 生成型要約")
    print(f"✓ モデル: {model_name}")
except Exception as e:
    print(f"❌ エラー: {e}")
    print("モデルの読み込みに失敗しました。")
    import traceback
    print(traceback.format_exc())

def clean_summary(text):
    """生成された要約をクリーニング"""
    # 不要なプレフィックスを削除
    text = re.sub(r'^(要約|概要|まとめ)[::]', '', text)
    text = re.sub(r'要約[::]', '', text)
    text = re.sub(r'概要[::]', '', text)
    
    # 連続する句読点を削除
    text = re.sub(r'[、。]{2,}', '。', text)
    
    # 空白の正規化
    text = re.sub(r'\s+', '', text)
    
    # 先頭・末尾の句読点を削除
    text = text.strip('、。:：')
    
    # 文として成立していない短すぎる出力を検出
    if len(text) < 10 or text.count('。') == 0:
        return None
    
    return text.strip()

def generate_summary(text, max_length=128, min_length=30, num_beams=4):
    """
    テキストから生成型要約を作成（改良版）
    
    Parameters:
    - text: 入力テキスト
    - max_length: 生成する要約の最大長（トークン数）
    - min_length: 生成する要約の最小長（トークン数）
    - num_beams: ビームサーチの幅
    """
    
    # 入力テキストの前処理
    text = text.strip()
    
    # T5用のタスクプレフィックス
    # 注: モデルによってはプレフィックスなしの方が良い場合もある
    input_text = text  # プレフィックスを削除してテスト
    
    # トークナイズ
    inputs = tokenizer(
        input_text,
        max_length=512,
        truncation=True,
        padding='longest',
        return_tensors='pt'
    )
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 要約生成（パラメータ調整）
    with torch.no_grad():
        output_ids = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            no_repeat_ngram_size=3,  # 2→3に変更
            early_stopping=True,
            length_penalty=2.0,  # 1.0→2.0に変更（長めの出力を促す）
            repetition_penalty=1.5,  # 繰り返しを抑制
            do_sample=False,  # 決定的な出力
        )
    
    # デコード
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # クリーニング
    cleaned_summary = clean_summary(summary)
    
    return cleaned_summary if cleaned_summary else summary.strip()

# UIコンポーネントの作成
input_text = widgets.Textarea(
    value='',
    placeholder='要約したいテキストをここに入力してください...\n生成型モデルが内容を理解して新しい文章を作成します。',
    description='入力:',
    disabled=False,
    layout=widgets.Layout(width='95%', height='200px')
)

summarize_button = widgets.Button(
    description='要約生成',
    button_style='primary',
    tooltip='クリックして新しい要約文を生成',
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

# 要約の長さ設定（デフォルト値を調整）
max_length_slider = widgets.IntSlider(
    value=128,  # 100→128
    min=50,     # 30→50
    max=256,    # 200→256
    step=16,    # 10→16
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

# ビームサーチの設定
num_beams_slider = widgets.IntSlider(
    value=4,
    min=2,      # 1→2
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
    
    # モデルが読み込まれているかチェック
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
        status_label.value = '<p style="color: orange;">⚠ より長いテキスト(50文字以上)を入力すると、より良い要約が生成されます。</p>'
        return
    
    try:
        status_label.value = '<p style="color: blue;">⏳ 要約を生成中... (数秒かかります)</p>'
        
        # パラメータの取得と調整
        max_len = max_length_slider.value
        min_len = min(min_length_slider.value, max_len - 20)
        num_beams = max(2, num_beams_slider.value)  # 最小2
        
        # 要約生成
        summary = generate_summary(
            text,
            max_length=max_len,
            min_length=min_len,
            num_beams=num_beams
        )
        
        if not summary or len(summary) < 10:
            status_label.value = '<p style="color: red;">⚠ 要約生成に失敗しました。パラメータを調整してもう一度お試しください。</p>'
            output_text.value = '(生成失敗)'
            return
        
        output_text.value = summary
        
        # 統計情報を表示
        char_reduction = round((1 - len(summary) / len(text)) * 100, 1)
        status_label.value = f'''
        <p style="color: green;">✓ 要約が生成されました!</p>
        <p style="color: #666; font-size: 12px;">
        入力: {len(text)}文字 → 出力: {len(summary)}文字 | 圧縮率: {char_reduction}%
        </p>
        '''
        
    except Exception as e:
        status_label.value = f'<p style="color: red;">❌ エラーが発生しました: {str(e)}</p>'
        output_text.value = ''
        import traceback
        print(traceback.format_exc())

# ボタンにイベントハンドラを設定
summarize_button.on_click(on_summarize_click)

# UIの表示
display(HTML("""
<h2>🤖 生成型テキスト要約システム v2 (改良版)</h2>
<p><strong>特徴:</strong> AIが内容を理解して新しい要約文を生成します</p>
<p><strong>改善点:</strong> 出力のクリーニング処理を追加、パラメータを最適化</p>
<hr>
"""))

display(input_text)
display(HTML("<h4>⚙️ 生成設定</h4>"))

display(HTML("<p style='font-weight: bold; margin-bottom: 5px;'>要約の長さ（トークン数）</p>"))
display(max_length_slider)
display(min_length_slider)
display(HTML("<p style='color: #666; font-size: 12px; margin-top: -10px;'>※ 推奨: 最大128、最小30から開始</p>"))

display(HTML("<p style='font-weight: bold; margin-bottom: 5px; margin-top: 15px;'>生成品質</p>"))
display(num_beams_slider)
display(HTML("<p style='color: #666; font-size: 12px; margin-top: -10px;'>※ 推奨: 4-6 (高いほど品質向上)</p>"))

display(summarize_button)
display(status_label)
display(HTML("<hr>"))
display(output_text)

print("\n" + "="*60)
print("✓ セットアップ完了！生成型要約システムv2が起動しました")
print("="*60)
print("\n💡 改善点:")
print("- 出力のクリーニング処理を追加")
print("- 繰り返しを抑制するパラメータを調整")
print("- より安定した生成を実現")
print("\n🎯 推奨設定 (280文字程度の文章の場合):")
print("- 最大長: 128")
print("- 最小長: 30")
print("- 品質: 4-6")