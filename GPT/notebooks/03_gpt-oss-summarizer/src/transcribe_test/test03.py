# Google Colab 抽出型テキスト要約システム v4
# 日本語対応 - 元の文章から重要な文を抽出して要約

# 必要なライブラリのインストール
!pip install -q transformers torch sentencepiece fugashi ipadic

import torch
from transformers import BertJapaneseTokenizer, BertModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import ipywidgets as widgets
from IPython.display import display, HTML
import warnings
warnings.filterwarnings('ignore')

print("ライブラリのインポート完了!")

# 日本語BERTモデルの初期化
print("モデルを読み込んでいます...")

try:
    model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"
    tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    print(f"✓ モデルの読み込み完了! (デバイス: {device})")
    print("✓ 日本語BERTモデル使用 - 抽出型要約")
except Exception as e:
    print(f"エラー: {e}")

# 文を分割する関数
def split_sentences(text):
    """テキストを文単位で分割"""
    import re
    # 句点で分割（。！？）
    sentences = re.split('[。！？]', text)
    sentences = [s.strip() + '。' for s in sentences if s.strip()]
    return sentences

# 文の重要度を計算する関数
def get_sentence_embeddings(sentences):
    """各文のベクトル表現を取得"""
    embeddings = []
    
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors='pt', 
                          max_length=128, truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            # [CLS]トークンの埋め込みを使用
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(embedding[0])
    
    return np.array(embeddings)

# 重要な文を抽出する関数
def extract_summary(text, num_sentences=3):
    """重要な文を抽出して要約を生成"""
    
    # 文に分割
    sentences = split_sentences(text)
    
    if len(sentences) <= num_sentences:
        return ''.join(sentences)
    
    # 各文の埋め込みを取得
    embeddings = get_sentence_embeddings(sentences)
    
    # 文書全体の埋め込み（平均）
    doc_embedding = np.mean(embeddings, axis=0).reshape(1, -1)
    
    # 各文と文書全体の類似度を計算
    similarities = cosine_similarity(embeddings, doc_embedding).flatten()
    
    # 類似度が高い順にソート
    ranked_indices = np.argsort(similarities)[::-1]
    
    # 上位N文を元の順序で選択
    selected_indices = sorted(ranked_indices[:num_sentences])
    
    # 選択された文を結合
    summary = ''.join([sentences[i] for i in selected_indices])
    
    return summary

# UIコンポーネントの作成
input_text = widgets.Textarea(
    value='',
    placeholder='要約したいテキストをここに入力してください...\n（複数の文を含む長めのテキストを推奨）',
    description='入力:',
    disabled=False,
    layout=widgets.Layout(width='95%', height='200px')
)

summarize_button = widgets.Button(
    description='要約する',
    button_style='success',
    tooltip='クリックして重要な文を抽出',
    icon='check',
    layout=widgets.Layout(width='200px', height='40px')
)

output_text = widgets.Textarea(
    value='',
    placeholder='抽出された重要な文がここに表示されます...',
    description='要約:',
    disabled=True,
    layout=widgets.Layout(width='95%', height='150px')
)

status_label = widgets.HTML(
    value='<p style="color: #666;">テキストを入力して「要約する」ボタンをクリックしてください。</p>'
)

# 抽出する文の数を設定
num_sentences_slider = widgets.IntSlider(
    value=3,
    min=1,
    max=10,
    step=1,
    description='抽出文数:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d'
)

# 要約処理の関数
def on_summarize_click(b):
    text = input_text.value.strip()
    
    if not text:
        status_label.value = '<p style="color: red;">⚠ テキストを入力してください。</p>'
        output_text.value = ''
        return
    
    # 文に分割してチェック
    sentences = split_sentences(text)
    
    if len(sentences) < 2:
        status_label.value = '<p style="color: orange;">⚠ 複数の文を含むテキストを入力してください。</p>'
        output_text.value = ''
        return
    
    try:
        status_label.value = '<p style="color: blue;">⏳ 重要な文を抽出中...</p>'
        
        # 抽出文数を調整
        num_sentences = min(num_sentences_slider.value, len(sentences))
        
        # 要約生成
        summary = extract_summary(text, num_sentences=num_sentences)
        
        output_text.value = summary
        
        # 統計情報を表示
        char_reduction = round((1 - len(summary) / len(text)) * 100, 1)
        status_label.value = f'''
        <p style="color: green;">✓ 要約が完成しました!</p>
        <p style="color: #666; font-size: 12px;">
        入力: {len(sentences)}文 ({len(text)}字) → 出力: {num_sentences}文 ({len(summary)}字) | 圧縮率: {char_reduction}%
        </p>
        '''
        
    except Exception as e:
        status_label.value = f'<p style="color: red;">エラーが発生しました: {str(e)}</p>'
        output_text.value = ''
        import traceback
        print(traceback.format_exc())

# ボタンにイベントハンドラを設定
summarize_button.on_click(on_summarize_click)

# UIの表示
display(HTML("""
<h2>📝 抽出型テキスト要約システム v4</h2>
<p><strong>特徴:</strong> 元の文章から重要な文を抽出します（内容の改変なし）</p>
<hr>
"""))

display(input_text)
display(HTML("<h4>⚙️ 設定</h4>"))
display(num_sentences_slider)
display(HTML("<p style='color: #666; font-size: 12px;'>※ 入力テキストの文数より多く設定すると、全ての文が出力されます</p>"))
display(summarize_button)
display(status_label)
display(HTML("<hr>"))
display(output_text)

print("\n" + "="*60)
print("✓ セットアップ完了！抽出型要約システムが起動しました")
print("="*60)
print("\n💡 ヒント:")
print("- 複数の文を含む長めのテキストを入力してください")
print("- 抽出文数を調整して、要約の長さをコントロールできます")
print("- 元の文をそのまま使うので、内容の改変はありません")