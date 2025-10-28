# Google Colab ハイブリッド型テキスト要約システム v1

!pip install -q transformers torch sentencepiece fugashi ipadic unidic-lite scikit-learn

import torch
from transformers import BertJapaneseTokenizer, BertModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import ipywidgets as widgets
from IPython.display import display, HTML
import warnings
import re
warnings.filterwarnings('ignore')

print("ライブラリのインポート完了!")

print("モデルを読み込んでいます...")

# グローバル変数として定義
tokenizer = None
model = None
device = None

try:
    model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"
    tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    print(f"✓ モデルの読み込み完了! (デバイス: {device})")
    print("✓ 日本語BERTモデル使用 - ハイブリッド型要約")
except Exception as e:
    print(f"❌ エラー: {e}")
    import traceback
    print(traceback.format_exc())

# 文を分割する関数
def split_sentences(text):
    """テキストを文単位で分割"""
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
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(embedding[0])
    
    return np.array(embeddings)

# 接続詞を追加する関数
def add_connectors(sentences, positions):
    """
    文と文を自然に繋ぐ接続詞を追加
    """
    result = []
    
    for i, (sentence, pos) in enumerate(zip(sentences, positions)):
        # 最初の文はそのまま
        if i == 0:
            result.append(sentence)
            continue
        
        # 前の文との関係性で接続詞を選択
        prev_pos = positions[i-1]
        
        # 文章の前半から後半への移行
        if prev_pos < len(sentences) / 3 and pos > len(sentences) * 2/3:
            # 大きく飛んだ場合
            connector = "一方、"
        # 近い位置の文同士
        elif abs(pos - prev_pos) <= 2:
            connector = "また、"
        # 最後の文
        elif i == len(sentences) - 1:
            connector = "そして、"
        else:
            connector = ""
        
        # 既に接続詞で始まっている場合は追加しない
        if re.match(r'^(そして|また|さらに|一方|しかし|ただし)', sentence):
            connector = ""
        
        result.append(connector + sentence)
    
    return result

# 冗長表現を削除する関数
def remove_redundancy(text):
    """
    冗長な表現を簡潔にする
    """
    # 重複する接続詞を削除
    text = re.sub(r'(しかし、)+', 'しかし、', text)
    text = re.sub(r'(また、)+', 'また、', text)
    text = re.sub(r'(そして、)+', 'そして、', text)
    
    # 「〜ています。〜ています。」の連続を避ける
    # (高度な処理のため、今回は基本的なもののみ)
    
    return text

# 文を簡潔化する関数
def simplify_sentence(sentence):
    """
    文を簡潔にする(基本的な処理)
    """
    # 「〜することができます」→「〜できます」
    sentence = re.sub(r'することができ', 'でき', sentence)
    
    # 「〜という」の削減
    sentence = re.sub(r'という(こと|もの)', '', sentence)
    
    return sentence

# ハイブリッド要約の生成
def hybrid_summarize(text, num_sentences=3, style='natural'):
    """
    ハイブリッド型要約を生成
    
    Parameters:
    - text: 入力テキスト
    - num_sentences: 抽出する文の数
    - style: 'natural'(自然な文章) or 'bullet'(箇条書き)
    """
    
    # Step 1: 文に分割
    sentences = split_sentences(text)
    
    if len(sentences) <= num_sentences:
        return ''.join(sentences)
    
    # Step 2: 各文の埋め込みを取得
    embeddings = get_sentence_embeddings(sentences)
    
    # Step 3: 文書全体の埋め込み
    doc_embedding = np.mean(embeddings, axis=0).reshape(1, -1)
    
    # Step 4: 各文と文書全体の類似度を計算
    similarities = cosine_similarity(embeddings, doc_embedding).flatten()
    
    # Step 5: 重要度スコアを計算
    scores = similarities.copy()
    
    # 位置ボーナス
    if len(sentences) > 0:
        scores[0] *= 1.2  # 第1文
    if len(sentences) > 1:
        scores[-1] *= 1.15  # 最終文
    
    # キーワードボーナス
    important_keywords = [
        'しかし', '一方', 'また', 'さらに', '今後', '将来',
        '課題', '重要', '問題', '必要', '求められ'
    ]
    
    for i, sentence in enumerate(sentences):
        for keyword in important_keywords:
            if keyword in sentence:
                scores[i] *= 1.1
                break
    
    # 文の長さ考慮
    for i, sentence in enumerate(sentences):
        if len(sentence) < 15:
            scores[i] *= 0.8
    
    # Step 6: スコアが高い順にソート
    ranked_indices = np.argsort(scores)[::-1]
    
    # Step 7: 上位N文を元の順序で選択
    selected_indices = sorted(ranked_indices[:num_sentences])
    selected_sentences = [sentences[i] for i in selected_indices]
    
    # Step 8: スタイルに応じて整形
    if style == 'bullet':
        # 箇条書き形式
        result = '\n'.join([f"• {s}" for s in selected_sentences])
    else:
        # 自然な文章形式
        # 接続詞を追加
        connected_sentences = add_connectors(selected_sentences, selected_indices)
        
        # 簡潔化
        simplified = [simplify_sentence(s) for s in connected_sentences]
        
        # 結合
        result = ''.join(simplified)
        
        # 冗長性削除
        result = remove_redundancy(result)
    
    return result

# UIコンポーネントの作成
input_text = widgets.Textarea(
    value='',
    placeholder='要約したいテキストをここに入力してください...',
    description='入力:',
    disabled=False,
    layout=widgets.Layout(width='95%', height='200px')
)

summarize_button = widgets.Button(
    description='要約生成',
    button_style='success',
    tooltip='ハイブリッド要約を生成',
    icon='magic',
    layout=widgets.Layout(width='200px', height='40px')
)

output_text = widgets.Textarea(
    value='',
    placeholder='生成された要約がここに表示されます...',
    description='要約:',
    disabled=True,
    layout=widgets.Layout(width='95%', height='150px')
)

status_label = widgets.HTML(
    value='<p style="color: #666;">テキストを入力して「要約生成」ボタンをクリックしてください。</p>'
)

# 設定スライダー
num_sentences_slider = widgets.IntSlider(
    value=3,
    min=2,
    max=10,
    step=1,
    description='文数:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d'
)

# スタイル選択
style_dropdown = widgets.Dropdown(
    options=[('自然な文章', 'natural'), ('箇条書き', 'bullet')],
    value='natural',
    description='スタイル:',
    disabled=False,
)

# 要約処理の関数
def on_summarize_click(b):
    global tokenizer, model, device
    
    if tokenizer is None or model is None:
        status_label.value = '<p style="color: red;">❌ モデルが読み込まれていません。</p>'
        output_text.value = ''
        return
    
    text = input_text.value.strip()
    
    if not text:
        status_label.value = '<p style="color: red;">⚠ テキストを入力してください。</p>'
        output_text.value = ''
        return
    
    sentences = split_sentences(text)
    
    if len(sentences) < 2:
        status_label.value = '<p style="color: orange;">⚠ 複数の文を含むテキストを入力してください。</p>'
        output_text.value = ''
        return
    
    try:
        status_label.value = '<p style="color: blue;">⏳ ハイブリッド要約を生成中...</p>'
        
        num_sentences = min(num_sentences_slider.value, len(sentences))
        style = style_dropdown.value
        
        # ハイブリッド要約生成
        summary = hybrid_summarize(text, num_sentences=num_sentences, style=style)
        
        output_text.value = summary
        
        char_reduction = round((1 - len(summary) / len(text)) * 100, 1)
        status_label.value = f'''
        <p style="color: green;">✓ ハイブリッド要約が完成しました!</p>
        <p style="color: #666; font-size: 12px;">
        入力: {len(sentences)}文 ({len(text)}字) → 出力: {num_sentences}文 ({len(summary)}字) | 圧縮率: {char_reduction}%
        </p>
        <p style="color: #666; font-size: 12px;">
        <strong>処理内容:</strong> 重要文抽出 → 接続詞追加 → 簡潔化 → 冗長性削除
        </p>
        '''
        
    except Exception as e:
        status_label.value = f'<p style="color: red;">エラー: {str(e)}</p>'
        output_text.value = ''
        import traceback
        print(traceback.format_exc())

summarize_button.on_click(on_summarize_click)

# UIの表示
display(HTML("""
<h2>🔀 ハイブリッド型テキスト要約システム v1</h2>
<p><strong>特徴:</strong> 抽出型の正確性 + 生成型の自然さを両立</p>
<p><strong>処理:</strong> 重要文抽出 → 接続詞追加 → 簡潔化 → 整形</p>
<hr>
"""))

display(input_text)
display(HTML("<h4>⚙️ 設定</h4>"))
display(num_sentences_slider)
display(style_dropdown)
display(HTML("""
<p style='color: #666; font-size: 12px;'>
<strong>自然な文章:</strong> 接続詞で繋いだ流暢な要約<br>
<strong>箇条書き:</strong> 重要なポイントを箇条書きで表示
</p>
"""))
display(summarize_button)
display(status_label)
display(HTML("<hr>"))
display(output_text)

print("\n" + "="*60)
print("✓ セットアップ完了！ハイブリッド型要約システムが起動しました")
print("="*60)
print("\n💡 ハイブリッド型の特徴:")
print("- ✅ 抽出型の信頼性(内容改変なし)")
print("- ✅ 自然な文章の流れ(接続詞追加)")
print("- ✅ 簡潔な表現(冗長性削除)")
print("- ✅ 高速処理(数秒で完了)")
print("- ✅ 完全無料(軽量モデル)")
print("\n🎯 推奨設定:")
print("- 文数: 3")
print("- スタイル: 自然な文章")