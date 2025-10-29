# Google Colab ハイブリッド型テキスト要約システム v2 (改良版)

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
    print("✓ 日本語BERTモデル使用 - テキスト要約")
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

# 文の対比関係を判定する関数
def has_contrast(prev_sentence, current_sentence):
    """
    2つの文に対比関係があるかを判定
    """
    # ネガティブキーワード
    negative_words = [
        '課題', '問題', '懸念', '欠点', 'デメリット', '困難', 
        '不足', '減少', '低下', '悪化', '負荷', '摩擦',
        '被害', '過度', '過剰', 'リスク', '危険'
    ]
    
    # ポジティブキーワード
    positive_words = [
        '効果', '利点', 'メリット', '向上', '改善', '進歩',
        '成長', '発展', '実現', '成功', '可能', '注目',
        '活性化', 'クリーン', '効率'
    ]
    
    # 前の文のセンチメント
    prev_positive = any(word in prev_sentence for word in positive_words)
    prev_negative = any(word in prev_sentence for word in negative_words)
    
    # 現在の文のセンチメント
    curr_positive = any(word in current_sentence for word in positive_words)
    curr_negative = any(word in current_sentence for word in negative_words)
    
    # 対比関係: ポジティブ→ネガティブ または ネガティブ→ポジティブ
    if (prev_positive and curr_negative) or (prev_negative and curr_positive):
        return True
    
    return False

# 接続詞を追加する関数(改良版v2.1)
def add_connectors(sentences, positions, original_sentences):
    """
    自然に繋ぐ接続詞を追加
    
    Parameters:
    - sentences: 選択された文のリスト
    - positions: 元の文章での位置
    - original_sentences: 元の全文のリスト
    """
    result = []
    used_connectors = []  # 使用した接続詞を記録
    
    for i, (sentence, pos) in enumerate(zip(sentences, positions)):
        # 最初の文はそのまま
        if i == 0:
            result.append(sentence)
            continue
        
        # 既に接続詞で始まっている場合は追加しない & 接続詞を記録
        existing_connector = re.match(r'^(そして|また|さらに|一方(で)?|しかし(ながら)?|ただし|加えて|他方)', sentence)
        if existing_connector:
            # 元の接続詞を記録(「一方で」→「一方」として記録)
            connector_text = existing_connector.group(1)
            # 「一方で」「一方」を統一
            if connector_text.startswith('一方'):
                used_connectors.append('一方')
            elif connector_text.startswith('しかし'):
                used_connectors.append('しかし')
            else:
                used_connectors.append(connector_text)
            
            result.append(sentence)
            continue
        
        prev_pos = positions[i-1]
        prev_sentence = sentences[i-1]
        
        # 接続詞の選択ロジック
        connector = ""
        
        # 1. 最後の文は「そして」で結論を示す
        if i == len(sentences) - 1:
            # 「今後」「将来」「課題」などの結論キーワードがあるか
            if any(word in sentence for word in ['今後', '将来', '課題', '求められ', '必要']):
                connector = "そして、"
            else:
                connector = "また、"
        
        # 2. 対比関係がある場合は「一方」「しかし」
        elif has_contrast(prev_sentence, sentence):
            # 既に「一方」または「しかし」が使われている場合は別の接続詞を使う
            if '一方' not in used_connectors and 'しかし' not in used_connectors:
                connector = "一方、"
            elif 'しかし' not in used_connectors:
                connector = "しかし、"
            elif 'ただし' not in used_connectors:
                connector = "ただし、"
            else:
                # 対比表現を使い切った場合は「また」に fallback
                connector = "また、"
        
        # 3. 近い位置の文(連続または1文飛ばし)
        elif abs(pos - prev_pos) <= 2:
            # 既に使った接続詞を避ける
            if 'また' not in used_connectors:
                connector = "また、"
            elif 'さらに' not in used_connectors:
                connector = "さらに、"
            elif '加えて' not in used_connectors:
                connector = "加えて、"
            else:
                connector = ""  # 接続詞なし
        
        # 4. 大きく飛んだ場合
        elif pos - prev_pos >= 3:
            if 'さらに' not in used_connectors:
                connector = "さらに、"
            else:
                connector = "また、"
        
        # 5. その他の場合
        else:
            if 'また' not in used_connectors:
                connector = "また、"
            else:
                connector = "さらに、"
        
        # 使用した接続詞を記録(「、」を除いて記録)
        if connector:
            used_connectors.append(connector.replace('、', ''))
        
        result.append(connector + sentence)
    
    return result

# 冗長表現を削除する関数(改良版v2.2)
def remove_redundancy(text):
    """
    冗長な表現を簡潔にする
    """
    # 同じ接続詞の連続を削除
    text = re.sub(r'(また、)+', 'また、', text)
    text = re.sub(r'(そして、)+', 'そして、', text)
    text = re.sub(r'(さらに、)+', 'さらに、', text)
    text = re.sub(r'(しかし、)+', 'しかし、', text)
    text = re.sub(r'(一方、)+', '一方、', text)
    
    # 「また、さらに、」→「さらに、」のような冗長な組み合わせを簡略化
    text = re.sub(r'また、さらに、', 'さらに、', text)
    text = re.sub(r'さらに、また、', 'さらに、', text)
    
    # 追加: 不自然な空白を削除
    text = re.sub(r'\s+', '', text)
    
    # 追加: 連続する「、」を整理
    text = re.sub(r'、+', '、', text)
    
    return text

# 最適な抽出文数を自動判定する関数
def determine_optimal_sentences(total_sentences):
    """
    原文の文数に応じて最適な抽出文数を自動判定
    
    Parameters:
    - total_sentences: 原文の総文数
    
    Returns:
    - optimal_num: 最適な抽出文数
    """
    if total_sentences <= 2:
        # 2文以下はそのまま
        return total_sentences
    elif total_sentences == 3:
        # 3文 → 2文抽出(約60%圧縮)
        return 2
    elif total_sentences == 4:
        # 4文 → 3文抽出(約50%圧縮)
        return 3
    elif total_sentences in [5, 6]:
        # 5-6文 → 3文抽出(約40-50%圧縮)
        return 3
    elif total_sentences in [7, 8, 9, 10]:
        # 7-10文 → 4-5文抽出(約40-50%圧縮)
        return max(4, int(total_sentences * 0.5))
    else:
        # 11文以上 → 約50%抽出
        return max(5, int(total_sentences * 0.5))

# 文を簡潔化する関数(修正版)
def simplify_sentence(sentence):
    """
    文を簡潔にする(修正版 - 文法エラーを防止)
    """
    original_length = len(sentence)
    
    # 1. 冗長な表現の簡略化(安全な変換のみ)
    sentence = re.sub(r'ということが', '', sentence)
    sentence = re.sub(r'という(こと|もの)', '', sentence)
    
    # 2. 副詞の削除(安全なもののみ)
    sentence = re.sub(r'大きく', '', sentence)
    sentence = re.sub(r'非常に', '', sentence)
    
    # 3. 文末表現の統一(慎重に)
    # 「〜が進んでいます」→「〜が進む」などは一旦保留(文法エラーのリスク)
    
    # 4. 不自然な空白を削除
    sentence = re.sub(r'\s+', '', sentence)
    
    # 5. 連続する句読点の整理
    sentence = re.sub(r'、+', '、', sentence)
    sentence = re.sub(r'。+', '。', sentence)
    
    return sentence

# ハイブリッド要約の生成(自動判定版)
def hybrid_summarize(text, style='natural'):
    """
    テキスト型要約を生成(自動判定)
    
    Parameters:
    - text: 入力テキスト
    - style: 'natural'(自然な文章) or 'bullet'(箇条書き)
    
    Returns:
    - summary: 要約文
    - num_sentences: 自動判定された抽出文数
    """
    
    # Step 1: 文に分割
    sentences = split_sentences(text)
    
    # Step 2: 最適な抽出文数を自動判定
    num_sentences = determine_optimal_sentences(len(sentences))
    
    if len(sentences) <= num_sentences:
        return ''.join(sentences), len(sentences)
    
    # Step 3: 各文の埋め込みを取得
    embeddings = get_sentence_embeddings(sentences)
    
    # Step 4: 文書全体の埋め込み
    doc_embedding = np.mean(embeddings, axis=0).reshape(1, -1)
    
    # Step 5: 各文と文書全体の類似度を計算
    similarities = cosine_similarity(embeddings, doc_embedding).flatten()
    
    # Step 6: 重要度スコアを計算
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
    
    # Step 7: スコアが高い順にソート
    ranked_indices = np.argsort(scores)[::-1]
    
    # Step 8: 上位N文を元の順序で選択
    selected_indices = sorted(ranked_indices[:num_sentences])
    selected_sentences = [sentences[i] for i in selected_indices]
    
    # Step 9: スタイルに応じて整形
    if style == 'bullet':
        # 箇条書き形式
        result = '\n'.join([f"• {s}" for s in selected_sentences])
    else:
        # 自然な文章形式
        # 接続詞を追加
        connected_sentences = add_connectors(selected_sentences, selected_indices, sentences)
        
        # 簡潔化
        simplified = [simplify_sentence(s) for s in connected_sentences]
        
        # 結合
        result = ''.join(simplified)
        
        # 冗長性削除
        result = remove_redundancy(result)
    
    return result, num_sentences

# UIコンポーネントの作成
input_text = widgets.Textarea(
    value='',
    placeholder='テキストをここに入力してください...',
    description='入力:',
    disabled=False,
    layout=widgets.Layout(width='95%', height='200px')
)

summarize_button = widgets.Button(
    description='要約生成',
    button_style='success',
    tooltip='要約を生成',
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
    value='<p style="color: #666;">テキストを入力して「要約生成」ボタンをクリックしてください。<br><small>💡 抽出する文数は自動で最適化されます</small></p>'
)

# スタイル選択のみ残す(文数スライダーは削除)
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
        status_label.value = '<p style="color: blue;">⏳ 要約を生成中...</p>'
        
        style = style_dropdown.value
        
        # ハイブリッド要約生成(文数は自動判定)
        summary, num_sentences = hybrid_summarize(text, style=style)
        
        output_text.value = summary
        
        char_reduction = round((1 - len(summary) / len(text)) * 100, 1)
        
        # 圧縮率の評価メッセージ
        compression_msg = ""
        if char_reduction >= 60:
            compression_msg = '<span style="color: green;">📊 優秀</span>'
        elif char_reduction >= 40:
            compression_msg = '<span style="color: green;">📊 標準</span>'
        elif char_reduction >= 30:
            compression_msg = '<span style="color: orange;">📊 良好</span>'
        elif char_reduction >= 20:
            compression_msg = '<span style="color: orange;">📊 やや低い</span>'
        else:
            compression_msg = '<span style="color: red;">📊 不足</span>'
        
        status_label.value = f'''
        <p style="color: green;">✓ 要約完成!</p>
        <p style="color: #666; font-size: 12px;">
        入力: {len(sentences)}文 ({len(text)}字) → 出力: {num_sentences}文 ({len(summary)}字) | 圧縮率: {char_reduction}% {compression_msg}
        </p>
        <p style="color: #666; font-size: 12px;">
        <strong>自動判定:</strong> {len(sentences)}文の原文から最適な{num_sentences}文を抽出しました
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
<h2>🔀 テキスト要約システム v2.3 (完全自動版)</h2>
<p><strong>特徴:</strong> 抽出文数を完全自動判定、設定不要で最適な要約を生成</p>
<p><strong>改良点:</strong> 原文の長さに応じて自動的に最適な圧縮率を実現</p>
<hr>
"""))

display(input_text)
display(HTML("<h4>⚙️ 設定</h4>"))
display(style_dropdown)
display(HTML("""
<p style='color: #666; font-size: 12px;'>
<strong>自然な文章:</strong> 接続詞で繋いだ流暢な要約<br>
<strong>箇条書き:</strong> 重要なポイントを箇条書きで表示<br>
<br>
<strong>💡 自動最適化:</strong><br>
• 3文の原文 → 2文抽出(約60%圧縮)<br>
• 4文の原文 → 3文抽出(約50%圧縮)<br>
• 5-6文の原文 → 3文抽出(約40-50%圧縮)<br>
• 7文以上の原文 → 約50%を抽出
</p>
"""))
display(summarize_button)
display(status_label)
display(HTML("<hr>"))
display(output_text)

print("\n" + "="*60)
print("✓ セットアップ完了。要約システムが起動しました")
print("="*60)
print("\n💡 v2.3の特徴:")
print("- ✅ 抽出文数を完全自動判定(設定不要)")
print("- ✅ 原文の長さに応じて最適な圧縮率を実現")
print("- ✅ 接続詞の重複防止機能を維持")
print("- ✅ シンプルで使いやすいUI")
print("\n🎯 自動判定ロジック:")
print("- 3文の原文 → 2文抽出(約60%圧縮)")
print("- 4文の原文 → 3文抽出(約50%圧縮)")
print("- 5-6文の原文 → 3文抽出(約40-50%圧縮)")
print("- 7-10文の原文 → 4-5文抽出(約40-50%圧縮)")
print("- 11文以上 → 約50%を抽出")
print("\n📊 圧縮率の評価:")
print("- 60%以上: 優秀")
print("- 40-60%: 標準(業界基準)")
print("- 30-40%: 良好")
print("- 20-30%: やや低い")