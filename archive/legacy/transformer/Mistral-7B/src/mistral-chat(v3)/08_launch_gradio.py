"""
08_launch_gradio.py
- モデル自動ロード
- 会話履歴保持
- パラメータ調整
- プリセット機能
- 評価メトリクス表示
"""

import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import gradio as gr
import json
import os
from datetime import datetime

# utils読み込み
from utils import cleanup_memory, save_json, print_gpu_memory
from utils_v3 import check_response_quality, calculate_keyword_coverage

# ==========================================
# 設定
# ==========================================

LOCAL_ADAPTER_DIR = "mistral7b_finetuned"
BASE_MODEL_ID = "mistralai/Mistral-7B-v0.1"
CHAT_LOGS_DIR = "results/chat_logs"

# ディレクトリ作成
os.makedirs(CHAT_LOGS_DIR, exist_ok=True)

# ==========================================
# 1. モデルのロード
# ==========================================

print("="*60)
print("🤖 Mistral-7B Fine-tuned Chatbot v3")
print("="*60)

# メモリクリーンアップ
cleanup_memory()

# 4bit量子化設定
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print("\n📥 Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

print("📥 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

print("📥 Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, LOCAL_ADAPTER_DIR)

print("\n✅ Model loaded successfully!\n")
print_gpu_memory()

# チャットログ用の変数
current_session = {
    "session_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
    "messages": []
}

# ==========================================
# 2. チャット関数
# ==========================================

def chat_with_model(message, history, temperature, max_tokens, repetition_penalty):
    """
    会話履歴を考慮してモデルから応答を生成
    
    Args:
        message: 現在のユーザーメッセージ
        history: 過去の会話履歴 [(user, bot), ...]
        temperature: 生成温度
        max_tokens: 最大トークン数
        repetition_penalty: 繰り返しペナルティ
    
    Returns:
        str: ボットの応答
    """
    # 会話履歴を含むプロンプト構築
    chat_prompt = ""
    for user_msg, bot_msg in history:
        chat_prompt += f"User: {user_msg}\nBot: {bot_msg}\n"
    chat_prompt += f"User: {message}\nBot:"
    
    # トークン化
    inputs = tokenizer(chat_prompt, return_tensors="pt").to("cuda")
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # デコード
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    bot_response = result.split("Bot:")[-1].strip()
    
    # クリーンアップ
    if "User:" in bot_response:
        bot_response = bot_response.split("User:")[0].strip()
    if "\n" in bot_response:
        bot_response = bot_response.split("\n")[0].strip()
    
    # チャットログに記録
    current_session["messages"].append({
        "user": message,
        "bot": bot_response,
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "repetition_penalty": repetition_penalty
        }
    })
    
    return bot_response


def evaluate_response(response: str) -> Dict[str, any]:
    """応答を評価してメトリクスを返す"""
    quality = check_response_quality(response)
    return {
        "quality_score": quality["score"],
        "word_count": quality["word_count"],
        "checks": quality["checks"]
    }

# ==========================================
# 3. Gradio UI構築
# ==========================================

# カスタムCSS
custom_css = """
.gradio-container {
    max-width: 1400px !important;
}
#chatbot {
    height: 550px;
}
.metric-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 15px;
    border-radius: 10px;
    color: white;
    text-align: center;
    font-weight: bold;
}
.parameter-box {
    background-color: #f8f9fa;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #dee2e6;
}
"""

with gr.Blocks(title="Mistral-7B v3 Chatbot", css=custom_css, theme=gr.themes.Soft()) as demo:
    
    # ヘッダー
    gr.Markdown("""
    # 🤖 Mistral-7B Fine-tuned Chatbot v3
    
    **最新版の改善点:**
    - ✅ 会話履歴を保持してコンテキスト理解
    - ✅ パラメータをリアルタイムで調整可能
    - ✅ 応答品質の自動評価
    - ✅ プリセット機能で簡単切り替え
    - ✅ チャットログの自動保存
    """)
    
    with gr.Row():
        # 左側: チャットエリア
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                height=550,
                label="💬 会話履歴",
                elem_id="chatbot",
                show_copy_button=True,
                avatar_images=(None, "https://em-content.zobj.net/source/twitter/376/robot_1f916.png")
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    label="メッセージ",
                    placeholder="メッセージを入力してEnterキーまたは送信ボタン...",
                    lines=2,
                    scale=4,
                    show_label=False
                )
                submit = gr.Button("📤 送信", variant="primary", scale=1)
            
            with gr.Row():
                clear = gr.Button("🗑️ 会話をクリア", variant="secondary", scale=1)
                retry = gr.Button("🔄 再生成", variant="secondary", scale=1)
                save_chat = gr.Button("💾 チャット保存", variant="secondary", scale=1)
            
            # 応答メトリクス表示
            with gr.Row():
                quality_display = gr.Textbox(
                    label="📊 最新応答の品質スコア",
                    value="待機中...",
                    interactive=False,
                    scale=1
                )
                length_display = gr.Textbox(
                    label="📏 応答長",
                    value="-",
                    interactive=False,
                    scale=1
                )
        
        # 右側: パラメータ調整
        with gr.Column(scale=1):
            with gr.Group(elem_classes="parameter-box"):
                gr.Markdown("### ⚙️ 生成パラメータ")
                
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.7,
                    step=0.1,
                    label="🌡️ Temperature（創造性）",
                    info="高いほど創造的、低いほど保守的"
                )
                
                max_tokens = gr.Slider(
                    minimum=50,
                    maximum=500,
                    value=200,
                    step=50,
                    label="📏 最大トークン数",
                    info="応答の最大長"
                )
                
                repetition_penalty = gr.Slider(
                    minimum=1.0,
                    maximum=2.0,
                    value=1.2,
                    step=0.1,
                    label="🔁 繰り返しペナルティ",
                    info="高いほど繰り返しを避ける"
                )
            
            # プリセット
            with gr.Group():
                gr.Markdown("### 🎛️ プリセット設定")
                
                with gr.Row():
                    preset_creative = gr.Button("✨ 創造的", size="sm")
                    preset_balanced = gr.Button("⚖️ バランス", size="sm")
                
                with gr.Row():
                    preset_precise = gr.Button("🎯 正確", size="sm")
                    preset_concise = gr.Button("📝 簡潔", size="sm")
            
            # モデル情報
            with gr.Group():
                gr.Markdown("### 📊 モデル情報")
                gr.Markdown(f"""
                **v3 仕様:**
                - ベースモデル: Mistral-7B-v0.1
                - LoRA Rank: 16
                - Target Modules: 4種類
                - 量子化: 4bit NF4
                - 学習データ: 100件（評価分割済み）
                
                **セッションID:**  
                `{current_session['session_id']}`
                """)
            
            # サンプル質問
            with gr.Group():
                gr.Markdown("### 📝 サンプル質問")
                
                examples = gr.Examples(
                    examples=[
                        "こんにちは！元気ですか？",
                        "機械学習とディープラーニングの違いは？",
                        "Pythonでリスト内包表記を使う方法は？",
                        "過学習を防ぐにはどうすればいい？",
                        "PyTorchとTensorFlowの違いを教えて",
                        "AIの勉強を始めたいです。何から始めるべき？",
                        "Hello! What is machine learning?",
                        "Explain the Transformer architecture"
                    ],
                    inputs=msg,
                    label=None
                )
    
    # フッター
    gr.Markdown("""
    ---
    💡 **使い方のヒント:**
    - 会話履歴が自動保存され、文脈を理解した応答が可能です
    - パラメータを調整して応答スタイルをカスタマイズできます
    - プリセットボタンで素早く設定を切り替えられます
    - 「再生成」で同じ質問に対する別の応答を試せます
    - 「チャット保存」で会話を記録できます
    
    ⚡ **パフォーマンス:** GPU T4使用時、応答生成は約2-3秒
    """)
    
    # ==========================================
    # 4. イベントハンドラー
    # ==========================================
    
    def respond(message, chat_history, temp, max_tok, rep_pen):
        """メッセージに応答し、メトリクスを更新"""
        bot_message = chat_with_model(message, chat_history, temp, max_tok, rep_pen)
        chat_history.append((message, bot_message))
        
        # メトリクス計算
        metrics = evaluate_response(bot_message)
        quality_text = f"品質: {metrics['quality_score']:.1%}"
        length_text = f"{metrics['word_count']} words"
        
        return "", chat_history, quality_text, length_text
    
    def retry_last(chat_history, temp, max_tok, rep_pen):
        """最後のメッセージを再生成"""
        if not chat_history:
            return chat_history, "待機中...", "-"
        
        last_user_msg = chat_history[-1][0]
        chat_history = chat_history[:-1]
        
        bot_message = chat_with_model(last_user_msg, chat_history, temp, max_tok, rep_pen)
        chat_history.append((last_user_msg, bot_message))
        
        # メトリクス更新
        metrics = evaluate_response(bot_message)
        quality_text = f"品質: {metrics['quality_score']:.1%}"
        length_text = f"{metrics['word_count']} words"
        
        return chat_history, quality_text, length_text
    
    def save_chat_log(chat_history):
        """チャットログを保存"""
        if not chat_history:
            return "⚠️ 保存するチャットがありません"
        
        log_file = f"{CHAT_LOGS_DIR}/chat_{current_session['session_id']}.json"
        save_json(current_session, log_file)
        
        return f"✅ 保存完了: {log_file}"
    
    # プリセット関数
    def set_preset_creative():
        return 1.2, 300, 1.1
    
    def set_preset_balanced():
        return 0.7, 200, 1.2
    
    def set_preset_precise():
        return 0.3, 150, 1.5
    
    def set_preset_concise():
        return 0.5, 100, 1.3
    
    # イベント接続
    msg.submit(
        respond,
        [msg, chatbot, temperature, max_tokens, repetition_penalty],
        [msg, chatbot, quality_display, length_display]
    )
    
    submit.click(
        respond,
        [msg, chatbot, temperature, max_tokens, repetition_penalty],
        [msg, chatbot, quality_display, length_display]
    )
    
    clear.click(
        lambda: (None, "待機中...", "-"),
        None,
        [chatbot, quality_display, length_display],
        queue=False
    )
    
    retry.click(
        retry_last,
        [chatbot, temperature, max_tokens, repetition_penalty],
        [chatbot, quality_display, length_display]
    )
    
    save_chat.click(
        save_chat_log,
        chatbot,
        quality_display
    )
    
    # プリセット
    preset_creative.click(set_preset_creative, None, [temperature, max_tokens, repetition_penalty])
    preset_balanced.click(set_preset_balanced, None, [temperature, max_tokens, repetition_penalty])
    preset_precise.click(set_preset_precise, None, [temperature, max_tokens, repetition_penalty])
    preset_concise.click(set_preset_concise, None, [temperature, max_tokens, repetition_penalty])

# ==========================================
# 5. 起動
# ==========================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Starting Gradio UI v3...")
    print("="*60)
    print("\n✨ Features:")
    print("  - ✅ 会話履歴保持")
    print("  - ✅ パラメータ調整")
    print("  - ✅ プリセット切り替え")
    print("  - ✅ 再生成機能")
    print("  - ✅ 品質メトリクス表示")
    print("  - ✅ チャットログ保存")
    print("\n📌 Access URLs:")
    print("  Local:  http://localhost:7860")
    print("  Public: (generated on launch)")
    print("\n💡 Tips:")
    print("  - Ctrl+C で終了")
    print("  - share=False に変更すると公開URLなし")
    print("\n")
    
    # UIの起動
    demo.launch(
        share=True,  # 公開URL生成
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        debug=True,
        show_api=False
    )