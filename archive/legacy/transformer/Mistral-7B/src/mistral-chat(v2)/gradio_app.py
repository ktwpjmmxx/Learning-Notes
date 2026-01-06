# src/gradio_app.py
import gradio as gr
from chatbot import chat, load_inference_pipeline

pipe = load_inference_pipeline()

def chat_with_model(chat_prompt_text):
    response = chat(pipe, chat_prompt_text, max_new_tokens=256, temperature=0.8)
    if "User:" in response:
        response = response.split("User:")[0].strip()
    return response

def add_message(message, history):
    return "", history + [[message, None]]

def generate_response(history):
    chat_prompt = ""
    for user_msg, bot_resp in history:
        if bot_resp is None:
            chat_prompt += f"User: {user_msg}\nBot:"
        else:
            chat_prompt += f"User: {user_msg}\nBot: {bot_resp}\n"
    bot_response = chat_with_model(chat_prompt)
    history[-1][1] = bot_response
    return history

custom_theme = gr.themes.Soft(font=["sans-serif"]).set(body_background_fill='white', button_primary_background_fill_hover='*primary_300')
custom_css = ".submit-button button { margin-top:10px !important; }"

with gr.Blocks(theme=custom_theme, title="Fine-tuned Mistral Chatbot", css=custom_css) as demo:
    gr.Markdown("# 📚 LoRA Fine-tuned Mistral Chatbot")
    chatbot_ui = gr.Chatbot(height=500, label="チャット履歴")
    with gr.Row():
        txt = gr.Textbox(show_label=False, placeholder="メッセージを入力...", lines=3, scale=4)
        with gr.Column(scale=1, elem_classes="submit-button"):
            submit_btn = gr.Button("送信", variant="primary")
    clear_btn = gr.ClearButton([txt, chatbot_ui])

    txt_msg = txt.submit(fn=add_message, inputs=[txt, chatbot_ui], outputs=[txt, chatbot_ui], queue=False)
    btn_msg = submit_btn.click(fn=add_message, inputs=[txt, chatbot_ui], outputs=[txt, chatbot_ui], queue=False)
    txt_msg.then(fn=generate_response, inputs=[chatbot_ui], outputs=[chatbot_ui], queue=False)
    btn_msg.then(fn=generate_response, inputs=[chatbot_ui], outputs=[chatbot_ui], queue=False)

if __name__ == "__main__":
    demo.launch(share=True)
