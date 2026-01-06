def format_prompt(question, answer=None):
    """
    Mistral用プロンプト整形（日本語出力強化版）
    
    Exp-005の変更点:
    - システムメッセージを追加して日本語出力を明示
    """
    # 日本語出力を明示的に指示
    system_msg = "あなたは日本語で回答するAIアシスタントです。必ず日本語で答えてください。"
    
    prompt = f"<s>[INST] {system_msg}\n\n質問: {question} [/INST]"
    if answer:
        prompt += f" {answer} </s>"
    return prompt

def generate_prompt_for_training(data_point):
    """学習用プロンプト生成"""
    return format_prompt(data_point["question"], data_point["answer"])