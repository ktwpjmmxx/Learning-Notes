def format_prompt(question, answer=None):
    """Mistral用プロンプト整形"""
    prompt = f"<s>[INST] {question} [/INST]"
    if answer:
        prompt += f" {answer} </s>"
    return prompt

def generate_prompt_for_training(data_point):
    return format_prompt(data_point["question"], data_point["answer"])