

!pip install gradio transformers torch -q

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import gradio as gr


model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)


def chat(message, history):
    inputs = tokenizer.encode(message, return_tensors="pt")
    outputs = model.generate(inputs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


interface = gr.ChatInterface(
    fn=chat,
    title="GPT-2 ChatBot",
    description="Pure_GPT-2"
)

interface.launch()