import gradio as gr
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

print("Loading expanded fine-tuned model...")

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-100-final")
model = GPT2LMHeadModel.from_pretrained("./gpt2-100-final")
model.to(device)
model.eval()

print("Model loaded successfully!")

def chat(message, history):
    prompt = f"\n\nHuman: {message}\n\nAssistant:"
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=len(inputs[0]) + 40,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.5,
            no_repeat_ngram_size=3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "Assistant:" in full_response:
        response = full_response.split("Assistant:")[-1].strip()
    else:
        response = full_response[len(prompt):].strip()
    
    response = response.split('\n')[0].strip()
    response = response.split('Human:')[0].strip()
    
    if not response or len(response) < 3:
        response = "I'm here to help! What would you like to know?"
    
    return response

interface = gr.ChatInterface(
    fn=chat,
    title="GPT-2 Fine-tuned Chatbot (100 conversations)",
    description="GPT-2 trained on 100 high-quality conversations",
    examples=[
        "Hello, how are you?",
        "Tell me a joke",
        "Tell me something interesting",
        "What can you do?",
        "Good morning"
    ],
)

interface.launch(share=True)