# src/chatbot.py
import torch, gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import PeftModel

def cleanup_vram():
    """GPUメモリ解放"""
    for var in ["model", "base_model", "trainer", "pipe"]:
        try:
            del globals()[var]
        except KeyError:
            pass
    torch.cuda.empty_cache()
    gc.collect()
    print("✅ VRAM cleaned up")

def load_inference_pipeline(base_model_id="mistralai/Mistral-7B-v0.1", adapter_dir="mistral7b_finetuned", device=0):
    """ベースモデル+LoRAアダプターをロードして推論パイプライン作成"""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    print("➡️ Loading Base Model (4-bit)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map={ '': device },
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token

    print("➡️ Applying LoRA Adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_dir)

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map={ '': device })
    print("✅ Pipeline ready")
    return pipe

def chat(pipe, prompt, max_new_tokens=100, temperature=0.7):
    """推論を実行"""
    result = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature)
    return result[0]["generated_text"]
