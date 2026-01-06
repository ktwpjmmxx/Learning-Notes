import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import config

def get_model_and_tokenizer(inference_mode=False):
    bnb_config = BitsAndBytesConfig(**config.BNB_CONFIG)
    
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME, quantization_config=bnb_config, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if not inference_mode:
        model = prepare_model_for_kbit_training(model)
        peft_config = LoraConfig(
            lora_alpha=config.LORA_ALPHA, lora_dropout=config.LORA_DROPOUT,
            r=config.LORA_R, bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"]
        )
        model = get_peft_model(model, peft_config)
        return model, tokenizer, peft_config
    
    return model, tokenizer