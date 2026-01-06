# src/lora_setup.py
from peft import LoraConfig, get_peft_model

def apply_lora(model, r=8, lora_alpha=32, target_modules=None, lora_dropout=0.1):
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print("✅ LoRA applied")
    return model
