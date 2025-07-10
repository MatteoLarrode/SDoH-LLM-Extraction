import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, PeftModel

# Quantization config (4-bit)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

def load_lora_llama(base_model_path: str, adapter_path: str = None, cache_dir: str = None, device: int = 0):
    # Load base model in 4-bit
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        cache_dir=cache_dir,
        quantization_config=bnb_config,
        device_map={"": device},
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, cache_dir=cache_dir, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    if adapter_path:
        print(f"🔗 Loading LoRA adapters from: {adapter_path}")
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        print(f"🧪 LoRA training setup — no adapter_path provided.")
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(base_model, lora_config)
        model.print_trainable_parameters()

    return model, tokenizer