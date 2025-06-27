import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_instruction_model(model_name="Qwen/Qwen2.5-7B-Instruct"):
    """Load an instruction-tuned model for zero-shot classification"""
    
    print(f"Loading {model_name}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=True,
            cache_dir="/data/resource/huggingface/hub"
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            local_files_only=True,
            cache_dir="/data/resource/huggingface/hub",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        print(f"✓ {model_name} loaded successfully!")
        return tokenizer, model
        
    except Exception as e:
        print(f"✗ Failed to load {model_name}: {e}")
        return None, None
