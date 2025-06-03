import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import re
from typing import List, Dict

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences"""
    # Simple sentence splitting
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]

