"""
Model management for the Streamlit app
"""

import streamlit as st
import sys
from typing import Tuple, Optional

# Add project root to path
sys.path.append('.')

from src.classification.model_helpers import load_instruction_model
from src.classification.SDoH_classification_helpers import SDoHExtractor

class ModelManager:
    """Handles model loading and management"""
    
    @st.cache_resource
    def get_model(_self, model_name: str) -> Tuple[Optional[object], Optional[object]]:
        """Load model with caching"""
        with st.spinner(f"Loading {model_name}..."):
            tokenizer, model = load_instruction_model(model_name)
            if model is None:
                st.error(f"Failed to load model: {model_name}")
                return None, None
            return tokenizer, model
    
    @staticmethod
    def create_extractor(tokenizer, model, config: dict) -> Optional[SDoHExtractor]:
        """Create SDoH extractor with current config"""
        if tokenizer is None or model is None:
            return None
            
        return SDoHExtractor(
            model=model,
            tokenizer=tokenizer,
            prompt_type=config['prompt_type'],
            level=config['level'],
            debug=config['debug_mode']
        )