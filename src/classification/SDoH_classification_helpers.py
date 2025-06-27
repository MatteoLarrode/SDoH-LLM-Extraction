import re
import pandas as pd
import json
import torch
from typing import List, Dict, Any, Optional, Tuple
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from src.classification.prompt_creation_helpers import create_automated_prompt
from src.data_cleaning.data_cleaning_helpers import (
    split_into_sentences
)

class SDoHExtractor:
    """Main class for extracting Social Determinants of Health from referral notes"""
    
    def __init__(self, model, tokenizer, prompt_type: str = "zero_shot_detailed", level: int = 1, max_length: int = 512, debug: bool = False):
        """
        Initialize the SDoH extractor
        
        Args:
            model: Loaded transformers model
            tokenizer: Loaded transformers tokenizer
            prompt_type: One of ["zero_shot_basic", "zero_shot_detailed", "five_shot_basic", "five_shot_detailed"]
            level: 1 for basic categories, 2 for adverse/non-adverse classification
            max_length: Maximum length for model generation
            debug: Whether to include debugging information in results
        """
        self.model = model
        self.tokenizer = tokenizer
        self.prompt_type = prompt_type
        self.level = level
        self.max_length = max_length
        self.debug = debug
        
        # Validate prompt type
        valid_prompt_types = ["zero_shot_basic", "zero_shot_detailed", "five_shot_basic", "five_shot_detailed"]
        if prompt_type not in valid_prompt_types:
            raise ValueError(f"Invalid prompt_type. Must be one of {valid_prompt_types}")
        
        # Validate level
        if level not in [1, 2]:
            raise ValueError(f"Invalid level. Must be 1 or 2, got {level}")
    
    def preprocess_referral_note(self, note: str) -> List[str]:
        """
        Preprocess referral note into sentences
        
        Args:
            note: Raw referral note text
            
        Returns:
            List of cleaned sentences
        """
        return split_into_sentences(note)
    
    def extract_from_sentence(self, sentence: str) -> Dict[str, Any]:
        """
        Extract SDoH factors from a single sentence
        
        Args:
            sentence: Single sentence to analyze
            
        Returns:
            Dictionary with SDoH factors and debugging info (if enabled)
        """
        # Create prompt using unified automated function
        prompt = create_automated_prompt(
            sentence=sentence,
            tokenizer=self.tokenizer,
            prompt_type=self.prompt_type,
            level=self.level
        )
        
        # Get response from model
        raw_response = self._get_model_response(prompt)
        
        # Parse response
        factors = self._parse_list_response(raw_response)
        
        # Build result
        result = {
            "sdoh_factors": factors
        }
        
        # Add debugging info if enabled
        if self.debug:
            result["debug"] = {
                "prompt": prompt,
                "raw_response": raw_response,
                # "formatted_prompt": self._get_formatted_prompt(prompt)
            }
        
        return result
    
    def extract_from_note(self, note: str) -> Dict[str, Any]:
        """
        Extract SDoH factors from entire referral note
        
        Args:
            note: Complete referral note text
            
        Returns:
            Dictionary with sentences and their SDoH factors
        """
        sentences = self.preprocess_referral_note(note)
        results = {
            "original_note": note,
            "sentences": [],
            "summary": {}
        }
        
        # Add extraction metadata
        if self.debug:
            results["extraction_metadata"] = {
                "prompt_type": self.prompt_type,
                "level": self.level,
                "max_length": self.max_length,
                "model_name": getattr(self.tokenizer, 'name_or_path', 'unknown'),
                "debug_enabled": self.debug
            }
        
        all_factors = []
        
        for i, sentence in enumerate(sentences):
            extraction_result = self.extract_from_sentence(sentence)
            factors = extraction_result["sdoh_factors"]
            
            sentence_result = {
                "sentence_number": i + 1,
                "sentence": sentence,
                "sdoh_factors": factors
            }
            
            # Add debugging info if available
            if self.debug and "debug" in extraction_result:
                sentence_result["debug"] = extraction_result["debug"]
            
            results["sentences"].append(sentence_result)
            
            # Collect all factors for summary (excluding NoSDoH)
            if factors and factors != ["NoSDoH"]:
                all_factors.extend(factors)
        
        # Create summary
        results["summary"] = self._create_summary(all_factors)
        
        return results
    
    def results_to_dataframe(self, results: Dict[str, Any], note_id: str = None) -> pd.DataFrame:
        """
        Convert extraction results to a pandas DataFrame (one row per sentence)
        
        Args:
            results: Output from extract_from_note()
            note_id: Optional identifier for the note
            
        Returns:
            DataFrame with one row per sentence
        """
        rows = []
        
        for sentence_data in results["sentences"]:
            factors = sentence_data["sdoh_factors"]
            
            # Create single row per sentence
            row = {
                "note_id": note_id,
                "sentence_number": sentence_data["sentence_number"],
                "sentence": sentence_data["sentence"],
                "has_sdoh": factors != ["NoSDoH"] and bool(factors),
                "sdoh_factors": ", ".join(factors) if factors else "NoSDoH",
                "num_sdoh_factors": len(factors) if factors != ["NoSDoH"] else 0
            }
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    # def _get_formatted_prompt(self, prompt: str) -> str:
    #     """Get the formatted prompt that would be sent to the model"""
    #     if "llama" in self.tokenizer.name_or_path.lower():
    #         return self._format_llama_prompt(prompt)
    #     elif "qwen" in self.tokenizer.name_or_path.lower():
    #         return self._format_qwen_prompt(prompt)
    #     elif "phi" in self.tokenizer.name_or_path.lower():
    #         return self._format_phi_prompt(prompt)
    #     elif "mistral" in self.tokenizer.name_or_path.lower():
    #         return self._format_mistral_prompt(prompt)
    #     else:
    #         return prompt
    
    def _get_model_response(self, prompt: str) -> str:
        """
        Get response from the local model
        
        Args:
            prompt: The formatted prompt
            
        Returns:
            Generated response string from model
        """
        try:
            # Format prompt for instruction-tuned models
            # if "llama" in self.tokenizer.name_or_path.lower():
            #     formatted_prompt = self._format_llama_prompt(prompt)
            # elif "qwen" in self.tokenizer.name_or_path.lower():
            #     formatted_prompt = self._format_qwen_prompt(prompt)
            # elif "phi" in self.tokenizer.name_or_path.lower():
            #     formatted_prompt = self._format_phi_prompt(prompt)
            # elif "mistral" in self.tokenizer.name_or_path.lower():
            #     formatted_prompt = self._format_mistral_prompt(prompt)
            # else:
                # Generic format
            formatted_prompt = prompt
            
            # Tokenize input
            inputs = self.tokenizer(
                formatted_prompt, 
                return_tensors="pt", 
                truncation=True,
                max_length=self.max_length
            )
            
            # Move to same device as model
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    tokenizer=self.tokenizer,
                    max_new_tokens=50,      
                    do_sample=False,  
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "NoSDoH"
        
    def _parse_list_response(self, response: str) -> List[str]:
        """
        Parse the <LIST></LIST> format response
        
        Args:
            response: Raw response from model
            
        Returns:
            List of SDoH factors
        """
        # Extract content between <LIST> and </LIST>
        match = re.search(r'<LIST>(.*?)</LIST>', response, re.DOTALL | re.IGNORECASE)
        
        if not match:
            # If no <LIST> tags found, return default
            return ["NoSDoH"]
        
        else:
            # Extract the content inside <LIST> tags
            list_content = match.group(1).strip()
        
        if not list_content or list_content.lower() == "nosdoh":
            return ["NoSDoH"]
        
        # Split by comma and clean
        factors = [factor.strip() for factor in list_content.split(',')]
        factors = [factor for factor in factors if factor]  # Remove empty strings
        
        return factors if factors else ["NoSDoH"]
    
    def _create_summary(self, all_factors: List[str]) -> Dict[str, Any]:
        """Create summary of all SDoH factors found"""
        if not all_factors:
            return {"unique_factors": [], "factor_counts": {}, "total_mentions": 0}
        
        # Count occurrences
        factor_counts = {}
        for factor in all_factors:
            factor_counts[factor] = factor_counts.get(factor, 0) + 1
        
        return {
            "unique_factors": list(set(all_factors)),
            "factor_counts": factor_counts,
            "total_mentions": len(all_factors)
        }