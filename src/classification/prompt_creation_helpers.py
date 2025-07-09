# ================================================
# ==== Helper functions for prompt generation ====
# ================================================
from typing import List, Dict


from typing import List, Union

def create_automated_prompt(sentence: str,
                            labels: Union[str, List[str], int] = None,
                            task_type: str = "sdoh_multilabel",
                            tokenizer=None) -> str:
    """
    Generate a prompt based on the sentence and task type.

    Args:
        sentence: The input sentence to classify
        labels: List of labels, or int (0/1) for binary task
        task_type: One of ['sdoh_detection', 'sdoh_multilabel', 'sdoh_adverse_only', 'sdoh_polarity']
        tokenizer: Optional tokenizer for applying chat template

    Returns:
        A formatted instruction-style prompt string
    """

    # ===== Instructions per task =====
    if task_type == "sdoh_detection":
        task_intro = "You are a helpful assistant identifying whether a sentence contains any Social Determinants of Health (SDoH)."
        task_instructions = """
Given a sentence, classify it as either containing at least one Social Determinant of Health (SDoH) or not.

Only consider the following SDoH categories:
Loneliness, Housing, Finances, FoodAccess, DigitalInclusion, Employment, EnglishProficiency.

Respond with one of:
<LIST>SDoH</LIST> — if the sentence contains any relevant SDoH from the list above.
<LIST>NoSDoH</LIST> — if it does not.

Do not add any other text or labels."""
        examples = """EXAMPLES:
Input: "He lost his job and can't afford groceries."
Output: <LIST>SDoH</LIST>

Input: "Patient was discharged home after treatment."
Output: <LIST>NoSDoH</LIST>"""

    elif task_type == "sdoh_multilabel":
        task_intro = "You are analyzing a referral note to identify all mentioned Social Determinants of Health (SDoH), labeled as Adverse or Protective."
        task_instructions = """
Given a sentence, extract all relevant SDoH categories from this list:
Loneliness, Housing, Finances, FoodAccess, DigitalInclusion, Employment, EnglishProficiency.

Each must be labeled as Adverse or Protective. If none apply, output <LIST>NoSDoH</LIST>.

Format your answer like: <LIST>Label1-Polarity, Label2-Polarity</LIST>
Strictly avoid extra text."""
        examples = """EXAMPLES:
Input: "She is unemployed and struggles to pay rent."
Output: <LIST>Employment-Adverse, Finances-Adverse, Housing-Adverse</LIST>

Input: "She enjoys a strong network of friends and volunteers weekly."
Output: <LIST>Loneliness-Protective</LIST>

Input: "Patient was discharged after surgery."
Output: <LIST>NoSDoH</LIST>"""

    elif task_type == "sdoh_adverse_only":
        task_intro = "You are detecting only *Adverse* Social Determinants of Health (SDoH) in a referral sentence."
        task_instructions = """
Only extract SDoH that are labeled as Adverse from the following list:
Loneliness, Housing, Finances, FoodAccess, DigitalInclusion, Employment, EnglishProficiency.

Format: <LIST>Label1-Adverse, Label2-Adverse</LIST>
If there are none, output <LIST>NoSDoH</LIST>."""
        examples = """EXAMPLES:
Input: "He lives alone and says he feels lonely."
Output: <LIST>Loneliness-Adverse</LIST>

Input: "Patient has a reliable income and lives with his daughter."
Output: <LIST>NoSDoH</LIST>"""

    elif task_type == "sdoh_polarity":
        task_intro = "You are classifying whether the mentioned SDoH in a sentence are Adverse or Protective."
        task_instructions = """
Each SDoH mention should be labeled as either Adverse or Protective.

Format your answer as: <LIST>Label1-Polarity</LIST>
Only include labels that appear in the sentence."""
        examples = """EXAMPLES:
Input: "She feels lonely and isolated."
Output: <LIST>Loneliness-Adverse</LIST>

Input: "He has strong family support and daily social visits."
Output: <LIST>Loneliness-Protective</LIST>"""

    else:
        raise ValueError(f"Unsupported task_type: {task_type}")

    # ==== Final formatting ====
    system_content = f"{task_intro}\n\n{task_instructions}\n\n{examples}".strip()
    user_content = f'Input: "{sentence}"'

    return _apply_chat_template(system_content, user_content, tokenizer)


def _apply_chat_template(system_content: str, user_content: str, tokenizer=None) -> str:
    """
    Apply the appropriate chat template based on the tokenizer/model
    
    Args:
        system_content: The system message content
        user_content: The user message content  
        tokenizer: Model tokenizer for template detection
    
    Returns:
        Formatted prompt string
    """
    
    if tokenizer is None:
        # Fallback to LLaMA-style manual template if tokenizer is not available
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_content}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    # Get model name for detection
    model_name = getattr(tokenizer, 'name_or_path', 'unknown').lower()
    
    # Try native chat template first (works for most modern models)
    if hasattr(tokenizer, 'apply_chat_template'):
        #print(f"Using chat template for {model_name}")
        try:
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]
            return tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        except Exception as e:
            print(f"Chat template failed for {model_name}: {e}")
            # Continue to manual formatting
    
    # Model-specific manual formatting
    if 'llama' in model_name:
        # Official Llama 3.1 format
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_content}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    elif 'qwen' in model_name:
        # Qwen 2.5 format
        return f"""<|im_start|>system
{system_content}<|im_end|>
<|im_start|>user
{user_content}<|im_end|>
<|im_start|>assistant
"""
    
    elif 'phi' in model_name:
        # Phi-4 format
        return f"""<|system|>
{system_content}<|end|>
<|user|>
{user_content}<|end|>
<|assistant|>
"""
    
    elif 'mistral' in model_name:
        # Mistral format
        return f"""<s>[INST] {system_content}

{user_content} [/INST]"""
    
    else:
        # Generic format for unknown models
        return f"""{system_content}

{user_content}
Output: """