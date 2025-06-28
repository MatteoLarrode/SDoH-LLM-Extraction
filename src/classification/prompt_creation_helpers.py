# ================================================
# ==== Helper functions for prompt generation ====
# ================================================
from typing import List, Dict

def create_automated_prompt(sentence: str, 
                          tokenizer=None, 
                          prompt_type: str = "five_shot_basic") -> str:
    """
    Unified prompt generator that handles all combinations automatically
    
    Args:
        sentence: Input sentence to analyze
        tokenizer: Model tokenizer (for template detection)
        prompt_type: "zero_shot_basic", "zero_shot_detailed", "five_shot_basic", "five_shot_detailed"
    
    Returns:
        Properly formatted prompt for the specific model and configuration
    """
    
    # === STEP 1: Build the core task description ===
    
    # Adverse/Protective classification
    task_intro = "You are analyzing a referral note sentence to identify Social Determinants of Health, and classifying them as Adverse or Protective."
    
    task_instructions = """Given a sentence, output all SDoH factors that can be inferred from that sentence from the following list: 
Loneliness, Housing, Finances, FoodAccess, Digital, Employment, EnglishProficiency.

Each SDoH must be classified as either "Adverse" or "Protective". 
If the sentence does NOT mention any of the above categories, output <LIST>NoSDoH</LIST>.

Your response must be a comma-separated list of SDoH-Polarity pairs embedded in <LIST> and </LIST> tags.

**STRICT RULES**:
- DO NOT generate any other text, explanations, or new SDoH labels.
- A sentence CAN be labeled with one or more SDoH factors.
- The only accepted format is <LIST>...</LIST>."""
    
    examples = """EXAMPLES:
Input: "She is unemployed and struggles to pay rent."
Output: <LIST>Employment-Adverse, Finances-Adverse, Housing-Adverse</LIST>

Input: "We are referring the above patient to you today for befriending."
Output: <LIST>Loneliness-Adverse</LIST>

Input: "She enjoys a strong network of friends and volunteers weekly."
Output: <LIST>Loneliness-Protective</LIST>

Input: "Sleeping at a friend's for now."
Output: <LIST>Housing-Adverse</LIST>

Input: "Cannot take public transport to do groceries."
Output: <LIST>FoodAccess-Adverse</LIST>

Input: "Daughter translates at GP visits."
Output: <LIST>EnglishProficiency-Adverse</LIST>
"""
    
    # === STEP 2: Add detailed guidelines if needed ===
    
    detailed_guidelines = ""
    if "detailed" in prompt_type:
        detailed_guidelines = """
Below are detailed guidelines:

Loneliness: Emotional or physical isolation, absence of social contact, lack of companionship. Excludes practical support needs unless clearly associated with emotional loneliness.

Housing: Covers housing quality, stability, and suitability. Includes homelessness, unsafe or overcrowded housing, or housing not adapted to a person’s health needs.

Finances: Encompasses financial insecurity or security, inability to meet basic needs, debt, or adequate and stable income.

FoodAccess: Describes access to and preparation of adequate nutrition. Includes lack of food, poor diet due to constraints, or reliable access to meals.

Digital: Includes access to and ability to use digital devices or the internet, confidence using them, or support needed to engage digitally.

Employment: Refers to employment status and satisfaction. Includes unemployment, barriers to work, or fulfilling employment.

EnglishProficiency: Captures English language proficiency. Includes limited English that hinders access to services, or confident communication."""
    
    # === STEP 3: Include examples based on shot type ===
    
    examples_section = ""
    if "five_shot" in prompt_type:
        examples_section = examples
    elif "zero_shot" in prompt_type:
        examples_section = ""  # No examples for zero-shot
    
    # === STEP 4: Build the system message ===
    
    system_content = f"""{task_intro}

{task_instructions}{detailed_guidelines}

{examples_section}""".strip()
    
    user_content = f'Input: "{sentence}"'
    
    # === STEP 5: Apply model-specific formatting ===
    
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
        # Generic fallback format
        return f"""{system_content}
{user_content}
Output: """
    
    # Get model name for detection
    model_name = getattr(tokenizer, 'name_or_path', 'unknown').lower()
    
    # Try native chat template first (works for most modern models)
    if hasattr(tokenizer, 'apply_chat_template'):
        print(f"Using chat template for {model_name}")
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


# ================================================
# Prompt generator for GUEVARA SDoH labels
# ================================================

GUEVARA_SDOH = [
    'EMPLOYMENT', 'HOUSING', 'LONELINESS', 'FINANCES', 'FOOD', 'DIGITAL', 'ENGLISH'
]

def create_guevara_prompt(sentence: str) -> str:
    """
    Create an instruction prompt for LLaMA 3.1 to classify a sentence
    using only the GUEVARA SDoH label set.
    
    Args:
        sentence: Input sentence (single sentence)
    
    Returns:
        Formatted prompt string with instruction and few-shot examples
    """
    
    label_list = ", ".join(GUEVARA_SDOH)

    instructions = f"""You are a helpful assistant trained to identify mentions of Social Determinants of Health (SDoH).

Given a sentence, output all relevant SDoH categories from the following list:
{label_list}.

- Use a comma-separated list embedded inside <LIST> and </LIST> tags.
- If none apply, output <LIST>NoSDoH</LIST>.
- Do NOT include any extra text or explanations."""

    few_shot = """EXAMPLES:
Input: "He lost his job and can't afford groceries."
Output: <LIST>EMPLOYMENT, FINANCES, FOOD</LIST>

Input: "She relies on her daughter to fill in online forms."
Output: <LIST>DIGITAL</LIST>

Input: "He lives alone and says he feels lonely most days."
Output: <LIST>LONELINESS</LIST>

Input: "Patient was prescribed medication for diabetes."
Output: <LIST>NoSDoH</LIST>"""

    user_input = f'Input: "{sentence}"'

    # === Format in LLaMA 3 chat template style ===
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{instructions}

{few_shot}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""



# === Convenience functions for backward compatibility ===

def create_zero_shot_basic_prompt(sentence: str, tokenizer=None) -> str:
    """Backward compatible zero-shot basic prompt"""
    return create_automated_prompt(sentence, tokenizer, "zero_shot_basic")

def create_zero_shot_detailed_prompt(sentence: str, tokenizer=None) -> str:
    """Backward compatible zero-shot detailed prompt"""
    return create_automated_prompt(sentence, tokenizer, "zero_shot_detailed")

def create_five_shot_basic_prompt(sentence: str, tokenizer=None) -> str:
    """Backward compatible five-shot basic prompt"""
    return create_automated_prompt(sentence, tokenizer, "five_shot_basic")

def create_five_shot_detailed_prompt(sentence: str, tokenizer=None) -> str:
    """Backward compatible five-shot detailed prompt"""
    return create_automated_prompt(sentence, tokenizer, "five_shot_detailed")