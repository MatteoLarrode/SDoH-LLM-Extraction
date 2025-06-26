# ================================================
# ==== Helper functions for prompt generation ====
# ================================================
from typing import List, Dict

def create_automated_prompt(sentence: str, 
                          tokenizer=None, 
                          prompt_type: str = "five_shot_basic",
                          level: int = 1) -> str:
    """
    Unified prompt generator that handles all combinations automatically
    
    Args:
        sentence: Input sentence to analyze
        tokenizer: Model tokenizer (for template detection)
        prompt_type: "zero_shot_basic", "zero_shot_detailed", "five_shot_basic", "five_shot_detailed"
        level: 1 (basic classification) or 2 (adverse/protective classification)
    
    Returns:
        Properly formatted prompt for the specific model and configuration
    """
    
    # === STEP 1: Build the core task description ===
    
    if level == 1:
        # Basic SDoH classification
        task_intro = "You are analyzing a referral note sentence to identify mentions of Social Determinants of Health (SDoH)."
        
        task_instructions = """Given a sentence, output all SDoH factors that can be inferred from that sentence from the following list: 
EmploymentStatus, Housing, Transportation, ParentalStatus, RelationshipStatus, SocialSupport, SubstanceUse, FinancialSituation, EducationLevel, FoodInsecurity. 

If the sentence does NOT mention any of the above categories, output <LIST>NoSDoH</LIST>.

Your response must be a comma-separated list of SDoH factors embedded with <LIST> and </LIST>.

**STRICT RULES**: 
- DO NOT generate any other text, explanations, or new SDoH labels.
- A sentence CAN be labeled with one or more SDoH factors.
- Your response must ONLY contain the <LIST>...</LIST> format.
- Do not continue or complete the input sentence."""

        examples = """EXAMPLES:
Input: "Person is unemployed and lives with his elderly mother."
Output: <LIST>EmploymentStatus</LIST>

Input: "She struggles to afford groceries and has no car to get to the store."
Output: <LIST>FinancialSituation, Transportation, FoodInsecurity</LIST>

Input: "Person smokes cigarettes and drinks alcohol daily."
Output: <LIST>SubstanceUse</LIST>

Input: "He has three young children at home and receives help from his sister."
Output: <LIST>ParentalStatus, SocialSupport</LIST>

Input: "Person was prescribed medication for diabetes."
Output: <LIST>NoSDoH</LIST>"""
    
    else:  # level 2
        # Adverse/Protective classification
        task_intro = "You are analyzing a referral note sentence to identify Social Determinants of Health, and classifying them as Adverse or Protective."
        
        task_instructions = """Given a sentence, output all SDoH factors that can be inferred from that sentence from the following list: 
EmploymentStatus, Housing, Transportation, ParentalStatus, RelationshipStatus, SocialSupport, SubstanceUse, FinancialSituation, EducationLevel, FoodInsecurity. 

If the sentence does NOT mention any of the above categories, output <LIST>NoSDoH</LIST>.

Your response must be a comma-separated list of SDoH factors embedded with <LIST> and </LIST>.

**STRICT RULES**:
- For EVERY SDoH mention found, you MUST classify it as either "Adverse" or "Protective" after a hyphen
- DO NOT generate any other text, explanations, or new SDoH labels.
- A sentence CAN be labeled with one or more SDoH factors.
- Your response must ONLY contain the <LIST>...</LIST> format.
- Do not continue or complete the input sentence."""

        examples = """EXAMPLES:
Input: "Person is unemployed and lives with his elderly mother."
Output: <LIST>EmploymentStatus-Adverse</LIST>

Input: "She struggles to afford groceries and has no car to get to the store."
Output: <LIST>FinancialSituation-Adverse, Transportation-Adverse, FoodInsecurity-Adverse</LIST>

Input: "Person smokes cigarettes and drinks alcohol daily."
Output: <LIST>SubstanceUse-Adverse</LIST>

Input: "He has three young children at home and receives help from his sister."
Output: <LIST>ParentalStatus-Adverse, SocialSupport-Protective</LIST>

Input: "Person was prescribed medication for diabetes."
Output: <LIST>NoSDoH</LIST>"""
    
    # === STEP 2: Add detailed guidelines if needed ===
    
    detailed_guidelines = ""
    if "detailed" in prompt_type:
        detailed_guidelines = """
Below are detailed guidelines:

EmploymentStatus: Annotate sentences that describe a person's work situation, including current employment, unemployment, retirement status, or disability affecting work capacity.

Housing: Annotate sentences that describe housing conditions, stability, or problems including homelessness, temporary housing, housing quality, overcrowding, unsafe living conditions, housing affordability, eviction risk, or housing-related health hazards.

Transportation: Annotate sentences that describe difficulties accessing transportation, lack of reliable transportation, inability to travel for medical appointments, public transit limitations, vehicle problems, or mobility barriers that affect daily activities or healthcare access.

ParentalStatus: Annotate sentences that indicate whether the person has children, has parental responsibilities, custody arrangements, child-rearing challenges, or family composition including dependents.

RelationshipStatus: Annotate sentences that describe marital status (married, divorced, widowed, separated), partnership status (single, dating, cohabiting).

SocialSupport: Annotate sentences that describe availability or absence of help from family, friends, or community including emotional support, practical assistance, social connections, isolation, loneliness.

SubstanceUse: Annotate sentences that mention current or past use of alcohol, illegal drugs, prescription drug misuse, tobacco products, smoking cessation attempts, substance abuse treatment, or substance-related health problems.

FinancialSituation: Annotate sentences that describe economic hardship, income adequacy, debt problems, inability to afford necessities, or financial stress.

EducationLevel: Annotate sentences that mention highest level of education completed, literacy skills, educational barriers, special education needs.

FoodInsecurity: Annotate sentences that describe inadequate food access, hunger, reliance on food assistance programs, poor nutrition due to cost, skipping meals, food scarcity, or difficulty obtaining healthy foods."""
    
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
    'EMPLOYMENT', 'HOUSING', 'PARENT', 
    'RELATIONSHIP', 'SUPPORT', 'TRANSPORTATION'
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
Input: "He lost his job during the pandemic and lives with a friend."
Output: <LIST>EMPLOYMENT, HOUSING</LIST>

Input: "She relies on public buses but missed several medical appointments."
Output: <LIST>TRANSPORTATION</LIST>

Input: "They are first-time parents with no family nearby."
Output: <LIST>PARENT, SUPPORT</LIST>

Input: "She lives alone and has no close relationships."
Output: <LIST>RELATIONSHIP, SUPPORT</LIST>

Input: "Patient was diagnosed with asthma at age 9."
Output: <LIST>NoSDoH</LIST>"""

    user_input = f'Input: "{sentence}"'

    # === Format in LLaMA 3 chat template style ===
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{instructions}

{few_shot}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""



# === Convenience functions for backward compatibility ===

def create_zero_shot_basic_prompt(sentence: str, level: int = 1, tokenizer=None) -> str:
    """Backward compatible zero-shot basic prompt"""
    return create_automated_prompt(sentence, tokenizer, "zero_shot_basic", level)

def create_zero_shot_detailed_prompt(sentence: str, level: int = 1, tokenizer=None) -> str:
    """Backward compatible zero-shot detailed prompt"""
    return create_automated_prompt(sentence, tokenizer, "zero_shot_detailed", level)

def create_five_shot_basic_prompt(sentence: str, level: int = 1, tokenizer=None) -> str:
    """Backward compatible five-shot basic prompt"""
    return create_automated_prompt(sentence, tokenizer, "five_shot_basic", level)

def create_five_shot_detailed_prompt(sentence: str, level: int = 1, tokenizer=None) -> str:
    """Backward compatible five-shot detailed prompt"""
    return create_automated_prompt(sentence, tokenizer, "five_shot_detailed", level)