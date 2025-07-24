from typing import List

# Example blocks
EXAMPLES_ANY_SDOH = """\
Input: "She only eats toast and skips meals."
Output: <LIST>FoodAccess</LIST>

Input: "He lives alone and struggles with money."
Output: <LIST>Finances</LIST>

Input: "He lives on his own."
Output: <LIST>NoSDoH</LIST>

Input: "She cannot pay for her food and is isolated."
Output: <LIST>FoodAccess, Finances, Loneliness</LIST>

Input: "They are sleeping at a friend's for now."
Output: <LIST>Housing</LIST>

Input: "Owns a smartphone but doesn't know how to use it."
Output: <LIST>DigitalInclusion</LIST>

Input: "She is unemployed and speaks limited English."
Output: <LIST>Employment, EnglishProficiency</LIST>

Input: "Patient was discharged from hospital."
Output: <LIST>NoSDoH</LIST>

Input: "She volunteers weekly and receives regular visits from friends."
Output: <LIST>Loneliness</LIST>

Input: "He has stable employment and uses digital tools independently."
Output: <LIST>Employment, DigitalInclusion</LIST>"""

EXAMPLES_ADVERSE_SDOH = """\
Input: "She only eats toast and skips meals."
Output: <LIST>FoodAccess-Adverse</LIST>

Input: "He struggles with money and is sleeping at a friend's for now."
Output: <LIST>Finances-Adverse, Housing-Adverse</LIST>

Input: "He lives on his own."
Output: <LIST>NoSDoH</LIST>

Input: "She cannot pay for her food and is isolated."
Output: <LIST>FoodAccess-Adverse, Finances-Adverse, Loneliness-Adverse</LIST>

Input: "Owns a smartphone but doesn't know how to use it."
Output: <LIST>DigitalInclusion-Adverse</LIST>

Input: "She is unemployed and speaks limited English."
Output: <LIST>Employment-Adverse, EnglishProficiency-Adverse</LIST>

Input: "She gets meals delivered daily and volunteers locally."
Output: <LIST>NoSDoH</LIST>

Input: "He uses his tablet independently and enjoys weekly clubs."
Output: <LIST>NoSDoH</LIST>

Input: "Patient was discharged from hospital."
Output: <LIST>NoSDoH</LIST>"""

# ===============================================================================
# Task: Multi-label 1 — classify *any* SDoH (protective or adverse) or NoSDoH
# ===============================================================================
def format_prompt(system_content: str, sentence: str, label: str = "") -> str:
    user_content = f'Input: "{sentence}"'
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_content}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{label}"""

def build_sdoh_multilabel_present_or_not_prompt(sentence: str, label: str) -> str:
    """
    Construct a LLaMA-Instruct prompt for identifying whether the sentence contains *any* SDoH (of any polarity),
    or none at all.
    """
    system_content = (
        "You are identifying whether a sentence contains any Social Determinants of Health (SDoH).\n"
        "Only consider information about the subject of the sentence, not their family members, carers, or others mentioned.\n\n"
        "If it contains one or more SDoH from the following list, return them:\n"
        "Loneliness, Housing, Finances, FoodAccess, DigitalInclusion, Employment, EnglishProficiency.\n\n"
        "If none apply, return <LIST>NoSDoH</LIST>\n"
        "Format: <LIST>Label1, Label2</LIST>\n\n"
        f"EXAMPLES:\n{EXAMPLES_ANY_SDOH}"
    )

    return format_prompt(system_content, sentence, label)

# Inference version of the prompt
def build_sdoh_multilabel_present_or_not_prompt_infer(sentence: str) -> str:
    """
    Construct a LLaMA-Instruct prompt for identifying whether the sentence contains *any* SDoH (of any polarity),
    or none at all.
    """
    system_content = (
        "You are identifying whether a sentence contains any Social Determinants of Health (SDoH).\n"
        "Only consider information about the subject of the sentence, not their family members, carers, or others mentioned.\n\n"
        "If it contains one or more SDoH from the following list, return them:\n"
        "Loneliness, Housing, Finances, FoodAccess, DigitalInclusion, Employment, EnglishProficiency.\n\n"
        "If none apply, return <LIST>NoSDoH</LIST>\n"
        "Format: <LIST>Label1, Label2</LIST>\n\n"
        f"EXAMPLES:\n{EXAMPLES_ANY_SDOH}"
    )

    return format_prompt(system_content, sentence)

# ===========================================================================================================
# Task: Multi-label 2 — classify sentence as containing adverse SDoH(s) or not (output: labels or NoSDoH)
# ===========================================================================================================
def build_sdoh_adverse_only_prompt(sentence: str, label: str) -> str:
    """
    Construct a LLaMA-Instruct formatted prompt for extracting *only* Adverse SDoH.
    """

    system_content = (
        "You are identifying *adverse* Social Determinants of Health (SDoH) from a sentence.\n"
        "Only consider information about the subject of the sentence, not their family members, carers, or others mentioned.\n\n"
        "Only extract SDoH that are Adverse from the following list:\n"
        "Loneliness, Housing, Finances, FoodAccess, DigitalInclusion, Employment, EnglishProficiency.\n\n"
        "If none apply, return <LIST>NoSDoH</LIST>\n"
        "Format: <LIST>Label1-Adverse, Label2-Adverse</LIST>\n\n"
        "Do not include any protective determinants.\n\n"
        f"EXAMPLES:\n{EXAMPLES_ADVERSE_SDOH}"
    )

    return format_prompt(system_content, sentence, label)

# Inference version of the prompt
def build_sdoh_adverse_only_prompt_infer(sentence: str) -> str:
    """
    Construct a LLaMA-Instruct formatted prompt for extracting *only* Adverse SDoH.
    """

    system_content = (
        "You are identifying *adverse* Social Determinants of Health (SDoH) from a sentence.\n"
        "Only consider information about the subject of the sentence, not their family members, carers, or others mentioned.\n\n"
        "Only extract SDoH that are Adverse from the following list:\n"
        "Loneliness, Housing, Finances, FoodAccess, DigitalInclusion, Employment, EnglishProficiency.\n\n"
        "If none apply, return <LIST>NoSDoH</LIST>\n"
        "Format: <LIST>Label1-Adverse, Label2-Adverse</LIST>\n\n"
        "Do not include any protective determinants.\n\n"
        f"EXAMPLES:\n{EXAMPLES_ADVERSE_SDOH}"
    )

    return format_prompt(system_content, sentence)