from typing import List

# ======================
#  === Task: Binary classification — does the sentence contain *any* SDoH (protective or adverse) or not?
# ======================
def build_sdoh_detection_prompt(sentence: str, label: str) -> str:
    """
    Construct a LLaMA-Instruct formatted prompt for binary SDoH detection,
    including both Protective and Adverse Social Determinants of Health.

    Args:
        sentence (str): Input sentence to classify.
        label (str): Target label, either "<LIST>SDoH</LIST>" or "<LIST>NoSDoH</LIST>"

    Returns:
        str: A formatted LLaMA-Instruct prompt for classification.
    """

    system_content = (
        "You are a helpful assistant identifying whether a sentence contains any Social Determinants of Health (SDoH).\n\n"
        "Given a sentence, classify it as either containing at least one Social Determinant of Health (SDoH) or not.\n\n"
        "Only consider the following SDoH categories:\n"
        "Loneliness, Housing, Finances, FoodAccess, DigitalInclusion, Employment, EnglishProficiency.\n\n"
        "These determinants may be Protective (e.g., helpful social support) or Adverse (e.g., financial struggles).\n\n"
        "Respond with one of:\n"
        "<LIST>SDoH</LIST> — if the sentence contains any relevant SDoH from the list above.\n"
        "<LIST>NoSDoH</LIST> — if it does not.\n\n"
        "Do not add any other text or labels.\n\n"
        "EXAMPLES:\n"
        "Input: \"He lost his job and can't afford groceries.\"\n"
        "Output: <LIST>SDoH</LIST>\n\n"
        "Input: \"She regularly receives help from her community center.\"\n"
        "Output: <LIST>SDoH</LIST>\n\n"
        "Input: \"Patient was discharged home after treatment.\"\n"
        "Output: <LIST>NoSDoH</LIST>\n"
    )

    user_content = f'Input: "{sentence}"'

    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_content}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{label}"""

# Inference version of the prompt
def build_sdoh_detection_prompt_infer(sentence: str) -> str:
    """
    Construct a LLaMA-Instruct formatted prompt for binary SDoH detection,
    including both Protective and Adverse Social Determinants of Health.

    Args:
        sentence (str): Input sentence to classify.
        label (str): Target label, either "<LIST>SDoH</LIST>" or "<LIST>NoSDoH</LIST>"

    Returns:
        str: A formatted LLaMA-Instruct prompt for classification.
    """

    system_content = (
        "You are a helpful assistant identifying whether a sentence contains any Social Determinants of Health (SDoH).\n\n"
        "Given a sentence, classify it as either containing at least one Social Determinant of Health (SDoH) or not.\n\n"
        "Only consider the following SDoH categories:\n"
        "Loneliness, Housing, Finances, FoodAccess, DigitalInclusion, Employment, EnglishProficiency.\n\n"
        "These determinants may be Protective (e.g., helpful social support) or Adverse (e.g., financial struggles).\n\n"
        "Respond with one of:\n"
        "<LIST>SDoH</LIST> — if the sentence contains any relevant SDoH from the list above.\n"
        "<LIST>NoSDoH</LIST> — if it does not.\n\n"
        "Do not add any other text or labels.\n\n"
        "EXAMPLES:\n"
        "Input: \"He lost his job and can't afford groceries.\"\n"
        "Output: <LIST>SDoH</LIST>\n\n"
        "Input: \"She regularly receives help from her community center.\"\n"
        "Output: <LIST>SDoH</LIST>\n\n"
        "Input: \"Patient was discharged home after treatment.\"\n"
        "Output: <LIST>NoSDoH</LIST>\n"
    )

    user_content = f'Input: "{sentence}"'

    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_content}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

# ======================
# Task: Multi-label 1 — classify *any* SDoH (protective or adverse) or NoSDoH
# ======================
def build_sdoh_multilabel_present_or_not_prompt(sentence: str, label: str) -> str:
    """
    Construct a LLaMA-Instruct prompt for identifying whether the sentence contains *any* SDoH (of any polarity),
    or none at all.
    """
    system_content = (
        "You are identifying whether a sentence contains any Social Determinants of Health (SDoH).\n\n"
        "If it contains one or more SDoH from the following list, return them with their polarity (Adverse or Protective):\n"
        "Loneliness, Housing, Finances, FoodAccess, DigitalInclusion, Employment, EnglishProficiency.\n\n"
        "If none apply, return <LIST>NoSDoH</LIST>\n"
        "Format: <LIST>Label1-Polarity, Label2-Polarity</LIST>\n\n"
        "EXAMPLES:\n"
        "Input: \"She has difficulty finding a job but gets help online.\"\n"
        "Output: <LIST>Employment-Adverse, DigitalInclusion-Protective</LIST>\n\n"
        "Input: \"He lives with family and is financially secure.\"\n"
        "Output: <LIST>Loneliness-Protective, Finances-Protective</LIST>\n\n"
        "Input: \"He was discharged from the hospital.\"\n"
        "Output: <LIST>NoSDoH</LIST>\n\n"
        "Input: \"She cannot pay for her food and is isolated.\"\n"
        "Output: <LIST>FoodAccess-Adverse, Loneliness-Adverse</LIST>\n\n"
        "Input: \"They live in temporary housing and rely on benefits.\"\n"
        "Output: <LIST>Housing-Adverse, Finances-Adverse</LIST>\n\n"
        "Input: \"He is unemployed and his English is limited.\"\n"
        "Output: <LIST>Employment-Adverse, EnglishProficiency-Adverse</LIST>"
    )

    user_content = f'Input: "{sentence}"'

    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_content}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{label}"""

# Inference version of the prompt
def build_sdoh_multilabel_present_or_not_prompt_infer(sentence: str) -> str:
    """
    Construct a LLaMA-Instruct prompt for identifying whether the sentence contains *any* SDoH (of any polarity),
    or none at all.
    """
    system_content = (
        "You are identifying whether a sentence contains any Social Determinants of Health (SDoH).\n\n"
        "If it contains one or more SDoH from the following list, return them with their polarity (Adverse or Protective):\n"
        "Loneliness, Housing, Finances, FoodAccess, DigitalInclusion, Employment, EnglishProficiency.\n\n"
        "If none apply, return <LIST>NoSDoH</LIST>\n"
        "Format: <LIST>Label1-Polarity, Label2-Polarity</LIST>\n\n"
        "EXAMPLES:\n"
        "Input: \"She has difficulty finding a job but gets help online.\"\n"
        "Output: <LIST>Employment-Adverse, DigitalInclusion-Protective</LIST>\n\n"
        "Input: \"He lives with family and is financially secure.\"\n"
        "Output: <LIST>Loneliness-Protective, Finances-Protective</LIST>\n\n"
        "Input: \"He was discharged from the hospital.\"\n"
        "Output: <LIST>NoSDoH</LIST>\n\n"
        "Input: \"She cannot pay for her food and is isolated.\"\n"
        "Output: <LIST>FoodAccess-Adverse, Loneliness-Adverse</LIST>\n\n"
        "Input: \"They live in temporary housing and rely on benefits.\"\n"
        "Output: <LIST>Housing-Adverse, Finances-Adverse</LIST>\n\n"
        "Input: \"He is unemployed and his English is limited.\"\n"
        "Output: <LIST>Employment-Adverse, EnglishProficiency-Adverse</LIST>"
    )

    user_content = f'Input: "{sentence}"'

    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_content}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

# ======================
# Task: Multi-label 2 — classify sentence as containing adverse SDoH(s) or not (output: labels or NoSDoH)
# ======================
def build_sdoh_adverse_only_prompt(sentence: str, label: str) -> str:
    """
    Construct a LLaMA-Instruct formatted prompt for extracting *only* Adverse SDoH.
    """

    system_content = (
        "You are identifying *adverse* Social Determinants of Health (SDoH) from a sentence.\n\n"
        "Only extract SDoH that are Adverse from the following list:\n"
        "Loneliness, Housing, Finances, FoodAccess, DigitalInclusion, Employment, EnglishProficiency.\n\n"
        "If none apply, return <LIST>NoSDoH</LIST>\n"
        "Format: <LIST>Label1-Adverse, Label2-Adverse</LIST>\n\n"
        "Do not include any protective determinants.\n\n"
        "EXAMPLES:\n"
        "Input: \"She can't afford rent and feels isolated.\"\n"
        "Output: <LIST>Finances-Adverse, Loneliness-Adverse</LIST>\n\n"
        "Input: \"Patient has internet access and receives unemployment benefits.\"\n"
        "Output: <LIST>Employment-Adverse</LIST>\n\n"
        "Input: \"He volunteers twice a week and lives with his family.\"\n"
        "Output: <LIST>NoSDoH</LIST>\n\n"
        "Input: \"He can't use a smartphone and doesn't speak English.\"\n"
        "Output: <LIST>DigitalInclusion-Adverse, EnglishProficiency-Adverse</LIST>\n\n"
        "Input: \"Her home was recently condemned.\"\n"
        "Output: <LIST>Housing-Adverse</LIST>\n\n"
        "Input: \"He is lonely despite living with a roommate.\"\n"
        "Output: <LIST>Loneliness-Adverse</LIST>"
    )

    user_content = f'Input: "{sentence}"'

    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_content}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{label}"""

# ======================
# Task: Multi-label 3 — classify SDoH from a sentence assumed to contain at least one SDoH (any polarity)
# ======================
def build_sdoh_from_sentence_prompt(sentence: str, label: str) -> str:
    """
    Construct a prompt assuming the sentence contains at least one SDoH (any polarity).
    """

    system_content = (
        "You are given a sentence that contains one or more Social Determinants of Health (SDoH).\n\n"
        "Identify all relevant categories from:\n"
        "Loneliness, Housing, Finances, FoodAccess, DigitalInclusion, Employment, EnglishProficiency.\n\n"
        "Label each SDoH as either Adverse or Protective.\n"
        "Format: <LIST>Label1-Polarity, Label2-Polarity</LIST>\n\n"
        "Do not include labels that are not present in the sentence.\n\n"
        "EXAMPLES:\n"
        "Input: \"She is unemployed but volunteers at the community center.\"\n"
        "Output: <LIST>Employment-Adverse, Loneliness-Protective</LIST>\n\n"
        "Input: \"He lives alone and enjoys digital skills workshops.\"\n"
        "Output: <LIST>Loneliness-Adverse, DigitalInclusion-Protective</LIST>\n\n"
        "Input: \"They rely on a food bank and recently moved into temporary housing.\"\n"
        "Output: <LIST>FoodAccess-Adverse, Housing-Adverse</LIST>\n\n"
        "Input: \"He is financially stable and has strong family support.\"\n"
        "Output: <LIST>Finances-Protective, Loneliness-Protective</LIST>\n\n"
        "Input: \"Her English is limited and she can't access online services.\"\n"
        "Output: <LIST>EnglishProficiency-Adverse, DigitalInclusion-Adverse</LIST>\n\n"
        "Input: \"She rents a clean and affordable flat.\"\n"
        "Output: <LIST>Housing-Protective</LIST>"
    )

    user_content = f'Input: "{sentence}"'

    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_content}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{label}"""

# ======================
# Task: Multi-label 4 — classify only adverse SDoH from a sentence assumed to contain at least one SDoH
# ======================
def build_sdoh_adverse_only_from_sentence_prompt(sentence: str, label: str) -> str:
    """
    Construct a prompt assuming the sentence contains at least one SDoH, but only extract Adverse ones.
    """

    system_content = (
        "You are given a sentence that contains at least one Social Determinant of Health (SDoH).\n\n"
        "From the following list, extract only those that are *Adverse*:\n"
        "Loneliness, Housing, Finances, FoodAccess, DigitalInclusion, Employment, EnglishProficiency.\n\n"
        "If none are adverse, output <LIST>NoSDoH</LIST>\n"
        "Format: <LIST>Label1-Adverse, Label2-Adverse</LIST>\n\n"
        "EXAMPLES:\n"
        "Input: \"She is unemployed and receives food assistance.\"\n"
        "Output: <LIST>Employment-Adverse</LIST>\n\n"
        "Input: \"He lives alone and recently lost internet access.\"\n"
        "Output: <LIST>Loneliness-Adverse, DigitalInclusion-Adverse</LIST>\n\n"
        "Input: \"They struggle to pay bills but get help from neighbors.\"\n"
        "Output: <LIST>Finances-Adverse</LIST>\n\n"
        "Input: \"She lives in a well-maintained community housing.\"\n"
        "Output: <LIST>NoSDoH</LIST>\n\n"
        "Input: \"He is often isolated and cannot navigate English-language forms.\"\n"
        "Output: <LIST>Loneliness-Adverse, EnglishProficiency-Adverse</LIST>\n\n"
        "Input: \"They have sufficient income and use the internet confidently.\"\n"
        "Output: <LIST>NoSDoH</LIST>"
    )

    user_content = f'Input: "{sentence}"'

    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_content}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{label}"""