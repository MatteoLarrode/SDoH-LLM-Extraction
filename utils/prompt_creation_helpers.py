# ================================================
# ==== Helper functions for prompt generation ====
# ================================================
from typing import List, Dict

def create_zero_shot_basic_prompt(sentence: str, level: int = 1) -> str:
    """Zero-shot prompt with just category list, no descriptions"""
    
    if level == 1:
        task_description = """
You are analyzing a referral note sentence to identify mentions of Social Determinants of Health (SDoH).

Given a sentence, output all SDoH factors that can be inferred from that sentence from the following list: 
EmploymentStatus, Housing, Transportation, ParentalStatus, RelationshipStatus, SocialSupport, SubstanceUse, FinancialSituation, EducationLevel, FoodInsecurity. If the sentence does NOT mention any of the above categories, output - NoSDoH.

Your response must be a comma-separated list of SDoH factors embedded with <LIST> and </LIST>.

**STRICT RULES**: 
- DO NOT generate any other text, explanations, or new SDoH labels.
- A sentence CAN be labeled with one or more SDoH factors.
"""
        
    else:  # level 2
        task_description = """
You are analyzing a referral note sentence to identify Social Determinants of Health, and classify them as Adverse or Protective.

Given a sentence, output all SDoH factors that can be inferred from that sentence from the following list: 
EmploymentStatus, Housing, Transportation, ParentalStatus, RelationshipStatus, SocialSupport, SubstanceUse, FinancialSituation, EducationLevel, FoodInsecurity. If the sentence does NOT mention any of the above categories, output - NoSDoH.

Your response must be a comma-separated list of SDoH factors embedded with <LIST> and </LIST>.

**STRICT RULES**:
- For EVERY SDoH mention found, you MUST classify it as either "Adverse" or "Protective" after a hyphen
- DO NOT generate any other text, explanations, or new SDoH labels.
- A sentence CAN be labeled with one or more SDoH factors.
"""
    
    prompt = f"""{task_description}

Input: {sentence}
"""
    
    return prompt

def create_zero_shot_detailed_prompt(sentence: str, level: int = 1) -> str:
    """Zero-shot prompt with detailed category descriptions"""
    
    if level == 1:
        task_description = """
You are analyzing a referral note sentence to identify mentions of Social Determinants of Health (SDoH).

Given a sentence, output all SDoH factors that can be inferred from that sentence from the following list: 
EmploymentStatus, Housing, Transportation, ParentalStatus, RelationshipStatus, SocialSupport, SubstanceUse, FinancialSituation, EducationLevel, FoodInsecurity. If the sentence does NOT mention any of the above categories, output - NoSDoH.

Your response must be a comma-separated list of SDoH factors embedded with <LIST> and </LIST>.

**STRICT RULES**: 
- DO NOT generate any other text, explanations, or new SDoH labels.
- A sentence CAN be labeled with one or more SDoH factors.

Below are some guidelines:

Employment status: Annotate sentences that describe a person's work situation, including current employment, unemployment, retirement status, or disability affecting work capacity.

Housing: Annotate sentences that describe housing conditions, stability, or problems including homelessness, temporary housing, housing quality, overcrowding, unsafe living conditions, housing affordability, eviction risk, or housing-related health hazards.

Transportation: Annotate sentences that describe difficulties accessing transportation, lack of reliable transportation, inability to travel for medical appointments, public transit limitations, vehicle problems, or mobility barriers that affect daily activities or healthcare access.

Parental status: Annotate sentences that indicate whether the person has children, has parental responsibilities, custody arrangements, child-rearing challenges, or family composition including dependents.

Relationship status: Annotate sentences that describe marital status (married, divorced, widowed, separated), partnership status (single, dating, cohabiting).

Social support: Annotate sentences that describe availability or absence of help from family, friends, or community including emotional support, practical assistance, social connections, isolation, loneliness.

Substance use: Annotate sentences that mention current or past use of alcohol, illegal drugs, prescription drug misuse, tobacco products, smoking cessation attempts, substance abuse treatment, or substance-related health problems.

Financial Situation: Annotate sentences that describe economic hardship, income adequacy, debt problems, inability to afford necessities, or financial stress.

Education level: Annotate sentences that mention highest level of education completed, literacy skills, educational barriers, special education needs.

Food insecurity: Annotate sentences that describe inadequate food access, hunger, reliance on food assistance programs, poor nutrition due to cost, skipping meals, food scarcity, or difficulty obtaining healthy foods.
"""
        
    else:  # level 2
        task_description = """
You are analyzing a referral note sentence to identify Social Determinants of Health, and classify them as Adverse or Protective.

Given a sentence, output all SDoH factors that can be inferred from that sentence from the following list: 
EmploymentStatus, Housing, Transportation, ParentalStatus, RelationshipStatus, SocialSupport, SubstanceUse, FinancialSituation, EducationLevel, FoodInsecurity. If the sentence does NOT mention any of the above categories, output - NoSDoH.

Your response must be a comma-separated list of SDoH factors embedded with <LIST> and </LIST>.

**STRICT RULES**:
- For EVERY SDoH mention found, you MUST classify it as either "Adverse" or "Protective" after a hyphen
- DO NOT generate any other text, explanations, or new SDoH labels.
- A sentence CAN be labeled with one or more SDoH factors.

Below are some guidelines and examples to follow:

Employment status: Annotate sentences that describe a person's work situation, including current employment, unemployment, retirement status, or disability affecting work capacity.

Housing issues: Annotate sentences that describe housing conditions, stability, or problems including homelessness, temporary housing, housing quality, overcrowding, unsafe living conditions, housing affordability, eviction risk, or housing-related health hazards.

Transportation issues: Annotate sentences that describe difficulties accessing transportation, lack of reliable transportation, inability to travel for medical appointments, public transit limitations, vehicle problems, or mobility barriers that affect daily activities or healthcare access.

Parental status: Annotate sentences that indicate whether the person has children, has parental responsibilities, custody arrangements, child-rearing challenges, or family composition including dependents.

Relationship status: Annotate sentences that describe marital status (married, divorced, widowed, separated), partnership status (single, dating, cohabiting).

Social support: Annotate sentences that describe availability or absence of help from family, friends, or community including emotional support, practical assistance, social connections, isolation, loneliness.

Substance use: Annotate sentences that mention current or past use of alcohol, illegal drugs, prescription drug misuse, tobacco products, smoking cessation attempts, substance abuse treatment, or substance-related health problems.

Financial issues: Annotate sentences that describe economic hardship, income adequacy, debt problems, inability to afford necessities, or financial stress.

Education level: Annotate sentences that mention highest level of education completed, literacy skills, educational barriers, special education needs.

Food insecurity: Annotate sentences that describe inadequate food access, hunger, reliance on food assistance programs, poor nutrition due to cost, skipping meals, food scarcity, or difficulty obtaining healthy foods.
"""
    
    prompt = f"""{task_description}

Input: {sentence}
"""
    
    return prompt

def create_five_shot_basic_prompt(sentence: str, level: int = 1) -> str:
    """5-shot prompt with category list and examples, no detailed descriptions"""
    
    if level == 1:
        task_description = """
You are analyzing a referral note sentence to identify mentions of Social Determinants of Health (SDoH).

Given a sentence, output all SDoH factors that can be inferred from that sentence from the following list: 
EmploymentStatus, Housing, Transportation, ParentalStatus, RelationshipStatus, SocialSupport, SubstanceUse, FinancialSituation, EducationLevel, FoodInsecurity. If the sentence does NOT mention any of the above categories, output - NoSDoH.

Your response must be a comma-separated list of SDoH factors embedded with <LIST> and </LIST>.

**STRICT RULES**: 
- DO NOT generate any other text, explanations, or new SDoH labels.
- A sentence CAN be labeled with one or more SDoH factors.

EXAMPLES:
Example 1: Input: "Person is unemployed and lives with his elderly mother."
Output: <LIST>EmploymentStatus</LIST>

Example 2: Input: "She struggles to afford groceries and has no car to get to the store."
Output: <LIST>FinancialSituation, Transportation, FoodInsecurity</LIST>

Example 3: Input: "Person smokes cigarettes and drinks alcohol daily."
Output: <LIST>SubstanceUse</LIST>

Example 4: Input: "He has three young children at home and receives help from his sister."
Output: <LIST>ParentalStatus, SocialSupport</LIST>

Example 5: Input: "Person was prescribed medication for diabetes."
Output: <LIST>NoSDoH</LIST>

"""
        
    else:  # level 2
        task_description = """
You are analyzing a referral note sentence to identify Social Determinants of Health, and classify them as Adverse or Protective.

Given a sentence, output all SDoH factors that can be inferred from that sentence from the following list: 
EmploymentStatus, Housing, Transportation, ParentalStatus, RelationshipStatus, SocialSupport, SubstanceUse, FinancialSituation, EducationLevel, FoodInsecurity. If the sentence does NOT mention any of the above categories, output - NoSDoH.

Your response must be a comma-separated list of SDoH factors embedded with <LIST> and </LIST>.

**STRICT RULES**:
- For EVERY SDoH mention found, you MUST classify it as either "Adverse" or "Protective" after a hyphen
- DO NOT generate any other text, explanations, or new SDoH labels.
- A sentence CAN be labeled with one or more SDoH factors.

EXAMPLES:
Example 1: Input: "Person is unemployed and lives with his elderly mother."
Output: <LIST>EmploymentStatus-Adverse</LIST>

Example 2: Input: "She struggles to afford groceries and has no car to get to the store."
Output: <LIST>FinancialSituation-Adverse, Transportation-Adverse, FoodInsecurity-Adverse</LIST>

Example 3: Input: "Person smokes cigarettes and drinks alcohol daily."
Output: <LIST>SubstanceUse-Adverse</LIST>

Example 4: Input: "He has three young children at home and receives help from his sister."
Output: <LIST>ParentalStatus-Adverse, SocialSupport-Protective</LIST>

Example 5: Input: "Person was prescribed medication for diabetes."
Output: <LIST>NoSDoH</LIST>
"""
    
    prompt = f"""{task_description}

Input: "{sentence}"
"""
    
    return prompt

def create_five_shot_detailed_prompt(sentence: str, level: int = 1) -> str:
    """5-shot prompt with detailed category descriptions and examples"""
    
    if level == 1:
        task_description = """
You are analyzing a referral note sentence to identify mentions of Social Determinants of Health (SDoH).

Given a sentence, output all SDoH factors that can be inferred from that sentence from the following list: 
EmploymentStatus, Housing, Transportation, ParentalStatus, RelationshipStatus, SocialSupport, SubstanceUse, FinancialSituation, EducationLevel, FoodInsecurity. If the sentence does NOT mention any of the above categories, output - NoSDoH.

Your response must be a comma-separated list of SDoH factors embedded with <LIST> and </LIST>.

**STRICT RULES**: 
- DO NOT generate any other text, explanations, or new SDoH labels.
- A sentence CAN be labeled with one or more SDoH factors.

Below are some guidelines and examples to follow:

Employment status: Annotate sentences that describe a person's work situation, including current employment, unemployment, retirement status, or disability affecting work capacity.

Housing: Annotate sentences that describe housing conditions, stability, or problems including homelessness, temporary housing, housing quality, overcrowding, unsafe living conditions, housing affordability, eviction risk, or housing-related health hazards.

Transportation: Annotate sentences that describe difficulties accessing transportation, lack of reliable transportation, inability to travel for medical appointments, public transit limitations, vehicle problems, or mobility barriers that affect daily activities or healthcare access.

Parental status: Annotate sentences that indicate whether the person has children, has parental responsibilities, custody arrangements, child-rearing challenges, or family composition including dependents.

Relationship status: Annotate sentences that describe marital status (married, divorced, widowed, separated), partnership status (single, dating, cohabiting).

Social support: Annotate sentences that describe availability or absence of help from family, friends, or community including emotional support, practical assistance, social connections, isolation, loneliness.

Substance use: Annotate sentences that mention current or past use of alcohol, illegal drugs, prescription drug misuse, tobacco products, smoking cessation attempts, substance abuse treatment, or substance-related health problems.

Financial Situation: Annotate sentences that describe economic hardship, income adequacy, debt problems, inability to afford necessities, or financial stress.

Education level: Annotate sentences that mention highest level of education completed, literacy skills, educational barriers, special education needs.

Food insecurity: Annotate sentences that describe inadequate food access, hunger, reliance on food assistance programs, poor nutrition due to cost, skipping meals, food scarcity, or difficulty obtaining healthy foods.

EXAMPLES:
Example 1: Input: "Person is unemployed and lives with his elderly mother."
Output: <LIST>EmploymentStatus</LIST>

Example 2: Input: "She struggles to afford groceries and has no car to get to the store."
Output: <LIST>FinancialIssues, TransportationIssues, FoodInsecurity</LIST>

Example 3: Input: "Person smokes cigarettes and drinks alcohol daily."
Output: <LIST>SubstanceUse</LIST>

Example 4: Input: "He has three young children at home and receives help from his sister."
Output: <LIST>ParentalStatus, SocialSupport</LIST>

Example 5: Input: "Person was prescribed medication for diabetes."
Output: <LIST>NoSDoH</LIST>
"""
        
    else:  # level 2
        task_description = """
You are analyzing a referral note sentence to identify Social Determinants of Health, and classify them as Adverse or Protective.

Given a sentence, output all SDoH factors that can be inferred from that sentence from the following list: 
EmploymentStatus, Housing, Transportation, ParentalStatus, RelationshipStatus, SocialSupport, SubstanceUse, FinancialSituation, EducationLevel, FoodInsecurity. If the sentence does NOT mention any of the above categories, output - NoSDoH.

Your response must be a comma-separated list of SDoH factors embedded with <LIST> and </LIST>.

**STRICT RULES**:
- For EVERY SDoH mention found, you MUST classify it as either "Adverse" or "Protective" after a hyphen
- DO NOT generate any other text, explanations, or new SDoH labels.
- A sentence CAN be labeled with one or more SDoH factors.

Below are some guidelines and examples to follow:

Employment status: Annotate sentences that describe a person's work situation, including current employment, unemployment, retirement status, or disability affecting work capacity.

Housing issues: Annotate sentences that describe housing conditions, stability, or problems including homelessness, temporary housing, housing quality, overcrowding, unsafe living conditions, housing affordability, eviction risk, or housing-related health hazards.

Transportation issues: Annotate sentences that describe difficulties accessing transportation, lack of reliable transportation, inability to travel for medical appointments, public transit limitations, vehicle problems, or mobility barriers that affect daily activities or healthcare access.

Parental status: Annotate sentences that indicate whether the person has children, has parental responsibilities, custody arrangements, child-rearing challenges, or family composition including dependents.

Relationship status: Annotate sentences that describe marital status (married, divorced, widowed, separated), partnership status (single, dating, cohabiting).

Social support: Annotate sentences that describe availability or absence of help from family, friends, or community including emotional support, practical assistance, social connections, isolation, loneliness.

Substance use: Annotate sentences that mention current or past use of alcohol, illegal drugs, prescription drug misuse, tobacco products, smoking cessation attempts, substance abuse treatment, or substance-related health problems.

Financial issues: Annotate sentences that describe economic hardship, income adequacy, debt problems, inability to afford necessities, or financial stress.

Education level: Annotate sentences that mention highest level of education completed, literacy skills, educational barriers, special education needs.

Food insecurity: Annotate sentences that describe inadequate food access, hunger, reliance on food assistance programs, poor nutrition due to cost, skipping meals, food scarcity, or difficulty obtaining healthy foods.

EXAMPLES:
Example 1: Input: "Person is unemployed and lives with his elderly mother."
Output: <LIST>EmploymentStatus-Adverse</LIST>

Example 2: Input: "She struggles to afford groceries and has no car to get to the store."
Output: <LIST>FinancialSituation-Adverse, Transportation-Adverse, FoodInsecurity-Adverse</LIST>

Example 3: Input: "Person smokes cigarettes and drinks alcohol daily."
Output: <LIST>SubstanceUse-Adverse</LIST>

Example 4: Input: "He has three young children at home and receives help from his sister."
Output: <LIST>ParentalStatus-Adverse, SocialSupport-Protective</LIST>

Example 5: Input: "Person was prescribed medication for diabetes."
Output: <LIST>NoSDoH</LIST>
"""
    
    prompt = f"""{task_description}

Input: "{sentence}"
"""
    
    return prompt

# def _format_llama_prompt(prompt: str) -> str:
#     """Format prompt for Llama models"""
#     return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    
# def _format_qwen_prompt(prompt: str) -> str:
#     """Format prompt for Qwen models"""
#     return f"""<|im_start|>user{prompt}<|im_end|><|im_start|>assistant"""
    
# def _format_phi_prompt(self, prompt: str) -> str:
#     """Format prompt for Phi models"""
#     return f"""<|user|>{prompt}<|end|><|assistant|>"""
    
# def _format_mistral_prompt(self, prompt: str) -> str:
#     """Format prompt for Mistral models"""
#     return f"""<s>[INST] {prompt} [/INST]"""