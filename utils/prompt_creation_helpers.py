# ================================================
# ==== Helper functions for prompt generation ====
# ================================================
from typing import List, Dict

def create_zero_shot_basic_prompt(sentence: str, level: int = 1) -> str:
    """Zero-shot prompt with just category list, no descriptions"""
    
    if level == 1:
        task_description = """
You are analyzing a referral note sentence to identify mentions of Social Determinants of Health (SDoH).

Identify if the sentence mentions ANY of these categories:
Employment status, Housing issues, Transportation issues, Parental status, Relationship status, Social support, Substance use, Financial issues, Education level, Food insecurity

If the sentence does NOT mention any of the above categories, return: "NoSDoH"

STRICT RULES: 
- You MUST only use the exact categories listed above OR "NoSDoH"
- DO NOT add explanations or notes
- Return ONLY the list format specified below
"""
        
    else:  # level 2
        task_description = """
You are analyzing a referral note sentence to identify Social Determinants of Health and classify them as adverse or non-adverse.

Identify if the sentence mentions ANY of these categories and classify each as ADVERSE or NON-ADVERSE:
Employment status, Housing issues, Transportation issues, Parental status, Relationship status, Social support, Substance use, Financial issues, Education level, Food insecurity

If the sentence does NOT mention any of the above categories, return: "NoSDoH"

STRICT RULES:
- For EVERY SDoH mention found, you MUST classify it as either "[adverse]" or "[non-adverse]"
- You MUST only use the exact categories listed above OR "NoSDoH"
- DO NOT add explanations or notes
- Return ONLY the list format specified below
"""
    
    prompt = f"""{task_description}

SENTENCE TO ANALYZE:
{sentence}

Response must be a comma-separated list of SDoH factors embedded with <LIST> and </LIST>.

<LIST>"""
    
    return prompt


def create_zero_shot_detailed_prompt(sentence: str, level: int = 1) -> str:
    """Zero-shot prompt with detailed category descriptions"""
    
    if level == 1:
        task_description = """
You are analyzing a referral note sentence to identify mentions of Social Determinants of Health (SDoH).

For the sentence below, identify if it mentions ANY of these 10 categories:
1. Employment status - any mention of work, job, career, employment situation
2. Housing issues - any mention of housing, living situation, accommodation
3. Transportation issues - any mention of transportation, travel, getting around
4. Parental status - any mention of children, parenting, being a parent
5. Relationship status - any mention of marital status, partnerships, relationships
6. Social support - any mention of help from others, family support, social connections
7. Substance use - any mention of alcohol, drugs, smoking, tobacco use
8. Financial issues - any mention of money problems, income, financial struggles, benefits
9. Education level - any mention of schooling, education, qualifications, literacy
10. Food insecurity - any mention of food access, hunger, food assistance, nutrition problems

If the sentence does NOT mention any of the above categories, return: "NoSDoH"

STRICT RULES: 
- You MUST only use the exact categories listed above OR "NoSDoH"
- DO NOT add explanations or notes
- Return ONLY the list format specified below
"""
        
    else:  # level 2
        task_description = """
You are analyzing a referral note sentence to identify Social Determinants of Health and classify them as adverse or non-adverse.

For the sentence below, identify if it mentions ANY of these 10 categories and classify each as ADVERSE or NON-ADVERSE:

1. Employment status - unemployed, underemployed, work disability, job loss, unstable employment
2. Housing issues - homelessness, housing insecurity, poor conditions, affordability problems, overcrowding
3. Transportation issues - lack of transportation, distance barriers, mobility problems, transport costs
4. Parental status - having children under 18 years old requiring care, single parenting challenges
5. Relationship status - widowed, divorced, single, separated (relationship loss/absence)
6. Social support - isolation, lack of support, being alone, no help available, family conflict
7. Substance use - alcohol abuse, drug use, smoking addiction, substance dependency
8. Financial issues - poverty, debt, inability to pay bills, benefit dependency, financial stress
9. Education level - low education, illiteracy, lack of qualifications, educational barriers
10. Food insecurity - hunger, inability to afford food, reliance on food banks, poor nutrition

If the sentence does NOT mention any of the above categories, return: "NoSDoH"

STRICT RULES:
- For EVERY SDoH mention found, you MUST classify it as either "[adverse]" or "[non-adverse]"
- You MUST only use the exact categories listed above OR "NoSDoH"
- DO NOT add explanations or notes
- Return ONLY the list format specified below
"""
    
    prompt = f"""{task_description}

SENTENCE TO ANALYZE:
{sentence}

Response must be a comma-separated list of SDoH factors embedded with <LIST> and </LIST>.

<LIST>"""
    
    return prompt


def create_five_shot_basic_prompt(sentence: str, level: int = 1) -> str:
    """5-shot prompt with category list and examples, no detailed descriptions"""
    
    if level == 1:
        task_description = """
You are analyzing a referral note sentence to identify mentions of Social Determinants of Health (SDoH).

Identify if the sentence mentions ANY of these categories:
Employment status, Housing issues, Transportation issues, Parental status, Relationship status, Social support, Substance use, Financial issues, Education level, Food insecurity

If the sentence does NOT mention any of the above categories, return: "NoSDoH"

EXAMPLES:
Sentence: "John is unemployed and lives with his elderly mother."
<LIST>Employment status, Relationship status</LIST>

Sentence: "She struggles to afford groceries and has no car to get to the store."
<LIST>Financial issues, Transportation issues, Food insecurity</LIST>

Sentence: "The patient smokes cigarettes and drinks alcohol daily."
<LIST>Substance use</LIST>

Sentence: "He has three young children at home and receives help from his sister."
<LIST>Parental status, Social support</LIST>

Sentence: "The patient was prescribed medication for diabetes."
<LIST>NoSDoH</LIST>

STRICT RULES: 
- You MUST only use the exact categories listed above OR "NoSDoH"
- DO NOT add explanations or notes
- Return ONLY the list format specified below
"""
        
    else:  # level 2
        task_description = """
You are analyzing a referral note sentence to identify Social Determinants of Health and classify them as adverse or non-adverse.

Identify if the sentence mentions ANY of these categories and classify each as ADVERSE or NON-ADVERSE:
Employment status, Housing issues, Transportation issues, Parental status, Relationship status, Social support, Substance use, Financial issues, Education level, Food insecurity

If the sentence does NOT mention any of the above categories, return: "NoSDoH"

EXAMPLES:
Sentence: "John is unemployed and lives with his elderly mother."
<LIST>Employment status[adverse], Relationship status[non-adverse]</LIST>

Sentence: "She struggles to afford groceries and has no car to get to the store."
<LIST>Financial issues[adverse], Transportation issues[adverse], Food insecurity[adverse]</LIST>

Sentence: "The patient smokes cigarettes and drinks alcohol daily."
<LIST>Substance use[adverse]</LIST>

Sentence: "He has three young children at home and receives help from his sister."
<LIST>Parental status[non-adverse], Social support[non-adverse]</LIST>

Sentence: "The patient was prescribed medication for diabetes."
<LIST>NoSDoH</LIST>

STRICT RULES:
- For EVERY SDoH mention found, you MUST classify it as either "[adverse]" or "[non-adverse]"
- You MUST only use the exact categories listed above OR "NoSDoH"
- DO NOT add explanations or notes
- Return ONLY the list format specified below
"""
    
    prompt = f"""{task_description}

SENTENCE TO ANALYZE:
{sentence}

Response must be a comma-separated list of SDoH factors embedded with <LIST> and </LIST>.

<LIST>"""
    
    return prompt


def create_five_shot_detailed_prompt(sentence: str, level: int = 1) -> str:
    """5-shot prompt with detailed category descriptions and examples"""
    
    if level == 1:
        task_description = """
You are analyzing a referral note sentence to identify mentions of Social Determinants of Health (SDoH).

For the sentence below, identify if it mentions ANY of these 10 categories:
1. Employment status - any mention of work, job, career, employment situation
2. Housing issues - any mention of housing, living situation, accommodation
3. Transportation issues - any mention of transportation, travel, getting around
4. Parental status - any mention of children, parenting, being a parent
5. Relationship status - any mention of marital status, partnerships, relationships
6. Social support - any mention of help from others, family support, social connections
7. Substance use - any mention of alcohol, drugs, smoking, tobacco use
8. Financial issues - any mention of money problems, income, financial struggles, benefits
9. Education level - any mention of schooling, education, qualifications, literacy
10. Food insecurity - any mention of food access, hunger, food assistance, nutrition problems

If the sentence does NOT mention any of the above categories, return: "NoSDoH"

EXAMPLES:
Sentence: "John is unemployed and lives with his elderly mother."
<LIST>Employment status, Relationship status</LIST>

Sentence: "She struggles to afford groceries and has no car to get to the store."
<LIST>Financial issues, Transportation issues, Food insecurity</LIST>

Sentence: "The patient smokes cigarettes and drinks alcohol daily."
<LIST>Substance use</LIST>

Sentence: "He has three young children at home and receives help from his sister."
<LIST>Parental status, Social support</LIST>

Sentence: "The patient was prescribed medication for diabetes."
<LIST>NoSDoH</LIST>

STRICT RULES: 
- You MUST only use the exact categories listed above OR "NoSDoH"
- DO NOT add explanations or notes
- Return ONLY the list format specified below
"""
        
    else:  # level 2
        task_description = """
You are analyzing a referral note sentence to identify Social Determinants of Health and classify them as adverse or non-adverse.

For the sentence below, identify if it mentions ANY of these 10 categories and classify each as ADVERSE or NON-ADVERSE:

1. Employment status - unemployed, underemployed, work disability, job loss, unstable employment
2. Housing issues - homelessness, housing insecurity, poor conditions, affordability problems, overcrowding
3. Transportation issues - lack of transportation, distance barriers, mobility problems, transport costs
4. Parental status - having children under 18 years old requiring care, single parenting challenges
5. Relationship status - widowed, divorced, single, separated (relationship loss/absence)
6. Social support - isolation, lack of support, being alone, no help available, family conflict
7. Substance use - alcohol abuse, drug use, smoking addiction, substance dependency
8. Financial issues - poverty, debt, inability to pay bills, benefit dependency, financial stress
9. Education level - low education, illiteracy, lack of qualifications, educational barriers
10. Food insecurity - hunger, inability to afford food, reliance on food banks, poor nutrition

If the sentence does NOT mention any of the above categories, return: "NoSDoH"

EXAMPLES:
Sentence: "John is unemployed and lives with his elderly mother."
<LIST>Employment status[adverse], Relationship status[non-adverse]</LIST>

Sentence: "She struggles to afford groceries and has no car to get to the store."
<LIST>Financial issues[adverse], Transportation issues[adverse], Food insecurity[adverse]</LIST>

Sentence: "The patient smokes cigarettes and drinks alcohol daily."
<LIST>Substance use[adverse]</LIST>

Sentence: "He has three young children at home and receives help from his sister."
<LIST>Parental status[non-adverse], Social support[non-adverse]</LIST>

Sentence: "The patient was prescribed medication for diabetes."
<LIST>NoSDoH</LIST>

STRICT RULES:
- For EVERY SDoH mention found, you MUST classify it as either "[adverse]" or "[non-adverse]"
- You MUST only use the exact categories listed above OR "NoSDoH"
- DO NOT add explanations or notes
- Return ONLY the list format specified below
"""
    
    prompt = f"""{task_description}

SENTENCE TO ANALYZE:
{sentence}

Response must be a comma-separated list of SDoH factors embedded with <LIST> and </LIST>.

<LIST>"""
    
    return prompt