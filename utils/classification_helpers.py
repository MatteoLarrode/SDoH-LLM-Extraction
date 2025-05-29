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

def create_classification_prompt(text: str, level: int = 1) -> str:
    """Create a prompt for SDoH classification"""
    
    if level == 1:
        task_description = """
You are analyzing clinical referral notes to identify mentions of Social Determinants of Health (SDoH).

For each sentence, identify if it mentions ANY of these 6 categories (regardless of whether positive or negative):
1. Employment status - any mention of work, job, career, employment situation
2. Housing issues - any mention of housing, living situation, accommodation
3. Transportation issues - any mention of transportation, travel, getting around
4. Parental status - any mention of children, parenting, being a parent
5. Relationship status - any mention of marital status, partnerships, relationships
6. Social support - any mention of help from others, family support, social connections

Return your analysis as a JSON object with this exact format:
{
  "sentences": [
    {
      "sentence": "exact sentence text",
      "categories": ["list of matching categories from above"]
    }
  ]
}
"""
    else:  # level 2
        task_description = """
You are analyzing clinical referral notes to identify ADVERSE Social Determinants of Health that indicate additional support needs.

For each sentence, identify if it mentions ANY of these ADVERSE situations:
1. Employment status - unemployed, underemployed, work disability, job loss
2. Housing issues - homelessness, housing insecurity, poor conditions, affordability problems
3. Transportation issues - lack of transportation, distance barriers, mobility problems
4. Parental status - having children under 18 years old requiring care
5. Relationship status - widowed, divorced, single, separated (relationship loss/absence)
6. Social support - isolation, lack of support, being alone, no help available

Return your analysis as a JSON object with this exact format:
{
  "sentences": [
    {
      "sentence": "exact sentence text", 
      "categories": ["list of matching ADVERSE categories from above"]
    }
  ]
}
"""

    prompt = f"""{task_description}

Text to analyze:
"{text}"

Analysis:"""
    
    return prompt

def create_summary(results: Dict, level: int) -> Dict:
    """Create a summary of the classification results"""
    
    all_categories = set()
    sentence_count = 0
    
    for sentence_result in results.get('sentences', []):
        if sentence_result.get('categories'):
            all_categories.update(sentence_result['categories'])
            sentence_count += 1
    
    return {
        'total_sentences_analyzed': len(results.get('sentences', [])),
        'sentences_with_sdoh': sentence_count,
        'unique_categories_found': list(all_categories),
        'category_count': len(all_categories),
        'level': level
    }

def parse_llm_response(response: str, original_text: str, level: int) -> Dict:
    """Parse the LLM response and extract structured results"""
    
    try:
        # Try to extract JSON from response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            parsed = json.loads(json_str)
            
            # Validate and clean the parsed results
            if 'sentences' in parsed and isinstance(parsed['sentences'], list):
                return {
                    'success': True,
                    'results': parsed,
                    'summary': create_summary(parsed, level)
                }
    
    except json.JSONDecodeError:
        pass
    
    # Fallback: try to extract information from free text response
    return {
        'success': False,
        'results': None,
        'fallback_analysis': extract_from_free_text(response, original_text, level),
        'raw_response': response
    }

def extract_from_free_text(response: str, text: str, level: int) -> Dict:
    """Fallback extraction when JSON parsing fails"""
    
    sentences = split_into_sentences(text)
    categories = ["Employment status", "Housing issues", "Transportation issues", 
                 "Parental status", "Relationship status", "Social support"]
    
    found_categories = []
    for category in categories:
        if category.lower() in response.lower():
            found_categories.append(category)
    
    return {
        'method': 'fallback_text_analysis',
        'sentences_analyzed': len(sentences),
        'categories_mentioned': found_categories,
        'raw_response_snippet': response[:200] + "..." if len(response) > 200 else response
    }

def classify_with_llm(text: str, tokenizer, model, level: int = 1) -> Dict:
    """Classify text using instruction-tuned LLM"""
    
    # Create the prompt
    prompt = create_classification_prompt(text, level)
    
    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    )
    
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1000,
            temperature=0.1,  # Low temperature for consistent results
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the generated part (after the prompt)
    response = full_response[len(tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)):]
    
    return {
        'prompt': prompt,
        'raw_response': response,
        'parsed_results': parse_llm_response(response, text, level)
    }