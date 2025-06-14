# ============================================
# ==== Helper functions for data cleaning ====
# ============================================
import pandas as pd
import numpy as np
import re
from typing import List

def clean_na_variations(text):
        """Remove various NA representations and normalize text"""
        if pd.isna(text):
            return np.nan
        
        # Convert to string
        text = str(text).strip()
        
        # Define NA variations (case-insensitive)
        na_patterns = [
            r'^n/?a$',           # n/a, na, N/A, NA
            r'^not applicable$',  # not applicable
            r'^none$',           # none
            r'^nil$',            # nil
            r'^-+$',             # dashes like -, --, ---
            r'^\.+$',            # dots like ., .., ...
            r'^null$',           # null
            r'^empty$',          # empty
            r'^blank$',          # blank
            r'^\s*$'             # whitespace only
        ]
        
        # Check if text matches any NA pattern
        for pattern in na_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return np.nan
        
        return text

def remove_duplicate_sentences_per_case(df):
        """
        Remove duplicate sentences within each case reference.
        As was done in Keloth et al. (2025).
        """
        print("Removing duplicate sentences within cases...")
        
        # Group by Case Reference
        grouped = df.groupby('Case Reference')
        
        cleaned_rows = []
        total_original_sentences = 0
        total_unique_sentences = 0
        
        for case_ref, group in grouped:
            # Collect all sentences for this case
            all_sentences = []
            
            for idx, row in group.iterrows():
                sentences = split_into_sentences(row['Referral Notes (depersonalised)'])
                all_sentences.extend(sentences)
            
            total_original_sentences += len(all_sentences)
            
            # Remove duplicates while preserving order
            unique_sentences = []
            seen_sentences = set()
            
            for sentence in all_sentences:
                # Normalize sentence for comparison (lowercase, remove extra whitespace)
                normalized = re.sub(r'\s+', ' ', sentence.lower().strip())
                
                if normalized not in seen_sentences and len(normalized) > 0:
                    seen_sentences.add(normalized)
                    unique_sentences.append(sentence)
            
            total_unique_sentences += len(unique_sentences)
            
            # If we have unique sentences, create new rows
            if unique_sentences:
                # Take the first row as template
                template_row = group.iloc[0].copy()
                
                # Combine unique sentences back into referral notes
                combined_notes = '. '.join(unique_sentences)
                if not combined_notes.endswith('.'):
                    combined_notes += '.'
                
                template_row['Referral Notes (depersonalised)'] = combined_notes
                cleaned_rows.append(template_row)
        
        print(f"Original sentences: {total_original_sentences}")
        print(f"Unique sentences: {total_unique_sentences}")
        print(f"Sentences removed: {total_original_sentences - total_unique_sentences}")
        
        # Create new dataframe
        if cleaned_rows:
            result_df = pd.DataFrame(cleaned_rows)
            return result_df.reset_index(drop=True)
        else:
            return pd.DataFrame()

def clean_text(text: str) -> str:
        """Basic text cleaning"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Handle anonymization markers
        text = re.sub(r'\bXXXX\b', '[REDACTED]', text)
        text = re.sub(r'\bPERSON\b', '[PERSON]', text)
        
        return text

def split_into_sentences(text: str) -> List[str]:
        """
        Split text into sentences with basic cleaning
        
        Args:
            text: Input text
            
        Returns:
            List of cleaned sentences
        """
        # Basic sentence splitting
        sentences = re.split(r'[.!?]+', text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            # Remove extra whitespace and anonymization markers
            cleaned = clean_text(sentence)
            
            # Skip very short sentences (likely fragments)
            if len(cleaned.split()) > 2:
                cleaned_sentences.append(cleaned)
        
        return cleaned_sentences

def get_note_length_category(note):
    """
    Categorizes a given text note based on its word count.

    Parameters:
    notes (str or any): The input note to analyze. Can be a string or any value.
                        If the input is NaN or an empty string, it is treated as 'No note'.

    Returns:
    str: A category describing the note:
         - 'No note': If the input is NaN or an empty string.
         - 'Short note (<5 words)': If the note contains fewer than 5 words.
         - 'Medium note (5-19 words)': If the note contains between 5 and 19 words.
         - 'Long note (20+ words)': If the note contains 20 or more words.

    Notes:
    - The function converts the input to a string before processing.
    - Leading and trailing whitespace in the input are ignored.
    """
    if pd.isna(note) or str(note).strip() == '':
        return 'No note'
    word_count = len(str(note).split())
    if word_count < 5:
        return 'Short note (<5 words)'
    elif word_count < 20:
        return 'Medium note (5-19 words)'
    else:
        return 'Long note (20+ words)'