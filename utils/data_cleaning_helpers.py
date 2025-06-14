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
            r'^none$',           # none
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

def clean_referrals_dataset(df):
    """
    Clean referrals dataset by removing identical rows and consolidating cases
    with multiple observations while preserving information about observation
    patterns and date ranges.
    """
    
    print("=== REFERRALS DATASET PRE-CLEANING ===")
    print(f"Initial dataset: {len(df):,} rows, {df['case_ref'].nunique():,} unique cases")
    
    # STEP 0: Apply NA cleaning to all text columns
    print("\nStep 0: Cleaning NA variations in text columns...")
    text_columns = df.select_dtypes(include=['object']).columns
    
    for col in text_columns:
        original_na_count = df[col].isna().sum()
        df[col] = df[col].apply(clean_na_variations)
        new_na_count = df[col].isna().sum()
        additional_na = new_na_count - original_na_count
        if additional_na > 0:
            print(f"  {col}: {additional_na:,} additional NAs identified")
    
    # Step 1: Remove identical rows based on specified columns
    print("\nStep 1: Removing identical rows...")
    dedup_cols = ['case_ref', 'Referral Notes (depersonalised)', 'Referral Date/Time']
    
    # Check which columns actually exist
    available_dedup_cols = [col for col in dedup_cols if col in df.columns]
    print(f"  Using columns: {available_dedup_cols}")
    
    initial_rows = len(df)
    df_clean = df.drop_duplicates(subset=available_dedup_cols, keep='first').copy()  # Added .copy()
    removed_identical = initial_rows - len(df_clean)
    print(f"  Removed {removed_identical:,} identical rows")
    print(f"  Remaining: {len(df_clean):,} rows, {df_clean['case_ref'].nunique():,} unique cases")
    
    # Step 2: Process cases with multiple rows
    print("\nStep 2: Processing cases with multiple observations...")
    
    # Convert date column to datetime if it exists
    date_col = 'Referral Date/Time' if 'Referral Date/Time' in df_clean.columns else None
    if date_col:
        df_clean[date_col] = pd.to_datetime(df_clean[date_col], errors='coerce')
    
    # Group by case_ref and create summary statistics
    case_groups = df_clean.groupby('case_ref')
    
    # Create summary information for each case
    case_summary = []
    for case_ref, group in case_groups:
        summary = {
            'case_ref': case_ref,
            'num_observations': len(group),
            'has_referral_notes': group['Referral Notes (depersonalised)'].notna().any() if 'Referral Notes (depersonalised)' in group.columns else False
        }
        
        # Add date range information if date column exists
        if date_col and group[date_col].notna().any():
            valid_dates = group[date_col].dropna()
            if len(valid_dates) > 0:
                summary['date_range_start'] = valid_dates.min()
                summary['date_range_end'] = valid_dates.max()
                summary['date_range_days'] = (valid_dates.max() - valid_dates.min()).days if len(valid_dates) > 1 else 0
            else:
                summary['date_range_start'] = None
                summary['date_range_end'] = None
                summary['date_range_days'] = None
        
        case_summary.append(summary)
    
    case_summary_df = pd.DataFrame(case_summary)
    
    # Step 2a: Add observation count and date range columns to original data
    print("  Step 2a: Adding observation count and date range columns...")
    
    df_with_summary = df_clean.merge(
        case_summary_df[['case_ref', 'num_observations', 'date_range_start', 'date_range_end', 'date_range_days']], 
        on='case_ref', 
        how='left'
    )
    
    # Step 2b: For cases with multiple rows, remove rows without referral notes if others have them
    print("  Step 2b: Removing rows without referral notes when others exist...")
    
    def filter_group(group):
        if len(group) == 1:
            return group
        
        # Check if referral notes column exists
        if 'Referral Notes (depersonalised)' not in group.columns:
            # If no referral notes column, just keep most recent
            if date_col and group[date_col].notna().any():
                return group.loc[group[date_col].idxmax()].to_frame().T
            else:
                return group.tail(1)
        
        has_notes = group['Referral Notes (depersonalised)'].notna()
        
        if has_notes.any() and not has_notes.all():
            # Some have notes, some don't - keep only those with notes
            return group[has_notes]
        else:
            # Either all have notes or none have notes - continue to next step
            return group
    
    df_filtered = df_with_summary.groupby('case_ref').apply(filter_group).reset_index(drop=True)
    removed_no_notes = len(df_with_summary) - len(df_filtered)
    print(f"    Removed {removed_no_notes:,} rows without referral notes")
    
    # Step 2c: For remaining cases with multiple rows, check if referral notes are identical
    print("  Step 2c: Consolidating cases with identical or missing referral notes...")
    
    def consolidate_group(group):
        if len(group) == 1:
            return group
        
        # Check referral notes similarity
        if 'Referral Notes (depersonalised)' in group.columns:
            unique_notes = group['Referral Notes (depersonalised)'].dropna().nunique()
            all_notes_na = group['Referral Notes (depersonalised)'].isna().all()
            
            if unique_notes <= 1 or all_notes_na:
                # All notes are the same or all are missing - keep most recent
                if date_col and group[date_col].notna().any():
                    return group.loc[group[date_col].idxmax()].to_frame().T
                else:
                    return group.tail(1)
            else:
                # Different notes - keep all rows
                return group
        else:
            # No referral notes column - keep most recent
            if date_col and group[date_col].notna().any():
                return group.loc[group[date_col].idxmax()].to_frame().T
            else:
                return group.tail(1)
    
    df_final = df_filtered.groupby('case_ref').apply(consolidate_group).reset_index(drop=True)
    removed_identical_notes = len(df_filtered) - len(df_final)
    print(f"    Consolidated {removed_identical_notes:,} rows with identical/missing referral notes")
    
    # Final summary
    print(f"\nFinal dataset: {len(df_final):,} rows, {df_final['case_ref'].nunique():,} unique cases")
    
    # Show cases that still have multiple rows (different referrals on different dates)
    remaining_duplicates = df_final['case_ref'].value_counts()
    cases_with_multiple_rows = remaining_duplicates[remaining_duplicates > 1]
    
    print(f"\nCases with multiple rows remaining (different referrals): {len(cases_with_multiple_rows):,}")
    if len(cases_with_multiple_rows) > 0:
        print("Top 5 cases by number of remaining rows:")
        for case_ref, count in cases_with_multiple_rows.head().items():
            print(f"  {case_ref}: {count} rows")
            
        # Show example of remaining duplicates
        example_case = cases_with_multiple_rows.index[0]
        example_rows = df_final[df_final['case_ref'] == example_case]
        print(f"\nExample case {example_case} with different referrals:")
        for idx, row in example_rows.iterrows():
            date_str = row[date_col].strftime('%Y-%m-%d') if date_col and pd.notna(row[date_col]) else "No date"
            note_preview = str(row['Referral Notes (depersonalised)'])[:100] + "..." if pd.notna(row['Referral Notes (depersonalised)']) else "No referral note"
            print(f"  {date_str}: {note_preview}")
    
    return df_final

def clean_snap_dataset(df):
    """
    Clean SNAP dataset focusing on baseline-outcome pairs and removing
    incomplete or duplicate assessments.
    """
    
    print("=== SNAP DATASET PRE-CLEANING ===")
    print(f"Initial dataset: {len(df):,} rows, {df['case_ref'].nunique():,} unique cases")
    
    # STEP 0: Apply NA cleaning to all text columns
    print("\nStep 0: Cleaning NA variations in text columns...")
    text_columns = df.select_dtypes(include=['object']).columns
    
    for col in text_columns:
        df[col] = df[col].apply(clean_na_variations)
    
    # STEP 1: Remove perfect duplicates
    print("\nStep 1: Removing perfect duplicates...")
    # Exclude depersonalized columns from duplicate detection
    exclude_cols = ['Has Disability', 'IMD Decile', 'Country', 'Age', 'Gender', 'Ethnicity', 'Living Arrangements']
    analysis_cols = [col for col in df.columns if col not in exclude_cols]
    
    initial_rows = len(df)
    df_clean = df.drop_duplicates(subset=analysis_cols, keep='first').copy()
    removed_duplicates = initial_rows - len(df_clean)
    print(f"  Removed {removed_duplicates:,} perfect duplicates")
    
    # STEP 2: Identify valid baseline and valid post-support pairs
    print("\nStep 2: Identifying valid baseline and valid post-support assessments...")
    
    # Group by case and check for paired data
    case_summary = []
    for case_ref, group in df_clean.groupby('case_ref'):
        summary = {
            'case_ref': case_ref,
            'num_assessments': len(group),
            'has_valid_baseline': False,
            'has_valid_post_support': False,
            'has_both': False
        }
        
        # Check for valid baseline (timepoint 1 or "at the start" with valid outcomes)
        if 'Timepoint' in group.columns:
            baseline_rows = group[group['Timepoint'] == 1.0]
            if len(baseline_rows) > 0 and 'Possible to record outcomes:' in group.columns:
                valid_baseline = baseline_rows[
                    baseline_rows['Possible to record outcomes:'].str.lower().isin(['yes', 'y'])
                ]
                summary['has_valid_baseline'] = len(valid_baseline) > 0
            
            # Check for valid post-support (timepoint 2 with valid outcomes)
            post_support_rows = group[group['Timepoint'] == 2.0]
            if len(post_support_rows) > 0 and 'Possible to record outcomes:' in group.columns:
                valid_post_support = post_support_rows[
                    post_support_rows['Possible to record outcomes:'].str.lower().isin(['yes', 'y'])
                ]
                summary['has_valid_post_support'] = len(valid_post_support) > 0
        elif 'Survey completed:' in group.columns:
            # Alternative check using survey completion field
            baseline_rows = group[group['Survey completed:'] == 'at the start of support']
            if len(baseline_rows) > 0 and 'Possible to record outcomes:' in group.columns:
                valid_baseline = baseline_rows[
                    baseline_rows['Possible to record outcomes:'].str.lower().isin(['yes', 'y'])
                ]
                summary['has_valid_baseline'] = len(valid_baseline) > 0
            
            post_support_rows = group[group['Survey completed:'] == 'at the end of support']
            if len(post_support_rows) > 0 and 'Possible to record outcomes:' in group.columns:
                valid_post_support = post_support_rows[
                    post_support_rows['Possible to record outcomes:'].str.lower().isin(['yes', 'y'])
                ]
                summary['has_valid_post_support'] = len(valid_post_support) > 0
        
        # Check if has both valid baseline and valid post-support
        summary['has_both'] = summary['has_valid_baseline'] and summary['has_valid_post_support']
        
        case_summary.append(summary)
    
    case_summary_df = pd.DataFrame(case_summary)
    
    # Report on data structure
    has_valid_baseline = case_summary_df['has_valid_baseline'].sum()
    has_valid_post = case_summary_df['has_valid_post_support'].sum()
    has_both = case_summary_df['has_both'].sum()
    baseline_only = case_summary_df['has_valid_baseline'].sum() - has_both
    
    print(f"  Cases with valid baseline: {has_valid_baseline:,}")
    print(f"  Cases with valid post-support: {has_valid_post:,}")
    print(f"  Cases with both (complete valid pairs): {has_both:,}")
    print(f"  Cases with valid baseline only: {baseline_only:,}")
    
    # STEP 3: Add metadata columns
    print("\nStep 3: Adding metadata columns...")
    df_with_metadata = df_clean.merge(
        case_summary_df[['case_ref', 'num_assessments', 'has_valid_baseline', 'has_valid_post_support', 'has_both']], 
        on='case_ref', 
        how='left'
    )
    
    # Add date range if date column exists
    if 'Date of assessment' in df_clean.columns:
        df_with_metadata['Date of assessment'] = pd.to_datetime(
            df_with_metadata['Date of assessment'], 
            format='%d/%m/%Y',  # Specify the format explicitly
            errors='coerce'
        )
        
        # Calculate date ranges for each case
        date_ranges = df_with_metadata.groupby('case_ref')['Date of assessment'].agg([
            ('first_assessment', 'min'),
            ('last_assessment', 'max'),
            ('assessment_span_days', lambda x: (x.max() - x.min()).days if x.notna().sum() > 1 else 0)
        ]).reset_index()
        
        df_with_metadata = df_with_metadata.merge(date_ranges, on='case_ref', how='left')
    
    print(f"\nFinal dataset: {len(df_with_metadata):,} rows, {df_with_metadata['case_ref'].nunique():,} unique cases")
    
    return df_with_metadata

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