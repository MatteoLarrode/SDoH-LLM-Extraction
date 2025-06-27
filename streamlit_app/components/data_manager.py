"""
Data management for the Streamlit app
"""

import streamlit as st
import pandas as pd
from typing import Tuple, Optional

class DataManager:
    """Handles data loading and management"""
    
    @st.cache_data
    def load_referral_data(_self) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Load the BRC referrals data"""
        try:
            df = pd.read_csv("data/processed/BRC-Cleaned/referrals_cleaned.csv")
            
            # Find the referral notes column
            note_columns = [col for col in df.columns if 'referral' in col.lower() and 'note' in col.lower()]
            if not note_columns:
                st.error("No referral notes column found in the dataset")
                return None, None
            
            note_column = note_columns[0]
            return df, note_column
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None, None
    
    @staticmethod
    def get_dataset_stats(df: pd.DataFrame, note_column: str) -> dict:
        """Get dataset statistics"""
        return {
            'total_notes': len(df),
            'notes_with_text': df[note_column].notna().sum(),
            'avg_length': df[note_column].str.len().mean(),
            'median_length': df[note_column].str.len().median(),
            'min_length': df[note_column].str.len().min(),
            'max_length': df[note_column].str.len().max()
        }
    
    @staticmethod
    def search_notes(df: pd.DataFrame, note_column: str, search_term: str, 
                    min_length: int = 0, max_length: int = 10000) -> pd.DataFrame:
        """Search and filter notes"""
        filtered_df = df.copy()
        
        if search_term:
            filtered_df = filtered_df[
                filtered_df[note_column].str.contains(search_term, case=False, na=False)
            ]
        
        if min_length > 0:
            filtered_df = filtered_df[
                filtered_df[note_column].str.len() >= min_length
            ]
        
        if max_length < 10000:
            filtered_df = filtered_df[
                filtered_df[note_column].str.len() <= max_length
            ]
        
        return filtered_df
    
    @staticmethod
    def get_notes_by_length_category(df: pd.DataFrame, note_column: str, category: str) -> pd.DataFrame:
        """Filter notes by length category"""
        lengths = df[note_column].str.len()
        
        if category == "Very Short (< 50 chars)":
            return df[lengths < 50]
        elif category == "Short (50-200 chars)":
            return df[(lengths >= 50) & (lengths < 200)]
        elif category == "Medium (200-500 chars)":
            return df[(lengths >= 200) & (lengths < 500)]
        elif category == "Long (500-1000 chars)":
            return df[(lengths >= 500) & (lengths < 1000)]
        else:  # Very Long
            return df[lengths >= 1000]
    
    @staticmethod
    def paginate_dataframe(df: pd.DataFrame, page_size: int, current_page: int) -> Tuple[pd.DataFrame, dict]:
        """Paginate dataframe and return pagination info"""
        total_pages = (len(df) - 1) // page_size + 1
        start_idx = (current_page - 1) * page_size
        end_idx = min(start_idx + page_size, len(df))
        
        page_df = df.iloc[start_idx:end_idx]
        
        pagination_info = {
            'total_pages': total_pages,
            'current_page': current_page,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'total_items': len(df)
        }
        
        return page_df, pagination_info