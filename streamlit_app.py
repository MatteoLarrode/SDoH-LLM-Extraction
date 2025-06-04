"""
Main Streamlit App for SDoH Extraction Tool
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
sys.path.append('.')

# Import components
from streamlit_components.config import configure_page, load_sidebar_config
from streamlit_components.data_manager import DataManager
from streamlit_components.model_manager import ModelManager
from streamlit_components.tabs import (
    single_analysis_tab,
    dataset_browser_tab,
    batch_processing_tab,
    results_analysis_tab
)

def main():
    """Main application function"""
    
    # Configure page
    configure_page()
    
    # Initialize managers
    data_manager = DataManager()
    model_manager = ModelManager()
    
    # Load data
    df, note_column = data_manager.load_referral_data()
    if df is None:
        st.stop()
    
    # Sidebar configuration
    config = load_sidebar_config(note_column)
    
    # Load model based on config
    tokenizer, model = model_manager.get_model(config['selected_model'])
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 Single Text Analysis", 
        "📊 Browse Dataset", 
        "⚡ Batch Processing", 
        "📈 Results Analysis"
    ])
    
    with tab1:
        single_analysis_tab(df, note_column, tokenizer, model, config)
    
    with tab2:
        dataset_browser_tab(df, note_column, tokenizer, model, config)
    
    with tab3:
        batch_processing_tab(df, note_column, tokenizer, model, config)
    
    with tab4:
        results_analysis_tab()

if __name__ == "__main__":
    main()