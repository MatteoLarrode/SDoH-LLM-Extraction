"""
Display utilities for the Streamlit app
"""

import streamlit as st
from typing import Dict, Any

def display_extraction_result(result, show_debug=False):
    """Display extraction results in a nice format"""
    if isinstance(result, dict) and "sdoh_factors" in result:
        # Single sentence result
        factors = result["sdoh_factors"]
        
        if factors == ["NoSDoH"]:
            st.markdown('<div class="sdoh-factor no-sdoh">No SDoH Factors</div>', unsafe_allow_html=True)
        else:
            for factor in factors:
                st.markdown(f'<div class="sdoh-factor">{factor}</div>', unsafe_allow_html=True)
        
        if show_debug and "debug" in result:
            with st.expander("🔍 Debug Information"):
                st.text_area("Prompt", result["debug"]["prompt"], height=100)
                st.text_area("Raw Response", result["debug"]["raw_response"], height=80)
    
    elif isinstance(result, dict) and "sentences" in result:
        # Full note result
        for i, sentence_data in enumerate(result["sentences"]):
            st.write(f"**Sentence {sentence_data['sentence_number']}:** {sentence_data['sentence']}")
            factors = sentence_data["sdoh_factors"]
            
            if factors == ["NoSDoH"]:
                st.markdown('<div class="sdoh-factor no-sdoh">No SDoH Factors</div>', unsafe_allow_html=True)
            else:
                for factor in factors:
                    st.markdown(f'<div class="sdoh-factor">{factor}</div>', unsafe_allow_html=True)
            
            if show_debug and "debug" in sentence_data:
                with st.expander(f"🔍 Debug Info - Sentence {sentence_data['sentence_number']}"):
                    st.text_area(f"Prompt {i}", sentence_data["debug"]["prompt"], height=100, key=f"prompt_{i}")
                    st.text_area(f"Raw Response {i}", sentence_data["debug"]["raw_response"], height=80, key=f"response_{i}")
            
            st.divider()

def display_dataset_stats(stats: Dict[str, Any]):
    """Display dataset statistics"""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Notes", f"{stats['total_notes']:,}")
    with col2:
        st.metric("Notes with Text", f"{stats['notes_with_text']:,}")
    with col3:
        st.metric("Avg Note Length", f"{stats['avg_length']:.0f} chars")
    with col4:
        st.metric("Median Length", f"{stats['median_length']:.0f} chars")

def display_config_summary(config: Dict[str, Any]):
    """Display configuration summary"""
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.write("**Current Settings:**")
    st.write(f"Model: `{config['selected_model'].split('/')[-1]}`")
    st.write(f"Prompt: `{config['prompt_type']}`")
    st.write(f"Level: `{config['level']}`")
    st.write(f"Debug: `{config['debug_mode']}`")
    st.markdown('</div>', unsafe_allow_html=True)

def display_note_with_analysis(idx, row, note_column, tokenizer, model, config, key_prefix=""):
    """Display a note with quick analysis option"""
    note_text = str(row[note_column])
    with st.expander(f"Note {idx} (Length: {len(note_text)} chars): {note_text[:100]}..."):
        st.write(f"**Full text:** {note_text}")
        
        if st.button(f"Quick Analyze", key=f"{key_prefix}_{idx}"):
            if tokenizer and model:
                from streamlit_app.components.model_manager import ModelManager
                extractor = ModelManager.create_extractor(tokenizer, model, config)
                if extractor:
                    with st.spinner("Analyzing..."):
                        result = extractor.extract_from_note(note_text)
                    display_extraction_result(result, False)
            else:
                st.error("Model not loaded")

def create_pagination_controls(pagination_info: Dict[str, Any]) -> int:
    """Create pagination controls and return new page number"""
    col1, col2, col3, col4 = st.columns(4)
    
    current_page = pagination_info['current_page']
    total_pages = pagination_info['total_pages']
    
    new_page = current_page
    
    with col1:
        if st.button("⏮️ First") and current_page > 1:
            new_page = 1
    with col2:
        if st.button("◀️ Previous") and current_page > 1:
            new_page = current_page - 1
    with col3:
        if st.button("▶️ Next") and current_page < total_pages:
            new_page = current_page + 1
    with col4:
        if st.button("⏭️ Last") and current_page < total_pages:
            new_page = total_pages
    
    return new_page