"""
Dataset browser component for the Streamlit app
"""

import streamlit as st
import pandas as pd
import plotly.express as px

from .display_utils import display_dataset_stats, display_note_with_analysis, create_pagination_controls
from .data_manager import DataManager

class DatasetBrowser:
    """Dataset browser component"""
    
    def __init__(self, df, note_column, tokenizer, model, config):
        self.df = df
        self.note_column = note_column
        self.tokenizer = tokenizer
        self.model = model
        self.config = config
        self.data_manager = DataManager()
    
    def render(self):
        """Render the dataset browser tab"""
        st.markdown('<h2 class="section-header">Dataset Browser</h2>', unsafe_allow_html=True)
        
        # Dataset overview
        stats = self.data_manager.get_dataset_stats(self.df, self.note_column)
        display_dataset_stats(stats)
        
        # Browse options
        st.subheader("🔍 Browse Options")
        
        browse_method = st.radio(
            "How would you like to browse the notes?",
            ["📊 Random Sample", "🔍 Search & Filter", "📖 Sequential Browse", "📈 By Length"],
            horizontal=True
        )
        
        if browse_method == "📊 Random Sample":
            self._render_random_sample()
        elif browse_method == "🔍 Search & Filter":
            self._render_search_filter()
        elif browse_method == "📖 Sequential Browse":
            self._render_sequential_browse()
        elif browse_method == "📈 By Length":
            self._render_by_length()
    
    def _render_random_sample(self):
        """Render random sample browser"""
        st.markdown("**Random Sample of Notes**")
        
        col1, col2 = st.columns(2)
        with col1:
            sample_size = st.slider("Sample size", 5, 50, 10)
        with col2:
            if st.button("🎲 Generate New Sample"):
                # Clear the cached sample when generating new one
                if 'random_sample_df' in st.session_state:
                    del st.session_state.random_sample_df
                if 'sample_size_state' in st.session_state:
                    del st.session_state.sample_size_state
                st.rerun()
        
        # Check if we need to generate a new sample
        if ('random_sample_df' not in st.session_state or 
            'sample_size_state' not in st.session_state or 
            st.session_state.sample_size_state != sample_size):
            
            # Generate new sample and store in session state
            st.session_state.random_sample_df = self.df.sample(
                n=min(sample_size, len(self.df)), 
                random_state=None
            )
            st.session_state.sample_size_state = sample_size
        
        # Use the cached sample
        sample_df = st.session_state.random_sample_df
        
        for idx, row in sample_df.iterrows():
            self._display_note_with_analysis_safe(idx, row, "random")
    
    def _render_search_filter(self):
        """Render search and filter browser"""
        st.markdown("**Search and Filter Notes**")
        
        col1, col2 = st.columns(2)
        with col1:
            search_term = st.text_input(
                "Search in note text:",
                placeholder="e.g., housing, employment, food..."
            )
        with col2:
            min_length = st.number_input("Minimum note length", min_value=0, value=0)
            max_length = st.number_input("Maximum note length", min_value=0, value=10000)
        
        # Apply filters
        filtered_df = self.data_manager.search_notes(
            self.df, self.note_column, search_term, min_length, max_length
        )
        
        st.write(f"**Found {len(filtered_df):,} matching notes**")
        
        if len(filtered_df) > 0:
            display_count = st.slider("Number to display", 1, min(50, len(filtered_df)), 10)
            display_df = filtered_df.head(display_count)
            
            for idx, row in display_df.iterrows():
                # Pass search term for highlighting
                self._display_note_with_analysis_safe(idx, row, "search", highlight_term=search_term)
        else:
            st.info("No notes match your search criteria.")
    
    def _render_sequential_browse(self):
        """Render sequential browse with pagination"""
        st.markdown("**Sequential Browse with Pagination**")
        
        # Pagination controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            page_size = st.selectbox("Notes per page", [10, 25, 50], index=1)
        
        total_pages = (len(self.df) - 1) // page_size + 1
        
        with col2:
            current_page = st.number_input(
                "Page number", 
                min_value=1, 
                max_value=total_pages, 
                value=1
            )
        
        with col3:
            st.write(f"Page {current_page} of {total_pages}")
            st.write(f"Total: {len(self.df):,} notes")
        
        # Get page data
        page_df, pagination_info = self.data_manager.paginate_dataframe(
            self.df, page_size, current_page
        )
        
        # Navigation buttons
        new_page = create_pagination_controls(pagination_info)
        if new_page != current_page:
            st.query_params.page = str(new_page)
            st.rerun()
        
        # Display notes
        for idx, row in page_df.iterrows():
            self._display_note_with_analysis_safe(idx, row, "seq")
    
    def _render_by_length(self):
        """Render browse by length categories"""
        st.markdown("**Browse Notes by Length**")
        
        # Length statistics
        col1, col2 = st.columns(2)
        with col1:
            length_category = st.selectbox(
                "Length category",
                ["Very Short (< 50 chars)", "Short (50-200 chars)", 
                 "Medium (200-500 chars)", "Long (500-1000 chars)", 
                 "Very Long (> 1000 chars)"]
            )
        
        # Filter by length
        length_df = self.data_manager.get_notes_by_length_category(
            self.df, self.note_column, length_category
        )
        
        with col2:
            st.write(f"**{len(length_df):,} notes** in this category")
            if len(length_df) > 0:
                avg_len = length_df[self.note_column].str.len().mean()
                st.write(f"Average length: {avg_len:.0f} chars")
        
        # Length distribution chart
        if len(length_df) > 0:
            fig = px.histogram(
                x=length_df[self.note_column].str.len(),
                nbins=20,
                title=f"Length Distribution for {length_category}"
            )
            fig.update_xaxes(title="Note Length (characters)")
            fig.update_yaxes(title="Number of Notes")
            st.plotly_chart(fig, use_container_width=True)
            
            # Sample from this category
            sample_size = st.slider("Sample size", 1, min(20, len(length_df)), 5)
            sample_df = length_df.sample(n=sample_size)
            
    def _display_note_with_analysis_safe(self, idx, row, key_prefix, highlight_term=None):
        """Display a note with analysis, using session state to preserve results"""
        note_text = str(row[self.note_column])
        
        # Create preview text with highlighting if needed
        preview_text = note_text[:200]
        if highlight_term:
            preview_text = preview_text.replace(highlight_term, f"**{highlight_term}**")
        else:
            preview_text = note_text[:100]
        
        with st.expander(f"Note {idx} (Length: {len(note_text)} chars): {preview_text}..."):
            st.write(f"**Full text:** {note_text}")
            
            # Create unique key for this note's analysis
            analysis_key = f"analysis_{key_prefix}_{idx}"
            
            col1, col2 = st.columns([1, 4])
            
            with col1:
                if st.button(f"Quick Analyze", key=f"{key_prefix}_{idx}"):
                    if self.tokenizer and self.model:
                        from .model_manager import ModelManager
                        extractor = ModelManager.create_extractor(self.tokenizer, self.model, self.config)
                        if extractor:
                            with st.spinner("Analyzing..."):
                                result = extractor.extract_from_note(note_text)
                            # Store result in session state
                            st.session_state[analysis_key] = result
                    else:
                        st.error("Model not loaded")
            
            with col2:
                # Display analysis result if it exists in session state
                if analysis_key in st.session_state:
                    st.write("**Analysis Result:**")
                    from .display_utils import display_extraction_result
                    display_extraction_result(st.session_state[analysis_key], False)