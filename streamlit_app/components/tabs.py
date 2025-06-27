"""
Tab implementations for the Streamlit app
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import json
from datetime import datetime
from pathlib import Path

from .display_utils import (
    display_extraction_result, 
    display_dataset_stats, 
    display_config_summary,
    display_note_with_analysis,
    create_pagination_controls
)
from .data_manager import DataManager
from .model_manager import ModelManager

def single_analysis_tab(df, note_column, tokenizer, model, config):
    """Single text analysis tab"""
    st.markdown('<h2 class="section-header">Single Text Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Text input options
        input_type = st.radio("Input Type", ["Custom Text", "Select from Dataset"], horizontal=True)
        
        if input_type == "Custom Text":
            user_text = st.text_area(
                "Enter text to analyze:",
                placeholder="e.g., Patient reports living in temporary housing and struggling to afford medications...",
                height=150
            )
        else:
            # Select from dataset with better handling for large datasets
            st.write("**Select from dataset:**")
            
            # Option to select by index or search
            selection_method = st.radio(
                "Selection method:", 
                ["By Index", "Search"], 
                horizontal=True
            )
            
            if selection_method == "By Index":
                selected_idx = st.number_input(
                    "Enter note index:",
                    min_value=0,
                    max_value=len(df)-1,
                    value=0,
                    help=f"Choose any index from 0 to {len(df)-1}"
                )
                user_text = df.iloc[selected_idx][note_column]
                st.write(f"**Selected Note {selected_idx}:**")
                st.text_area("Selected text:", user_text, height=150, disabled=True)
            
            else:  # Search method
                search_term = st.text_input(
                    "Search for notes containing:",
                    placeholder="e.g., housing, employment, food"
                )
                
                if search_term:
                    matching_notes = DataManager.search_notes(df, note_column, search_term)
                    
                    if len(matching_notes) > 0:
                        st.write(f"Found {len(matching_notes)} matching notes")
                        
                        # Select from matching notes
                        match_idx = st.selectbox(
                            "Select from matching notes:",
                            range(min(20, len(matching_notes))),
                            format_func=lambda x: f"Match {x}: {str(matching_notes.iloc[x][note_column])[:100]}..."
                        )
                        user_text = matching_notes.iloc[match_idx][note_column]
                        st.text_area("Selected text:", user_text, height=150, disabled=True)
                    else:
                        st.info("No notes found matching your search term")
                        user_text = ""
                else:
                    user_text = ""
        
        analyze_sentence = st.checkbox("Analyze as single sentence (vs. full note)")
    
    with col2:
        display_config_summary(config)
    
    if st.button("🔍 Analyze Text", type="primary"):
        if user_text and user_text.strip():
            if tokenizer and model:
                # Create extractor
                extractor = ModelManager.create_extractor(tokenizer, model, config)
                
                if extractor:
                    # Analyze
                    with st.spinner("Analyzing..."):
                        if analyze_sentence:
                            result = extractor.extract_from_sentence(user_text.strip())
                        else:
                            result = extractor.extract_from_note(user_text.strip())
                    
                    st.markdown('<h3 class="section-header">Results</h3>', unsafe_allow_html=True)
                    display_extraction_result(result, config['debug_mode'])
            else:
                st.error("Model not loaded")
        else:
            st.warning("Please enter some text to analyze")

def dataset_browser_tab(df, note_column, tokenizer, model, config):
    """Dataset browsing tab"""
    from .dataset_browser import DatasetBrowser
    browser = DatasetBrowser(df, note_column, tokenizer, model, config)
    browser.render()

def batch_processing_tab(df, note_column, tokenizer, model, config):
    """Batch processing tab"""
    st.markdown('<h2 class="section-header">Batch Processing</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        batch_size = st.number_input("Batch Size", min_value=1, max_value=100, value=10)
        start_index = st.number_input("Start Index", min_value=0, max_value=len(df)-1, value=0)
    
    with col2:
        st.info(f"Will process notes {start_index} to {min(start_index + batch_size - 1, len(df) - 1)}")
        st.write(f"Dataset size: {len(df)} notes")
    
    if st.button("🚀 Start Batch Processing", type="primary"):
        if tokenizer and model:
            # Create extractor
            extractor = ModelManager.create_extractor(tokenizer, model, config)
            
            if extractor:
                # Run batch processing
                from src.classification.batch_processing_helpers import BatchProcessor
                processor = BatchProcessor(output_dir="results/batch_results")
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Process batch
                    batch_results = processor.process_batch(
                        df=df,
                        extractor=extractor,
                        note_column=note_column,
                        batch_size=batch_size,
                        start_index=start_index
                    )
                    
                    progress_bar.progress(1.0)
                    status_text.success("Batch processing completed!")
                    
                    # Save results
                    results_file = processor.save_results(batch_results)
                    
                    st.success(f"✅ Batch processing completed! Results saved to: `{results_file}`")
                    
                    # Show summary
                    stats = batch_results["summary_stats"]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Notes Processed", stats["total_notes"])
                    with col2:
                        st.metric("Total Sentences", stats["total_sentences"])
                    with col3:
                        st.metric("Sentences with SDoH", stats["sentences_with_sdoh"])
                    with col4:
                        detection_rate = stats["sentences_with_sdoh"] / stats["total_sentences"] if stats["total_sentences"] > 0 else 0
                        st.metric("Detection Rate", f"{detection_rate:.1%}")
                    
                    # Factor distribution
                    if stats["factor_frequencies"]:
                        st.subheader("📊 Factor Distribution")
                        factor_df = pd.DataFrame(
                            list(stats["factor_frequencies"].items()),
                            columns=["Factor", "Count"]
                        ).sort_values("Count", ascending=False)
                        
                        fig = px.bar(factor_df, x="Factor", y="Count", title="SDoH Factors Found")
                        fig.update_xaxes(tickangle=45)
                        st.plotly_chart(fig, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Error during batch processing: {e}")
        else:
            st.error("Model not loaded")

def results_analysis_tab():
    """Results analysis tab"""
    st.markdown('<h2 class="section-header">Results Analysis</h2>', unsafe_allow_html=True)
    
    # Load existing results
    results_dir = Path("results/batch_results")
    if results_dir.exists():
        result_files = list(results_dir.glob("*.json"))
        
        if result_files:
            st.subheader("📁 Available Result Files")
            
            selected_files = st.multiselect(
                "Select result files to compare:",
                [f.name for f in result_files],
                default=[f.name for f in result_files[:3]]  # Select first 3 by default
            )
            
            if selected_files and st.button("📊 Generate Comparison Analysis"):
                try:
                    # Create analyzer
                    from src.classification.batch_processing_helpers import ResultsAnalyzer, BatchProcessor
                    analyzer = ResultsAnalyzer()
                    processor = BatchProcessor()
                    
                    # Load selected files
                    selected_paths = [str(results_dir / f) for f in selected_files]
                    comparison_df = processor.create_comparison_dataset(selected_paths)
                    
                    # Generate analysis
                    analysis = analyzer.compare_models(comparison_df)
                    
                    st.subheader("🔍 Model Comparison")
                    
                    # Create comparison table
                    comparison_data = []
                    for config, stats in analysis.items():
                        comparison_data.append({
                            "Configuration": config,
                            "Total Sentences": stats["total_sentences"],
                            "SDoH Detection Rate": f"{stats['sdoh_detection_rate']:.1%}",
                            "Avg Factors/Sentence": f"{stats['avg_factors_per_sentence']:.2f}",
                            "Unique Factors": stats["unique_factors"]
                        })
                    
                    comparison_table_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_table_df, use_container_width=True)
                    
                    # Detection rate chart
                    fig = px.bar(
                        comparison_table_df,
                        x="Configuration",
                        y="SDoH Detection Rate",
                        title="SDoH Detection Rate by Configuration"
                    )
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Save analysis
                    analysis_dir = Path("results/comparison_analyses")
                    analysis_dir.mkdir(parents=True, exist_ok=True)
                    
                    analysis_file = analysis_dir / f"comparison_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    
                    with open(analysis_file, 'w') as f:
                        json.dump({
                            "analysis": analysis,
                            "summary": comparison_data,
                            "timestamp": datetime.now().isoformat()
                        }, f, indent=2)
                    
                    st.success(f"Analysis saved to: `{analysis_file}`")
                    
                except Exception as e:
                    st.error(f"Error during analysis: {e}")
        else:
            st.info("No result files found. Run batch processing first.")
    else:
        st.info("Results directory not found. Run batch processing first.")