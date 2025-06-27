"""
Configuration and styling for the Streamlit app
"""

import streamlit as st

def configure_page():
    """Configure page settings and styling"""
    
    st.set_page_config(
        page_title="SDoH Extraction Tool",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2e7d32;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .sdoh-factor {
        background-color: #e3f2fd;
        padding: 0.3rem 0.8rem;
        margin: 0.2rem;
        border-radius: 1rem;
        display: inline-block;
        font-size: 0.9rem;
    }
    .no-sdoh {
        background-color: #f5f5f5;
        color: #666;
    }
    .stAlert > div {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown('<h1 class="main-header">🏥 Social Determinants of Health Extraction Tool</h1>', unsafe_allow_html=True)

def load_sidebar_config(note_column):
    """Load sidebar configuration and return config dictionary"""
    
    st.sidebar.header("⚙️ Configuration")
    
    # Model selection
    available_models = [
        "Qwen/Qwen2.5-7B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "microsoft/Phi-4-mini-instruct",
        "mistralai/Mistral-7B-Instruct-v0.3"
    ]
    
    selected_model = st.sidebar.selectbox(
        "Select Model",
        available_models,
        index=0
    )
    
    # Prompt configuration
    prompt_type = st.sidebar.selectbox(
        "Prompt Type",
        ["zero_shot_detailed", "zero_shot_basic", "five_shot_detailed", "five_shot_basic"],
        index=0
    )
    
    level = st.sidebar.selectbox(
        "Classification Level",
        [1, 2],
        index=0
    )
    
    debug_mode = st.sidebar.checkbox("Enable Debug Mode", value=False)
    
    # Display current configuration
    st.sidebar.success(f"Using column: {note_column}")
    
    # Configuration summary
    with st.sidebar.expander("📋 Current Settings"):
        st.write(f"**Model:** `{selected_model.split('/')[-1]}`")
        st.write(f"**Prompt:** `{prompt_type}`")
        st.write(f"**Level:** `{level}`")
        st.write(f"**Debug:** `{debug_mode}`")
    
    return {
        'selected_model': selected_model,
        'prompt_type': prompt_type,
        'level': level,
        'debug_mode': debug_mode
    }