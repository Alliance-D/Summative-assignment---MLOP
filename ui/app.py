"""
Streamlit UI Configuration
Main entry point for the Streamlit application
"""

import streamlit as st
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Page config
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Plant Disease Detection and Classification System using Deep Learning"
    }
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictions_count' not in st.session_state:
    st.session_state.predictions_count = 0
if 'uptime_start' not in st.session_state:
    from datetime import datetime
    st.session_state.uptime_start = datetime.now()

# Sidebar navigation
st.sidebar.title("🌱 Plant Disease Detector")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate to:",
    ["Dashboard", "Predict", "Batch Process", "Visualizations & Analytics", "Retraining", "Metrics"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "📱 **Plant Disease Detection System**\n\n"
    "Two-stage ML Pipeline for accurate plant disease diagnosis"
)

# Load pages
if page == "Dashboard":
    from ui.pages import dashboard
    dashboard.show()
elif page == "Predict":
    from ui.pages import predict
    predict.show()
elif page == "Batch Process":
    from ui.pages import batch_process
    batch_process.show()
elif page == "Visualizations & Analytics":
    from ui.pages import analytics
    analytics.show()
elif page == "Retraining":
    from ui.pages import retraining
    retraining.show()
elif page == "Metrics":
    from ui.pages import metrics
    metrics.show()
