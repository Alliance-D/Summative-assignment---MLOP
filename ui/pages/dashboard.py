"""
Dashboard Page
Shows system status, metrics, and model information
"""

import streamlit as st
import requests
from datetime import datetime
import time
from pathlib import Path
import sys
import os

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

API_URL = os.getenv("API_URL", "http://localhost:8000")


def show():
    """Display dashboard page"""
    st.title(" System Dashboard")
    st.write("Monitor system health, model status, and performance metrics")
    
    st.divider()
    
    # Auto-refresh toggle
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader(" Live Status")
    with col2:
        auto_refresh = st.checkbox("Auto-refresh (5s)", value=False)
    
    if auto_refresh:
        time.sleep(5)
        st.rerun()
    
    # Fetch data from API
    try:
        # Health check
        health_response = requests.get(f"{API_URL}/health", timeout=5)
        health_data = health_response.json() if health_response.status_code == 200 else None
        
        # Status
        status_response = requests.get(f"{API_URL}/status", timeout=5)
        status_data = status_response.json() if status_response.status_code == 200 else None
        
        # Model info
        model_response = requests.get(f"{API_URL}/model/info", timeout=5)
        model_data = model_response.json() if model_response.status_code == 200 else None
        
        # Metrics
        metrics_response = requests.get(f"{API_URL}/metrics", timeout=5)
        metrics_data = metrics_response.json() if metrics_response.status_code == 200 else None
        
        # Display status indicators
        st.divider()
        
        # Status cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if health_data and health_data.get("status") == "healthy":
                st.success(" **API Status**\n\nHealthy")
            else:
                st.error(" **API Status**\n\nUnhealthy")
        
        with col2:
            if status_data and status_data.get("model_loaded"):
                st.success(" **Model Status**\n\nLoaded")
            else:
                st.error(" **Model Status**\n\nNot Loaded")
        
        with col3:
            if status_data and status_data.get("retraining_in_progress"):
                st.warning(" **Retraining**\n\nIn Progress")
            else:
                st.info(" **Retraining**\n\nIdle")
        
        with col4:
            uptime_hours = health_data.get("uptime_seconds", 0) / 3600 if health_data else 0
            st.info(f" **Uptime**\n\n{uptime_hours:.1f} hours")
        
        st.divider()
        
        # Model Information
        st.subheader(" Model Information")
        
        if model_data:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("🌱 Plant Classes", model_data.get("num_plant_classes", "N/A"))
                st.metric("🦠 Disease Classes", model_data.get("num_disease_classes", "N/A"))
                
                if "plant_classes" in model_data:
                    with st.expander("📋 Plant Types"):
                        for plant in model_data["plant_classes"]:
                            st.write(f"- {plant}")
            
            with col2:
                st.metric("📦 Model Version", model_data.get("version", "v1.0.0"))
                st.metric("🏗️ Architecture", "MobileNetV2")
                
                if "disease_classes" in model_data:
                    with st.expander("📋 Disease Types"):
                        for disease in model_data["disease_classes"][:10]:
                            st.write(f"- {disease}")
                        if len(model_data["disease_classes"]) > 10:
                            st.write(f"... and {len(model_data['disease_classes']) - 10} more")
        else:
            st.warning(" Could not fetch model information")
        
        st.divider()
        
        # Performance Metrics
        st.subheader(" Performance Metrics")
        
        if metrics_data and "predictions" in metrics_data:
            pred_metrics = metrics_data["predictions"]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Predictions",
                    pred_metrics.get("total", 0)
                )
            
            with col2:
                st.metric(
                    "Successful",
                    pred_metrics.get("successful", 0),
                    delta=None
                )
            
            with col3:
                st.metric(
                    "Failed",
                    pred_metrics.get("failed", 0),
                    delta=None
                )
            
            with col4:
                success_rate = pred_metrics.get("success_rate", 0)
                st.metric(
                    "Success Rate",
                    f"{success_rate:.1f}%",
                    delta=None
                )
            
            # Progress bar for success rate
            st.progress(success_rate / 100, text=f"Success Rate: {success_rate:.1f}%")
        else:
            st.info(" No predictions made yet")
        
        st.divider()
        
        # System Information
        st.subheader(" System Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**API Configuration**")
            st.write(f"- Host: `localhost:8000`")
            st.write(f"- Version: `2.0.0`")
            st.write(f"- Docs: [Swagger UI](http://localhost:8000/docs)")
        
        with col2:
            st.write("**Model Details**")
            if model_data:
                st.write(f"- Created: {model_data.get('created_at', 'N/A')[:10]}")
                st.write(f"- Input Shape: `224x224x3`")
                st.write(f"- Parameters: `{model_data.get('total_parameters', 'N/A'):,}`")
        
        st.divider()
        
        # Refresh button
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button(" Refresh Data", use_container_width=True):
                st.rerun()
        with col2:
            if st.button(" View Metrics", use_container_width=True):
                st.switch_page("ui/pages/metrics.py")
    
    except requests.exceptions.ConnectionError:
        st.error(" **Cannot connect to API**")
        st.info("Please make sure the API is running at `http://localhost:8000`")
        st.code("python api/app.py", language="bash")
    
    except Exception as e:
        st.error(f" Error: {str(e)}")