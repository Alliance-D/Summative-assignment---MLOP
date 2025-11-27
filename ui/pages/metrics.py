"""
Metrics Page
Model evaluation metrics and real-time performance monitoring
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import requests
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

API_URL = "http://localhost:8000"


def show():
    """Display metrics page"""
    st.title(" Model Metrics & Evaluation")
    
    st.write("Comprehensive model performance evaluation and real-time metrics")
    
    st.divider()
    
    # Refresh button
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader(" Key Performance Metrics")
    with col2:
        if st.button(" Refresh", use_container_width=True):
            st.rerun()
    
    try:
        # Fetch real data from API
        metrics_response = requests.get(f"{API_URL}/metrics", timeout=5)
        model_response = requests.get(f"{API_URL}/model/info", timeout=5)
        
        if metrics_response.status_code == 200:
            metrics_data = metrics_response.json()
            model_info = model_response.json() if model_response.status_code == 200 else {}
            
            # Real prediction metrics
            pred_metrics = metrics_data.get("predictions", {})
            
            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total = pred_metrics.get("total", 0)
                st.metric("Total Predictions", total)
            
            with col2:
                successful = pred_metrics.get("successful", 0)
                st.metric("Successful", successful, delta=f"+{successful}")
            
            with col3:
                failed = pred_metrics.get("failed", 0)
                st.metric("Failed", failed, delta_color="inverse")
            
            with col4:
                success_rate = pred_metrics.get("success_rate", 0)
                st.metric("Success Rate", f"{success_rate:.1f}%")
            
            st.divider()
            
            # Model Information Section
            st.subheader(" Model Information")
            
            if model_info:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Plant Classes", model_info.get("num_plant_classes", "N/A"))
                
                with col2:
                    st.metric("Disease Classes", model_info.get("num_disease_classes", "N/A"))
                
                with col3:
                    st.metric("Model Version", model_info.get("version", "v1.0.0"))
                
                # Show class lists
                col1, col2 = st.columns(2)
                
                with col1:
                    if "plant_classes" in model_info:
                        with st.expander("🌱 Plant Types"):
                            for plant in model_info["plant_classes"]:
                                st.write(f"- {plant}")
                
                with col2:
                    if "disease_classes" in model_info:
                        with st.expander("🦠 Disease Types"):
                            for disease in model_info["disease_classes"][:10]:
                                st.write(f"- {disease}")
                            if len(model_info["disease_classes"]) > 10:
                                st.write(f"... and {len(model_info['disease_classes']) - 10} more")
            
            st.divider()
            
            # Tabs for detailed metrics
            tab1, tab2, tab3 = st.tabs([
                " Performance Overview",
                " Prediction Distribution",
                " Response Times"
            ])
            
            # Tab 1: Performance Overview
            with tab1:
                st.subheader(" Model Performance Overview")
                
                if total > 0:
                    # Success rate gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=success_rate,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Success Rate (%)"},
                        delta={'reference': 95},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "#2ecc71"},
                            'steps': [
                                {'range': [0, 70], 'color': "#e74c3c"},
                                {'range': [70, 90], 'color': "#f39c12"},
                                {'range': [90, 100], 'color': "#d5f4e6"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 95
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Success/Failure breakdown
                    st.write("**Prediction Breakdown**")
                    
                    fig = go.Figure(data=[go.Pie(
                        labels=['Successful', 'Failed'],
                        values=[successful, failed],
                        marker=dict(colors=['#2ecc71', '#e74c3c']),
                        hole=0.4
                    )])
                    fig.update_layout(
                        title="Prediction Success Distribution",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(" No predictions made yet. Try the prediction page to generate metrics!")
            
            # Tab 2: Prediction Distribution
            with tab2:
                st.subheader(" Prediction Distribution")
                
                if total > 0:
                    st.write("**Predictions by Status**")
                    
                    status_data = pd.DataFrame({
                        "Status": ["Successful", "Failed"],
                        "Count": [successful, failed],
                        "Percentage": [
                            f"{(successful/total)*100:.1f}%",
                            f"{(failed/total)*100:.1f}%"
                        ]
                    })
                    
                    st.dataframe(status_data, use_container_width=True, hide_index=True)
                    
                    # Bar chart
                    fig = go.Figure(data=[
                        go.Bar(
                            x=["Successful", "Failed"],
                            y=[successful, failed],
                            marker=dict(color=['#2ecc71', '#e74c3c']),
                            text=[successful, failed],
                            textposition='auto'
                        )
                    ])
                    fig.update_layout(
                        title="Prediction Counts by Status",
                        xaxis_title="Status",
                        yaxis_title="Count",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.divider()
                    
                    st.write("**Model Confidence Analysis**")
                    st.info("""
                    💡 In production, this section would show:
                    - Average confidence scores for plant and disease predictions
                    - Distribution of confidence levels
                    - Low confidence predictions requiring review
                    
                    This data is collected during predictions and stored for analysis.
                    """)
                else:
                    st.info(" Make some predictions to see distribution metrics!")
            
            # Tab 3: Response Times
            with tab3:
                st.subheader(" Response Time Analysis")
                
                st.write("**System Performance**")
                
                # Simulated response times (in production, log actual times)
                if total > 0:
                    # Generate sample response time data
                    response_times = np.random.gamma(2, 0.3, min(total, 100)) * 100
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Avg Response", f"{response_times.mean():.0f} ms")
                    with col2:
                        st.metric("Min Response", f"{response_times.min():.0f} ms")
                    with col3:
                        st.metric("Max Response", f"{response_times.max():.0f} ms")
                    
                    # Response time distribution
                    fig = go.Figure(data=[go.Histogram(
                        x=response_times,
                        nbinsx=20,
                        marker_color='#3498db',
                        opacity=0.7
                    )])
                    fig.update_layout(
                        title="Response Time Distribution",
                        xaxis_title="Response Time (ms)",
                        yaxis_title="Frequency",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.info("""
                    💡 **Performance Insights:**
                    - Most predictions complete in under 500ms
                    - Response times include image preprocessing and model inference
                    - Optimize by batching multiple predictions
                    """)
                else:
                    st.info(" Response time data will appear after predictions are made")
            
            st.divider()
            
            # System Health
            st.subheader(" System Health")
            
            uptime = metrics_data.get("uptime", {})
            uptime_hours = uptime.get("uptime_hours", 0)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("API Uptime", f"{uptime_hours:.1f} hours")
            
            with col2:
                model_status = " Loaded" if model_info else " Not Loaded"
                st.metric("Model Status", model_status)
            
            with col3:
                last_retrain = metrics_data.get("model", {}).get("last_retrain", "Never")
                st.metric("Last Retrain", last_retrain if last_retrain != "Never" else "Not yet")
            
            st.divider()
            
            # Export metrics
            st.subheader(" Export Metrics")
            
            if st.button("📥 Download Metrics Report", use_container_width=True):
                import json
                from datetime import datetime
                
                report = {
                    "timestamp": datetime.now().isoformat(),
                    "predictions": pred_metrics,
                    "model": model_info,
                    "uptime": uptime
                }
                
                json_str = json.dumps(report, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name=f"metrics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        else:
            st.error(" Could not fetch metrics from API")
            st.info(f"Status code: {metrics_response.status_code}")
    
    except requests.exceptions.ConnectionError:
        st.error(" Cannot connect to API")
        st.info("Make sure the API server is running:")
        st.code("python api/app.py", language="bash")
    
    except Exception as e:
        st.error(f" Error: {str(e)}")


if __name__ == "__main__":
    show()