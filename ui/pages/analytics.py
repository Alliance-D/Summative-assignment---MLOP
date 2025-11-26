"""
Analytics & Visualizations Page
Real-time analytics from database
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
    """Display analytics page with real data"""
    st.title("📊 Data Analytics & Visualizations")
    
    st.write("Real-time analytics and insights from prediction history")
    
    st.divider()
    
    # Refresh button
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("📈 Live Analytics")
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    
    try:
        # Fetch analytics data
        summary_response = requests.get(f"{API_URL}/analytics/summary", timeout=10)
        
        if summary_response.status_code == 200:
            analytics = summary_response.json()
            
            pred_stats = analytics.get("prediction_stats", {})
            plant_dist = analytics.get("plant_distribution", {})
            disease_dist = analytics.get("disease_distribution", {})
            confidence_stats = analytics.get("confidence_stats", {})
            
            # Check if we have data
            if pred_stats.get("total", 0) == 0:
                st.info("📊 No predictions recorded yet. Make some predictions to see analytics!")
                st.divider()
                st.subheader("ℹ️ What You'll See Here")
                st.write("""
                Once you start making predictions, this page will show:
                
                **Feature Interpretation #1: Plant Distribution**
                - Which plant types are most commonly identified
                - Distribution across different species
                
                **Feature Interpretation #2: Disease Patterns**
                - Most common diseases detected
                - Healthy vs diseased plant ratios
                
                **Feature Interpretation #3: Model Confidence**
                - Confidence score distributions
                - Model performance insights
                """)
                return
            
            # Show overall stats
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Predictions", pred_stats.get("total", 0))
            with col2:
                st.metric("Success Rate", f"{pred_stats.get('success_rate', 0):.1f}%")
            with col3:
                st.metric("Plant Types", len(plant_dist))
            with col4:
                st.metric("Disease Types", len(disease_dist))
            
            st.divider()
            
            # Tabs
            tab1, tab2, tab3 = st.tabs([
                "🌱 Plant Distribution",
                "🦠 Disease Patterns",
                "💡 Confidence Analysis"
            ])
            
            # ============================================================
            # TAB 1: PLANT DISTRIBUTION
            # ============================================================
            with tab1:
                st.subheader("🌱 Plant Type Distribution")
                
                st.write("""
                **Feature Interpretation #1: Plant Species in Predictions**
                
                This shows which plant types users are most frequently analyzing.
                Different plants have distinct leaf characteristics that enable accurate classification.
                """)
                
                if plant_dist:
                    # Bar chart
                    fig = go.Figure(data=[
                        go.Bar(
                            x=list(plant_dist.keys()),
                            y=list(plant_dist.values()),
                            marker=dict(
                                color=list(plant_dist.values()),
                                colorscale='Viridis',
                                showscale=True,
                                colorbar=dict(title="Count")
                            ),
                            text=list(plant_dist.values()),
                            textposition='auto'
                        )
                    ])
                    fig.update_layout(
                        title="Plant Type Prediction Distribution",
                        xaxis_title="Plant Type",
                        yaxis_title="Number of Predictions",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Pie chart
                    fig2 = go.Figure(data=[go.Pie(
                        labels=list(plant_dist.keys()),
                        values=list(plant_dist.values()),
                        hole=0.3
                    )])
                    fig2.update_layout(
                        title="Plant Type Distribution (%)",
                        height=400
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    # Table
                    st.write("**Detailed Breakdown:**")
                    plant_df = pd.DataFrame({
                        "Plant Type": list(plant_dist.keys()),
                        "Predictions": list(plant_dist.values()),
                        "Percentage": [f"{(v/pred_stats['total'])*100:.1f}%" for v in plant_dist.values()]
                    })
                    plant_df = plant_df.sort_values("Predictions", ascending=False)
                    st.dataframe(plant_df, use_container_width=True, hide_index=True)
                    
                    st.info(f"""
                    💡 **Insight**: 
                    - Most common plant: **{max(plant_dist, key=plant_dist.get)}** ({max(plant_dist.values())} predictions)
                    - This indicates {max(plant_dist, key=plant_dist.get)} is the most frequently analyzed crop
                    """)
                else:
                    st.warning("No plant distribution data available yet")
            
            # ============================================================
            # TAB 2: DISEASE PATTERNS
            # ============================================================
            with tab2:
                st.subheader("🦠 Disease Detection Patterns")
                
                st.write("""
                **Feature Interpretation #2: Disease Prevalence**
                
                Understanding which diseases are most common helps in prevention strategies.
                The model identifies diseases based on visual symptoms like discoloration, spots, and lesions.
                """)
                
                if disease_dist:
                    # Horizontal bar chart
                    diseases = list(disease_dist.keys())
                    counts = list(disease_dist.values())
                    
                    # Color code healthy vs diseased
                    colors = ['#2ecc71' if 'healthy' in d.lower() else '#e74c3c' for d in diseases]
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            y=diseases,
                            x=counts,
                            orientation='h',
                            marker=dict(color=colors),
                            text=counts,
                            textposition='auto'
                        )
                    ])
                    fig.update_layout(
                        title="Disease Detection Distribution",
                        xaxis_title="Number of Predictions",
                        yaxis_title="Disease Type",
                        height=max(400, len(diseases) * 30)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Healthy vs Diseased ratio
                    healthy_count = sum(v for k, v in disease_dist.items() if 'healthy' in k.lower())
                    diseased_count = sum(v for k, v in disease_dist.items() if 'healthy' not in k.lower())
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Healthy Plants", healthy_count)
                    with col2:
                        st.metric("Diseased Plants", diseased_count)
                    
                    # Health ratio pie
                    fig2 = go.Figure(data=[go.Pie(
                        labels=['Healthy', 'Diseased'],
                        values=[healthy_count, diseased_count],
                        marker=dict(colors=['#2ecc71', '#e74c3c']),
                        hole=0.4
                    )])
                    fig2.update_layout(
                        title="Health Status Distribution",
                        height=400
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    # Table
                    st.write("**Disease Breakdown:**")
                    disease_df = pd.DataFrame({
                        "Disease": diseases,
                        "Cases": counts,
                        "Percentage": [f"{(c/pred_stats['total'])*100:.1f}%" for c in counts],
                        "Status": ["✅ Healthy" if 'healthy' in d.lower() else "⚠️ Diseased" for d in diseases]
                    })
                    disease_df = disease_df.sort_values("Cases", ascending=False)
                    st.dataframe(disease_df, use_container_width=True, hide_index=True)
                    
                    if diseased_count > 0:
                        disease_rate = (diseased_count / pred_stats['total']) * 100
                        st.warning(f"""
                        ⚠️ **Alert**: 
                        - Disease rate: **{disease_rate:.1f}%** of analyzed plants show disease symptoms
                        - Most common disease: **{max((k for k in disease_dist if 'healthy' not in k.lower()), key=disease_dist.get, default='None')}**
                        """)
                else:
                    st.warning("No disease distribution data available yet")
            
            # ============================================================
            # TAB 3: CONFIDENCE ANALYSIS
            # ============================================================
            with tab3:
                st.subheader(" Model Confidence Analysis")
                
                st.write("""
                **Feature Interpretation #3: Prediction Confidence**
                
                Confidence scores indicate how certain the model is about its predictions.
                Higher confidence means stronger pattern recognition.
                """)
                
                if confidence_stats:
                    # Confidence metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        avg_plant = confidence_stats.get("avg_plant_confidence", 0) * 100
                        st.metric("Avg Plant Confidence", f"{avg_plant:.1f}%")
                    
                    with col2:
                        avg_disease = confidence_stats.get("avg_disease_confidence", 0) * 100
                        st.metric("Avg Disease Confidence", f"{avg_disease:.1f}%")
                    
                    with col3:
                        avg_overall = confidence_stats.get("avg_overall_confidence", 0) * 100
                        st.metric("Avg Overall Confidence", f"{avg_overall:.1f}%")
                    
                    # Confidence comparison
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        name='Plant Classification',
                        x=['Average', 'Minimum', 'Maximum'],
                        y=[
                            confidence_stats.get("avg_plant_confidence", 0) * 100,
                            confidence_stats.get("min_plant_confidence", 0) * 100,
                            confidence_stats.get("max_plant_confidence", 0) * 100
                        ],
                        marker_color='#2ecc71'
                    ))
                    
                    fig.add_trace(go.Bar(
                        name='Disease Classification',
                        x=['Average', 'Minimum', 'Maximum'],
                        y=[
                            confidence_stats.get("avg_disease_confidence", 0) * 100,
                            confidence_stats.get("min_disease_confidence", 0) * 100,
                            confidence_stats.get("max_disease_confidence", 0) * 100
                        ],
                        marker_color='#e74c3c'
                    ))
                    
                    fig.update_layout(
                        title="Confidence Score Comparison",
                        yaxis_title="Confidence (%)",
                        barmode='group',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Interpretation
                    if avg_plant > avg_disease:
                        st.info("""
                        💡 **Insight**: 
                        - Plant classification has higher confidence than disease classification
                        - This is expected as plants have more distinct visual differences
                        - Disease symptoms can be more subtle and require careful analysis
                        """)
                    else:
                        st.success("""
                        ✅ **Excellent**: 
                        - Both plant and disease classification show high confidence
                        - Model is performing well across both tasks
                        """)
                else:
                    st.warning("No confidence statistics available yet")
        
        else:
            st.error(f" Could not fetch analytics: Status {summary_response.status_code}")
    
    except requests.exceptions.ConnectionError:
        st.error(" Cannot connect to API")
        st.info("Make sure the API is running:")
        st.code("python api/app.py", language="bash")
    
    except Exception as e:
        st.error(f" Error: {str(e)}")


if __name__ == "__main__":
    show()