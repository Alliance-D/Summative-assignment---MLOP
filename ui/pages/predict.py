"""
Prediction Page
Single image prediction interface
"""

import streamlit as st
import requests
from PIL import Image
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

API_URL = "http://localhost:8000"


def show():
    """Display prediction page"""
    st.title("🔮 Single Plant Prediction")
    
    st.write("Upload a leaf image to get disease prediction and recommendations.")
    
    st.divider()
    
    # Upload section
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📸 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a leaf image",
            type=["jpg", "jpeg", "png", "bmp"],
            help="Upload a clear image of a plant leaf"
        )
        
        if uploaded_file:
            st.info(f"📄 File: `{uploaded_file.name}`")
            st.info(f"📊 Size: `{uploaded_file.size / 1024:.1f} KB`")
    
    with col2:
        st.subheader("📝 Image Preview")
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
    
    st.divider()
    
    # Prediction button
    if uploaded_file is not None:
        if st.button("🔍 PREDICT", use_container_width=True, type="primary"):
            with st.spinner("🔄 Analyzing leaf... Please wait..."):
                try:
                    # Reset file pointer
                    uploaded_file.seek(0)
                    
                    # Send to API
                    files = {"file": uploaded_file}
                    response = requests.post(f"{API_URL}/predict", files=files, timeout=30)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Display results
                        st.success("✅ Prediction Complete!")
                        st.divider()
                        
                        # Main prediction results
                        res_col1, res_col2 = st.columns(2)
                        
                        with res_col1:
                            st.metric(
                                "🌱 Plant Type",
                                result.get("plant_type", "Unknown"),
                                f"{result.get('plant_confidence', 0)*100:.1f}% confidence"
                            )
                        
                        with res_col2:
                            disease = result.get("disease", "Unknown")
                            confidence = result.get('disease_confidence', 0)
                            
                            # Color code based on disease
                            if "healthy" in disease.lower():
                                st.success(f"**🦠 Disease Status**\n\n{disease}")
                                st.caption(f"Confidence: {confidence*100:.1f}%")
                            else:
                                st.warning(f"**🦠 Disease Status**\n\n{disease}")
                                st.caption(f"Confidence: {confidence*100:.1f}%")
                        
                        st.divider()
                        
                        # Overall confidence
                        overall_conf = result.get("overall_confidence", 0)
                        st.progress(overall_conf, text=f"Overall Confidence: {overall_conf*100:.1f}%")
                        
                        st.divider()
                        
                        # Top predictions
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("🏆 Top Plant Predictions")
                            for i, pred in enumerate(result.get("top_3_plants", []), 1):
                                st.write(f"{i}. **{pred['label']}** - {pred['confidence']*100:.1f}%")
                        
                        with col2:
                            st.subheader("🏆 Top Disease Predictions")
                            for i, pred in enumerate(result.get("top_3_diseases", []), 1):
                                st.write(f"{i}. **{pred['label']}** - {pred['confidence']*100:.1f}%")
                        
                        st.divider()
                        
                        # Recommendations
                        st.subheader("💡 Treatment Recommendations")
                        recommendation = result.get("recommendation", "No recommendation available")
                        
                        if "healthy" in disease.lower():
                            st.success(recommendation)
                        elif "🚨" in recommendation:
                            st.error(recommendation)
                        elif "⚠️" in recommendation:
                            st.warning(recommendation)
                        else:
                            st.info(recommendation)
                        
                        st.divider()
                        
                        # Additional info
                        with st.expander("📊 Detailed Results"):
                            st.json(result)
                        
                    else:
                        st.error(f"❌ Error: {response.status_code}")
                        st.error(response.text)
                
                except requests.exceptions.ConnectionError:
                    st.error("❌ **Cannot connect to API**")
                    st.info("Make sure the API server is running:")
                    st.code("python api/app.py", language="bash")
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    else:
        st.info("👆 Please upload an image to get started")
        
        # Example section
        st.divider()
        st.subheader("ℹ️ How to Use")
        st.write("""
        1. **Upload** a clear image of a plant leaf
        2. Click **PREDICT** button
        3. View the **plant type** and **disease diagnosis**
        4. Read **treatment recommendations**
        
        **Supported Plants:** Pepper, Potato, Tomato
        
        **Supported Diseases:** Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, and more
        """)