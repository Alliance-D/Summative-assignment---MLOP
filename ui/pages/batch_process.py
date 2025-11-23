"""
Batch Processing Page
Upload and process multiple images at once
"""

import streamlit as st
import requests
from PIL import Image
import io
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

API_URL = "http://localhost:8000"


def show():
    """Display batch processing page"""
    st.title("📦 Batch Image Processing")
    
    st.write("Upload multiple leaf images for bulk disease prediction.")
    
    st.divider()
    
    # File uploader
    st.subheader("📤 Upload Multiple Images")
    
    uploaded_files = st.file_uploader(
        "Choose leaf images",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
        help="Upload up to 50 images at once"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} image(s) selected")
        
        if len(uploaded_files) > 50:
            st.error("❌ Maximum 50 images allowed per batch")
            return
        
        # Show thumbnails
        with st.expander(f"📋 Preview ({len(uploaded_files)} images)"):
            cols = st.columns(5)
            for idx, file in enumerate(uploaded_files[:10]):  # Show first 10
                with cols[idx % 5]:
                    image = Image.open(file)
                    st.image(image, use_container_width=True, caption=file.name[:15])
            if len(uploaded_files) > 10:
                st.info(f"... and {len(uploaded_files) - 10} more images")
        
        st.divider()
        
        # Process button
        if st.button("🚀 PROCESS BATCH", use_container_width=True, type="primary"):
            with st.spinner(f"🔄 Processing {len(uploaded_files)} images..."):
                try:
                    # Prepare files for API
                    files = []
                    for file in uploaded_files:
                        file.seek(0)  # Reset file pointer
                        files.append(("files", (file.name, file.read(), file.type)))
                    
                    # Send to API
                    response = requests.post(
                        f"{API_URL}/predict/batch",
                        files=files,
                        timeout=120
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        st.success("✅ Batch Processing Complete!")
                        st.divider()
                        
                        # Summary
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Processed", result["total_processed"])
                        with col2:
                            st.metric("Successful", result["successful"])
                        with col3:
                            st.metric("Failed", result["failed"])
                        
                        st.divider()
                        
                        # Results table
                        st.subheader("📊 Results")
                        
                        results_data = []
                        for item in result["results"]:
                            if item.get("status") == "success":
                                results_data.append({
                                    "Filename": item["filename"],
                                    "Plant": item["plant_type"],
                                    "Disease": item["disease"],
                                    "Plant Conf.": f"{item['plant_confidence']*100:.1f}%",
                                    "Disease Conf.": f"{item['disease_confidence']*100:.1f}%",
                                    "Status": "✅"
                                })
                            else:
                                results_data.append({
                                    "Filename": item["filename"],
                                    "Plant": "N/A",
                                    "Disease": "N/A",
                                    "Plant Conf.": "N/A",
                                    "Disease Conf.": "N/A",
                                    "Status": "❌"
                                })
                        
                        st.dataframe(results_data, use_container_width=True, hide_index=True)
                        
                        st.divider()
                        
                        # Download results
                        st.subheader("💾 Export Results")
                        
                        import json
                        json_str = json.dumps(result, indent=2)
                        
                        st.download_button(
                            label="📥 Download JSON",
                            data=json_str,
                            file_name="batch_predictions.json",
                            mime="application/json"
                        )
                        
                        # CSV export
                        import pandas as pd
                        df = pd.DataFrame(results_data)
                        csv = df.to_csv(index=False)
                        
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv,
                            file_name="batch_predictions.csv",
                            mime="text/csv"
                        )
                    
                    else:
                        st.error(f"❌ Error: {response.status_code}")
                        st.error(response.text)
                
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to API")
                    st.info("Make sure the API is running:")
                    st.code("python api/app.py", language="bash")
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    else:
        st.info(" Upload images to get started")
        
        st.divider()
        st.subheader("ℹ️ How to Use")
        st.write("""
        1. **Upload** multiple leaf images (up to 50)
        2. **Preview** thumbnails of uploaded images
        3. Click **PROCESS BATCH** to analyze all images
        4. **View results** in a table format
        5. **Download** results as JSON or CSV
        
        **Best for:** Processing multiple plants at once, farm surveys, bulk diagnosis
        """)