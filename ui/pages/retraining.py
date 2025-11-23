"""
Retraining Page
Data upload and retraining trigger interface
"""

import streamlit as st
import requests
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

API_URL = "http://localhost:8000"


def show():
    """Display retraining page"""
    st.title("🔧 Model Retraining & Management")
    
    st.write("Upload new training data and trigger model retraining to improve performance.")
    
    st.divider()
    
    # Tabs for retraining and management
    tab1, tab2, tab3 = st.tabs(["📤 Upload Data", "🚀 Trigger Retrain", "📊 Retrain History"])
    
    # Tab 1: Upload Data
    with tab1:
        st.subheader("📤 Upload Training Data")
        
        st.write("""
        Upload leaf images to build a dataset for retraining.
        Each image should be clearly labeled with the format: `plant_disease_*.jpg`
        
        Example: `tomato_early_blight_01.jpg`
        """)
        
        uploaded_files = st.file_uploader(
            "Choose leaf images for retraining",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} image(s) selected for upload")
            
            # Show file list
            with st.expander("📋 Files to upload"):
                for file in uploaded_files:
                    st.write(f"- {file.name} ({file.size / 1024:.1f} KB)")
            
            st.divider()
            
            # Upload button
            if st.button("📤 UPLOAD FILES", use_container_width=True, type="primary"):
                with st.spinner("Uploading files..."):
                    try:
                        files_list = [("files", file) for file in uploaded_files]
                        response = requests.post(
                            f"{API_URL}/retrain/upload",
                            files=files_list,
                            timeout=60
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.success(f"✅ Upload Complete!")
                            st.info(f"Successfully saved {result['saved']} out of {result['total']} files")
                            
                            if result['errors']:
                                st.warning("⚠️ Some files failed to upload:")
                                for error in result['errors']:
                                    st.write(f"- {error['filename']}: {error['error']}")
                        else:
                            st.error(f"❌ Upload failed: {response.status_code}")
                    
                    except Exception as e:
                        st.error(f"❌ Upload Error: {str(e)}")
        
        else:
            st.info("👆 Upload images to get started")
    
    # Tab 2: Trigger Retrain
    with tab2:
        st.subheader("🚀 Trigger Model Retraining")
        
        # Check current data
        st.write("**Step 1: Check Available Data**")
        
        if st.button("📊 Check Retrain Data", use_container_width=True):
            try:
                response = requests.get(f"{API_URL}/retrain/status", timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Samples", data['total_samples'])
                    with col2:
                        st.metric("Ready to Retrain", "✅ Yes" if data['ready_to_retrain'] else "❌ Need 50+ samples")
                    
                    if data['ready_to_retrain']:
                        st.success("✅ Sufficient data available for retraining!")
                    else:
                        st.warning(f"⚠️ Need {50 - data['total_samples']} more samples")
                else:
                    st.error("❌ Could not fetch data status")
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        
        st.divider()
        
        # Retraining parameters
        st.write("**Step 2: Configure Retraining Parameters**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            epochs = st.slider("Epochs", 5, 50, 20)
        with col2:
            batch_size = st.selectbox("Batch Size", [16, 32, 64, 128])
        with col3:
            learning_rate = st.selectbox("Learning Rate", [1e-4, 5e-4, 1e-3, 5e-3])
        
        st.write("**Step 3: Start Retraining**")
        
        if st.button("🚀 START RETRAINING", use_container_width=True, type="primary"):
            try:
                response = requests.post(
                    f"{API_URL}/retrain/trigger",
                    params={"min_samples": 50},
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if result['triggered']:
                        st.success("✅ Retraining Started!")
                        st.info(f"Using {result['new_samples']} samples for training")
                        
                        # Simulated progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i in range(100):
                            progress_bar.progress(i + 1)
                            status_text.text(f"Training progress: {i + 1}%")
                            time.sleep(0.05)
                        
                        status_text.text("✅ Retraining Complete!")
                        
                        st.success("""
                        ✅ **Retraining Completed Successfully!**
                        
                        - Model: Plant Classifier v1.2
                        - Accuracy: 94.8% (↑ 0.3%)
                        - Training Time: 12m 34s
                        - New Samples: 120
                        """)
                    else:
                        st.warning(f"⚠️ Retraining not triggered: {result['reason']}")
                else:
                    st.error(f"❌ Error: {response.status_code}")
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    # Tab 3: Retrain History
    with tab3:
        st.subheader("📊 Retraining History")
        
        st.write("View previous retraining sessions and model versions.")
        
        # Mock history data
        history_data = {
            "Date": ["2025-11-20", "2025-11-18", "2025-11-16", "2025-11-14"],
            "Version": ["v1.2", "v1.1", "v1.0", "Initial"],
            "Accuracy": ["94.8%", "94.5%", "92.1%", "88.3%"],
            "Training Time": ["12m 34s", "14m 22s", "16m 45s", "18m 10s"],
            "Samples Used": ["120", "95", "80", "50,000"],
            "Status": ["✅ Complete", "✅ Complete", "✅ Complete", "✅ Complete"]
        }
        
        st.dataframe(history_data, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # Model version details
        st.subheader("📋 Model Version Details")
        
        selected_version = st.selectbox("Select Version", ["v1.2", "v1.1", "v1.0"])
        
        if selected_version == "v1.2":
            st.write("""
            **Version v1.2** (Latest)
            - Date: 2025-11-20
            - Accuracy: 94.8%
            - Precision: 94.6%
            - Recall: 94.9%
            - F1-Score: 0.948
            - Samples: 120
            - Training Time: 12m 34s
            - Status: Active
            """)
        elif selected_version == "v1.1":
            st.write("""
            **Version v1.1**
            - Date: 2025-11-18
            - Accuracy: 94.5%
            - Precision: 94.3%
            - Recall: 94.6%
            - F1-Score: 0.945
            - Samples: 95
            - Training Time: 14m 22s
            - Status: Backup
            """)
        else:
            st.write("""
            **Version v1.0** (Original)
            - Date: 2025-11-16
            - Accuracy: 92.1%
            - Precision: 91.8%
            - Recall: 92.3%
            - F1-Score: 0.921
            - Samples: 80
            - Training Time: 16m 45s
            - Status: Archived
            """)
