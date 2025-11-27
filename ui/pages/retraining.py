"""
Retraining Page
Data upload and retraining trigger interface
"""
import pandas as pd
import streamlit as st
import requests
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

API_URL = "http://localhost:8000"


def show():
    """Display retraining page"""
    st.title(" Model Retraining & Management")
    
    st.write("Upload new training data and trigger model retraining to improve performance.")
    
    st.divider()
    
    # Tabs for retraining and management
    tab1, tab2, tab3 = st.tabs([" Upload Data", " Trigger Retrain", " Retrain History"])
    
    # Tab 1: Upload Data
    with tab1:
        st.subheader(" Upload Training Data")
        
        st.info("""
        ** Filename Format Required:**
        
        Your images MUST follow this naming convention:
        - `PlantName___DiseaseName_number.jpg`
        - Example: `Tomato___Early_Blight_001.jpg`
        - Example: `Pepper___Bacterial_Spot_042.jpg`
        
        **Note:** Three underscores (___) between plant and disease name!
        """)
        
        uploaded_files = st.file_uploader(
            "Choose leaf images for retraining",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            help="Images must follow naming convention: PlantName___DiseaseName_*.jpg"
        )
        
        if uploaded_files:
            st.success(f" {len(uploaded_files)} image(s) selected for upload")
            
            # Validate filenames
            valid_files = []
            invalid_files = []
            
            for file in uploaded_files:
                # Check filename format
                parts = file.name.rsplit('_', 1)[0] if '_' in file.name else file.name.split('.')[0]
                if '___' in parts:
                    valid_files.append(file)
                else:
                    invalid_files.append(file.name)
            
            # Show file list
            with st.expander(" Files to upload"):
                if valid_files:
                    st.success(f" Valid files ({len(valid_files)}):")
                    for file in valid_files:
                        st.write(f"- {file.name} ({file.size / 1024:.1f} KB)")
                
                if invalid_files:
                    st.error(f" Invalid filenames ({len(invalid_files)}):")
                    for fname in invalid_files:
                        st.write(f"- {fname}")
                    st.warning(" These files will NOT be uploaded due to incorrect naming format")
            
            st.divider()
            
            # Upload button (only if there are valid files)
            if valid_files:
                if st.button("📤 UPLOAD FILES", use_container_width=True, type="primary"):
                    with st.spinner("Uploading files..."):
                        try:
                            # Only upload valid files
                            files_list = [("files", file) for file in valid_files]
                            response = requests.post(
                                f"{API_URL}/retrain/upload",
                                files=files_list,
                                timeout=120
                            )
                            
                            if response.status_code == 200:
                                result = response.json()
                                st.success(f" Upload Complete!")
                                st.info(f"Successfully uploaded {result['uploaded']} out of {result['total']} files")
                                
                                if result['errors']:
                                    st.warning(" Some files failed to upload:")
                                    for error in result['errors']:
                                        st.write(f"- {error['filename']}: {error['error']}")
                            else:
                                st.error(f" Upload failed: {response.status_code}")
                                st.error(response.text)
                        
                        except Exception as e:
                            st.error(f" Upload Error: {str(e)}")
            else:
                st.warning(" No valid files to upload. Please check filename format.")
        
        else:
            st.info("💡 Upload images to get started")
            st.write("**Example filenames:**")
            st.code("""
Tomato___Early_Blight_001.jpg
Tomato___Late_Blight_002.jpg
Pepper___Bacterial_Spot_001.jpg
Potato___Early_Blight_001.jpg
Tomato___healthy_001.jpg
            """)
    
    # Tab 2: Trigger Retrain
    with tab2:
        st.subheader(" Trigger Model Retraining")
        
        # Check current data
        st.write("**Step 1: Check Available Data**")
        
        if st.button(" Check Retrain Data", use_container_width=True):
            try:
                response = requests.get(f"{API_URL}/retrain/status", timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Samples", data['total_samples'])
                    with col2:
                        st.metric("Minimum Required", data['min_required'])
                    with col3:
                        ready_status = " Ready" if data['ready_to_retrain'] else " Not Ready"
                        st.metric("Status", ready_status)
                    
                    st.info(f"💡 {data['message']}")
                    
                    # Store in session state
                    st.session_state['retrain_data'] = data
                else:
                    st.error(" Could not fetch data status")
            
            except Exception as e:
                st.error(f" Error: {str(e)}")
        
        st.divider()
        
        # Retraining parameters
        st.write("**Step 2: Configure Retraining Parameters**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            epochs = st.slider("Epochs", 5, 50, 20, help="Number of training epochs")
        with col2:
            st.info(f"**Minimum Samples:** 50 (automatic)")
        
        st.divider()
        
        st.write("**Step 3: Start Retraining**")
        
        # Check if ready
        ready_to_train = st.session_state.get('retrain_data', {}).get('ready_to_retrain', False)
        
        if not ready_to_train:
            st.warning(" Not enough data for retraining. Click 'Check Retrain Data' first.")
        
        if st.button(" START RETRAINING", use_container_width=True, type="primary", disabled=not ready_to_train):
            try:
                # Trigger retraining with automatic min_samples=50
                response = requests.post(
                    f"{API_URL}/retrain/trigger",
                    params={"min_samples": 50, "epochs": epochs},
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if result['triggered']:
                        st.success(" Retraining Started!")
                        st.info(f"Using {result['new_samples']} samples for training")
                        
                        st.info("""
                         **Retraining in Progress**

                        Training is running in the background. This may take 5-15 minutes.
                        
                        **To check progress:**
                        1. Wait a few minutes
                        2. Go to "Retrain History" tab
                        3. Refresh to see the latest entry
                        
                        You can continue using the app while training runs.
                        """)
                    else:
                        st.warning(f" Retraining not triggered: {result['reason']}")
                else:
                    st.error(f" Error: {response.status_code}")
                    st.error(response.text)
            
            except Exception as e:
                st.error(f" Error: {str(e)}")
                
        st.divider()

        st.write("**Step 4: Clear Training Data (Optional)**")

        if st.button("Clear All Uploaded Data", type="secondary", use_container_width=True):
            try:
                response = requests.delete(f"{API_URL}/retrain/data", timeout=10)
                
                if response.status_code == 200:
                    st.success("All retrain data cleared! You can upload new samples now.")
                    # Clear session state
                    if 'retrain_data' in st.session_state:
                        del st.session_state['retrain_data']
                else:
                    st.error(f"Failed to clear data: {response.text}")
            except Exception as e:
                st.error(f"Error: {str(e)}")

        st.info("Note: Training data is automatically cleared after successful retraining")
    
    # Tab 3: Retrain History
    with tab3:
        st.subheader(" Retraining History")
        
        st.write("View previous retraining sessions and model versions.")
        
        # Add refresh button
        if st.button(" Refresh History", use_container_width=True):
            st.rerun()
        
        st.divider()
        
        try:
            # Fetch REAL retraining history from database
            history_response = requests.get(f"{API_URL}/retrain/history?limit=20", timeout=10)
            
            if history_response.status_code == 200:
                history_data = history_response.json()
                
                # Filter out upload entries for main table
                training_sessions = [h for h in history_data if h['status'] != 'upload']
                
                if training_sessions and len(training_sessions) > 0:
                    # Convert to DataFrame
                    df = pd.DataFrame(training_sessions)
                    
                    # Format columns
                    if 'timestamp' in df.columns:
                        df['Date'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
                    
                    if 'training_time_seconds' in df.columns:
                        df['Training Time'] = df['training_time_seconds'].apply(
                            lambda x: f"{int(x//60)}m {int(x%60)}s" if pd.notnull(x) and x > 0 else "N/A"
                        )
                    
                    if 'plant_accuracy' in df.columns:
                        df['Plant Acc'] = df['plant_accuracy'].apply(
                            lambda x: f"{x*100:.1f}%" if pd.notnull(x) and x > 0 else "N/A"
                        )
                    
                    if 'disease_accuracy' in df.columns:
                        df['Disease Acc'] = df['disease_accuracy'].apply(
                            lambda x: f"{x*100:.1f}%" if pd.notnull(x) and x > 0 else "N/A"
                        )
                    
                    # Add status emoji
                    if 'status' in df.columns:
                        status_map = {
                            'success': ' Success',
                            'failed': ' Failed',
                            'error': ' Error',
                            'in_progress': ' Running'
                        }
                        df['Status'] = df['status'].map(lambda x: status_map.get(x, x))
                    
                    # Select and display columns
                    display_cols = []
                    col_names = {}
                    
                    for col, display_name in [
                        ('version', 'Version'),
                        ('Date', 'Date'),
                        ('samples_used', 'Samples'),
                        ('epochs', 'Epochs'),
                        ('Plant Acc', 'Plant Acc'),
                        ('Disease Acc', 'Disease Acc'),
                        ('Training Time', 'Time'),
                        ('Status', 'Status')
                    ]:
                        if col in df.columns:
                            display_cols.append(col)
                            col_names[col] = display_name
                    
                    display_df = df[display_cols].rename(columns=col_names)
                    
                    # Show table
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
                    
                    st.divider()
                    
                    # Show details of selected version
                    st.subheader(" Version Details")
                    
                    versions = [item['version'] for item in training_sessions]
                    selected_version = st.selectbox("Select Version", versions)
                    
                    # Find selected version data
                    version_data = next((item for item in training_sessions if item['version'] == selected_version), None)
                    
                    if version_data:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Version:** `{version_data['version']}`")
                            st.write(f"**Date:** {version_data['timestamp'][:19]}")
                            st.write(f"**Samples Used:** {version_data.get('samples_used', 'N/A')}")
                            st.write(f"**Epochs:** {version_data.get('epochs', 'N/A')}")
                        
                        with col2:
                            plant_acc = version_data.get('plant_accuracy')
                            disease_acc = version_data.get('disease_accuracy')
                            
                            if plant_acc and plant_acc > 0:
                                st.metric("Plant Accuracy", f"{plant_acc*100:.2f}%")
                            else:
                                st.write("**Plant Accuracy:** N/A")
                            
                            if disease_acc and disease_acc > 0:
                                st.metric("Disease Accuracy", f"{disease_acc*100:.2f}%")
                            else:
                                st.write("**Disease Accuracy:** N/A")
                            
                            training_time = version_data.get('training_time_seconds')
                            if training_time and training_time > 0:
                                st.write(f"**Training Time:** {int(training_time//60)}m {int(training_time%60)}s")
                            
                            status = version_data.get('status', 'N/A')
                            status_color = {
                                'success': 'green',
                                'failed': 'red',
                                'error': 'orange'
                            }.get(status, 'gray')
                            st.write(f"**Status:** :{status_color}[{status.upper()}]")
                        
                        if version_data.get('notes'):
                            st.info(f"💡 **Notes:** {version_data['notes']}")
                
                else:
                    st.info(" No retraining history yet. Trigger a retraining to see history here.")
            
            else:
                st.warning(" Could not fetch retraining history from API")
                if history_response.status_code == 404:
                    st.error("Endpoint not found. Make sure your API is updated.")
        
        except requests.exceptions.ConnectionError:
            st.error(" Cannot connect to API. Make sure it's running at http://localhost:8000")
        except Exception as e:
            st.error(f" Error fetching history: {str(e)}")