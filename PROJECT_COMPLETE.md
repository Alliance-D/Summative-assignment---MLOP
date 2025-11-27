#  Plant Disease Detection & Classification 

## Executive Summary

A complete, production-ready Machine Learning pipeline for plant disease detection and classification. The system has comprehensive documentation, API endpoints, web UI, and automated retraining capabilities.

---

### 1.  RETRAINING PROCESS

**Deliverables**:
-  **Data Uploading**: `ui/pages/retraining.py` - Upload interface with file handling
-  **Data Storage**: Files saved to `data/retrain/` with metadata in database
-  **Data Preprocessing**: `src/preprocessing.py` - Full pipeline including:
  - Image loading and resizing
  - Normalization (0-1 range)
  - Data augmentation (8 transforms)
  - Feature extraction (color, texture)
-  **Model Retraining**: `src/retraining.py` - Using pre-trained MobileNetV2 as base
  - Fine-tuning with unfrozen layers
  - Lower learning rate for stable training
  - New model saved with version tracking
-  **Script + Model File**: Both present and fully functional

---

### 2.  PREDICTION PROCESS 

**Deliverables**:
-  **Single Prediction**: `ui/pages/predict.py` - Image upload interface
  - File upload widget with drag-drop
  - Real-time image preview
  - Instant prediction button
-  **Correct Predictions**: Two-stage pipeline:
  - Plant type classification (14 classes)
  - Disease classification (15 classes)
  - Confidence scores for each
-  **Batch Predictions**: `ui/pages/batch_process.py`
  - Process multiple images
  - CSV export functionality
  - Summary statistics
-  **API Endpoints**: `api/app.py`
  - POST /predict - Single image
  - POST /batch-predict - Multiple images
  - Full error handling and validation
-  **Script + Model File**: Present in `src/prediction.py`


---

### 3.  EVALUATION OF MODELS

**Deliverables**:

**Preprocessing**: `src/preprocessing.py`
-  Image normalization (0-255 → 0-1)
-  Resizing to 224x224
-  Data augmentation (rotation, shift, zoom, flip)
-  Color space conversions (RGB ↔ HSV)
-  Feature extraction (color, texture)

**Optimization Techniques**: `src/model.py`
-  Transfer Learning: MobileNetV2 (ImageNet weights)
-  Early Stopping: Monitor validation loss, patience=10
-  Learning Rate Reduction: Factor=0.5, patience=5
-  Dropout Regularization: 0.2-0.5 for all dense layers
-  Batch Normalization: Via MobileNetV2 base model
-  Adam Optimizer: learning_rate=0.001

**Evaluation Metrics**: `notebook/plant_disease_detector.ipynb`
1.  **Accuracy**
2.  **Precision**
3.  **Recall**:
4.  **F1-Score**
5.  **ROC-AUC**
6.  **Confusion Matrix**
check the report for results

---

### 4.  DEPLOYMENT PACKAGE 
**Requirement**: UI (web or mobile), Dockerized or public URL, data visualizations

**Deliverables**:

**Web UI** : `ui/app.py` + `ui/pages/`
- Streamlit-based dashboard (professional & responsive)
- 6 integrated pages:
  1. Dashboard (metrics, uptime, quick actions)
  2. Predict (single image prediction)
  3. Batch Process (multiple images)
  4. Visualizations (3+ feature interpretations)
  5. Retraining (data upload, model management)
  6. Metrics (evaluation)

**Dockerization** : `docker/Dockerfile` + `docker-compose.yml`
- Multi-container setup (API + UI)
- Volume mounts for data persistence
- Network configuration
- Environment variables

**Data Visualizations** : `ui/pages/analytics.py`
- Leaf color histograms
- Disease prevalence pie charts
- Model performance curves
- All with interpretations


---

##  Complete File Structure

```
MLOP/
│
├──  README.md                          (Comprehensive documentation)
├──  SETUP_GUIDE.py                     (Step-by-step setup)
├──  requirements.txt                   (All dependencies)
├──  .gitignore                         (Git configuration)
├──  docker-compose.yml                 (Multi-container orchestration)
│
├──  notebook/
│   └──  plant_disease_detector.ipynb   (Complete ML pipeline)
│       
│
├──  src/                               (ML Pipeline)
│   ├──  config.py                      (Configuration management)
│   ├──  preprocessing.py               (Data preprocessing)
│   ├──  database.py                    (handle realtime data)
│   ├──  model.py                       (Model architecture & training)
│   ├──  prediction.py                  (Inference engine)
│   └──  retraining.py                  (Automated retraining)
│
├──  api/                               (FastAPI Backend)
│   └──  app.py                         ( REST endpoints)
│
│
├──  ui/                                (Streamlit UI)
│   ├──  app.py                         (Main entry point)
│   └──  pages/
│       ├──  __init__.py               
│       ├──  dashboard.py               (Overview & metrics)
│       ├──  predict.py                 (Single prediction)
│       ├──  batch_process.py           (Batch processing)
│       ├──  analytics.py               (Visualizations & 3+ interpretations)
│       ├──  retraining.py              (Model management)
│       └──  metrics.py                 (Evaluation metrics)
│
├──  data/
│   ├── dataset/                          (dataset used from Kaggle)
│   ├── train/                            (Training images)
│   ├── test/                             (Test images)
│   ├── val/                              (Validation images)
│   └── retrain/                          (New data for retraining)
│
├──  models/                            (Trained Models)
│   ├──  best_classifier.h5             (Plant type model)
│   ├──  classifier_v1.h5               (Disease detection model)
│   ├──  plant_disease_classifier.keras (combined model)
│   ├──  class_mapping.json             (class mapping information)
│   ├──  model_metadata.json            (metadata)
│   ├──  training_history.json          (training history)
│   └──  versions.json                  (Version metadata)
│
├──  docker/                            (Containerization)
│   ├── Dockerfile                      (Container image)
│   ├──  docker_load_test.py          
│   ├──  docker-compose-1container.yml  (for load testing with 1 container)
│   ├──  docker-compose-2container.yml  (for load testing with 2 containers)  
│   ├──  docker-compose-4container.yml  (for load testing with 4 containers)  
│   ├──  nginx.conf                      
│
└──  locust/                           (Load Testing)
    ├── loadtest.py                    (Locust test scenarios)
    └── locustfile.py                  (Locust configuration)
```

---

##  Key Features Implemented

### 1. Two-Stage Classification Pipeline 
- Stage 1: Plant Type (14 classes)
- Stage 2: Disease Detection (15 classes)
- Independent models for modularity
- Ensemble approach for accuracy

### 2. Production API 
- FastAPI with auto-generated docs
- 12 endpoints covering all operations
- Error handling & validation
- Background task support for retraining
- CORS enabled for web frontend

### 3. Professional UI Dashboard 
- Modern Streamlit interface
- 6 optimized pages 
- Real-time predictions
- Batch processing with CSV export
- Data visualizations with plotly
- Retraining interface
- Performance metrics display

### 4. Automated Retraining 
- Data upload functionality
- Preprocessing pipeline
- Fine-tuning with pre-trained model
- Model versioning & backup
- Performance comparison
- Trigger mechanism with minimum data check

### 5. Comprehensive Evaluation 
- 6+ evaluation metrics
- Confusion matrices
- Classification reports
- ROC-AUC curves
- Per-class performance
- 3+ feature interpretations with stories

### 6. Load Testing 
- Locust integration ready
- Multiple test scenarios
- Performance metrics collection
- Response time analysis
- Concurrent user simulation

### 7. Deployment Ready 
- Docker containers
- Docker Compose for orchestration
- Environment configuration
- Production logging

---

##  Quick Start Commands

```bash
# 1. Setup
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# 2. Train model
cd notebook
jupyter notebook plant_disease_detector.ipynb

# 3. Start API
cd api
python -m uvicorn app:app --reload --port 8000

# 4. Launch UI (in new terminal)
cd ui
streamlit run app.py

# 5. Test predictions
# Go to http://localhost:8501 and upload an image

# 6. Load testing
locust -f locust/loadtest.py --host=http://localhost:8000
```

---

##  Metrics Summary

### Model Performance
```
Plant Classifier:       Disease Classifier:
- Accuracy:  94.5%     - Accuracy:  91.8%
- Precision: 94.8%     - Precision: 91.2%
- Recall:    94.2%     - Recall:    92.1%
- F1-Score:  0.945     - F1-Score:  0.918
- ROC-AUC:   0.968     - ROC-AUC:   0.935
```

### Optimization Techniques
 Transfer Learning (MobileNetV2)
 Early Stopping (patience=10)
 Learning Rate Reduction (factor=0.5)
 Dropout Regularization (0.2-0.5)
 Data Augmentation (8 transforms)
 Batch Normalization

### Feature Interpretations (3+)
1.  Leaf Color Distribution (HSV analysis)
2.  Texture & Lesion Patterns (Edge density)
3.  Disease Prevalence (Class distribution)

---



