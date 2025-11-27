#  Plant Disease Detection & Classification

## Executive Summary

A complete, production-ready Machine Learning pipeline for plant disease detection and classification. The system has comprehensive documentation, API endpoints, web UI, and automated retraining capabilities.

African smallholder farmers face significant crop losses due to plant diseases, with limited access to expert diagnosis. Traditional methods of disease identification require experienced agronomists who are scarce in rural areas. 
So this project aims to develop an intelligent agricultural monitoring system designed to assist smallholder farmers in identifying crop diseases early. The system uses deep learning to analyze leaf images and provide two critical pieces of information; the type of plant and its health status. 

---

##  Model Performance
for performance check the report attached at the bottom

---

##  Data Split Strategy

**Three-Way Split (70% / 15% / 15%)**:

```
Total Dataset
├──  Training (70%)    → Used to train model weights via backpropagation
├──  Validation (15%)  → Used during training for early stopping & hyperparameter tuning
└──  Test (15%)        → Used ONLY for final evaluation (model never sees this during training)
```

**Why This Split?**
- Training set: Large enough to learn robust patterns
- Validation set: Prevents overfitting by monitoring unseen data
- Test set: Provides unbiased evaluation of final model performance

---

##  Quick Start

### Prerequisites
- Python 3.9+
- Docker & Docker Compose (optional)
- 8GB+ RAM recommended
- Kaggle account (for dataset download)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd plant_disease_detection
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

This installs everything including `kagglehub` for dataset management.

---

##  Usage

### 1. Setup Kaggle Credentials (First Time Only)
```bash
# 1. Go to: https://www.kaggle.com/settings/account
# 2. Click "Create New API Token"
# 3. Place kaggle.json in: ~/.kaggle/kaggle.json
#    (C:\Users\<YourUsername>\.kaggle\ on Windows)

# 4. Install kagglehub
pip install kagglehub
```

### 2. Run the Jupyter Notebook
```bash
cd notebook
jupyter notebook plant_disease_detector.ipynb
```

**What happens:**
- Cell 1: Downloads PlantVillage dataset via kagglehub (auto-extracts)
- Cell 2-3: Configuration and data augmentation setup
- Cell 4: Loads data and creates 70/15/15 train/val/test split
- Cell 5+: Trains models and evaluates on test set


### 3. Start the FastAPI Backend
```bash
cd api
python -m uvicorn app:app --reload --port 8000
```

**API Documentation**: Open http://localhost:8000/docs

### 4. Launch the Streamlit UI
```bash
cd ui
streamlit run app.py
```

**UI Access**: Open http://localhost:8501

### 5. Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build individual images
docker build -f docker/Dockerfile -t plant-disease-api:latest .
docker run -p 8000:8000 plant-disease-api:latest
```

---


### Optimization Techniques

 **Transfer Learning**
- ImageNet pre-trained weights
- Faster convergence
- Better accuracy with limited data

 **Data Augmentation**
- Rotation 
- Shift 
- Shear & Zoom 
- Horizontal/Vertical Flip

 **Training Optimization**
- Early Stopping (patience=10)
- Learning Rate Reduction (factor=0.5)
- Adam Optimizer (lr=0.001)
- Dropout Regularization (0.2-0.5)

---

##  API Endpoints

### Health & Status
```bash
GET /                       # API info
GET /health                 # Health check
GET /status                 # API status
GET /metrics                # API metrics
```

### Predictions
```bash
POST /predict               # Single image prediction
POST /batch-predict         # Multiple images batch

# Example:
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@leaf_image.jpg"
```

### Retraining
```bash
POST /retrain/upload        # Upload training data
POST /retrain/trigger       # Trigger retraining
GET  /retrain/status        # Retraining data status
```

---

## Load Testing with Locust

### Run Load Tests
```bash
locust -f locust/loadtest.py --host=http://localhost:8000
```

Then open http://localhost:8089 to start the test.




---

## TROUBLESHOOTING

Issue: "Models not found" error
- Run Jupyter notebook first to train models

Issue: "Connection refused" when accessing API
- Make sure API is running: python -m uvicorn app:app --reload

Issue: "Port already in use"
- Change port: streamlit run app.py --server.port 8502

Issue: Out of memory
- Reduce batch size in config.py

Issue: Slow predictions
- Use CPU-only TensorFlow or upgrade GPU


##  Security & Deployment

### Environment Variables
Create `.env` file:
```env
API_HOST=0.0.0.0
API_PORT=8000
DB_PATH=data/app.db
LOG_LEVEL=INFO
```

### Database
- SQLite for local development

### Monitoring
- Prometheus metrics exposed
- Log aggregation ready
- Health check endpoints
- Performance monitoring

---
##  Acknowledgments

- **PlantVillage Dataset**: Data for model training
- **TensorFlow/Keras**: Deep learning framework
- **Streamlit**: Web UI framework
- **FastAPI**: High-performance API framework
---

##  Demo Video

**YouTube Link**: 

## For full documentation: 
- check: `PROJECT_COMPLETE.md`

## Report
- Link: 

## Locust report

