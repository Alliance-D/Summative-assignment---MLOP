"""
Setup and Execution Guide
Complete instructions for setting up and running the Plant Disease Detection system
"""

QUICK_START = """
═════════════════════════════════════════════════════════════════
  PLANT DISEASE DETECTION - QUICK START GUIDE
═════════════════════════════════════════════════════════════════

 PREREQUISITES:
- Python 3.9+
- pip or conda
- 4GB+ RAM
- (Optional) Docker & Docker Compose

═════════════════════════════════════════════════════════════════
STEP 1: ENVIRONMENT SETUP
═════════════════════════════════════════════════════════════════

1.1 Create virtual environment:
    python -m venv venv
    
1.2 Activate virtual environment:
    Windows: venv\\Scripts\\activate
    Linux/Mac: source venv/bin/activate

1.3 Install dependencies:
    pip install -r requirements.txt

═════════════════════════════════════════════════════════════════
STEP 2: DATA PREPARATION
═════════════════════════════════════════════════════════════════

2.1 Download PlantVillage Dataset:
    - From Kaggle
    - Visit: https://github.com/spMohanty/PlantVillage-Dataset
    - Download the dataset
    
2.2 Organize data structure:
    data/
    ├── train/      (70% of images)
    ├── test/       (15% of images)
    ├── val/       (15% of images)
    └── retrain/    (for future retraining)

═════════════════════════════════════════════════════════════════
STEP 3: MODEL TRAINING (Jupyter Notebook)
═════════════════════════════════════════════════════════════════

3.1 Navigate to notebook:
    cd notebook

3.2 Start Jupyter:
    jupyter notebook

3.3 Open plant_disease_detector.ipynb

3.4 Run all cells sequentially:
    - Cell 1: Import libraries
    - Cell 2: Configuration
    - Cell 3: Data preprocessing
    - Cell 4: Model architecture
    - Cell 5: Training
    - Cell 6: Evaluation
    - Cell 7: Feature interpretation

3.5 Output:
    - Models saved to models/
    - Training history saved
    - Metrics displayed in notebook

═════════════════════════════════════════════════════════════════
STEP 4: START API SERVER
═════════════════════════════════════════════════════════════════

4.1 Navigate to api:
    cd api

4.2 Start FastAPI:
    python -m uvicorn app:app --reload --port 8000

4.3 Verify API:
    - Open: http://localhost:8000/docs
    - Should see Swagger UI with all endpoints

4.4 Health check:
    curl http://localhost:8000/health

═════════════════════════════════════════════════════════════════
STEP 5: LAUNCH WEB DASHBOARD
═════════════════════════════════════════════════════════════════

5.1 Navigate to ui:
    cd ui

5.2 Start Streamlit:
    streamlit run app.py

5.3 Access dashboard:
    - Automatically opens: http://localhost:8501
    - Or manually visit that URL

5.4 Dashboard pages:
    1. Dashboard - Overview & metrics
    2. Predict - Single image prediction
    3. Batch Process - Multiple images
    4. Visualizations - Feature interpretation
    5. Retraining - Model updates
    6. Metrics - Evaluation results

═════════════════════════════════════════════════════════════════
STEP 6: TEST PREDICTIONS
═════════════════════════════════════════════════════════════════

6.1 Via Streamlit UI:
    - Go to "Predict" page
    - Upload a leaf image
    - Click "PREDICT"
    - View results

6.2 Via API (curl):
    curl -X POST "http://localhost:8000/predict" \\
      -H "Content-Type: multipart/form-data" \\
      -F "file=@path/to/image.jpg"

6.3 Via Python:
    import requests
    with open('image.jpg', 'rb') as f:
        r = requests.post('http://localhost:8000/predict',
                         files={'file': f})
    print(r.json())

═════════════════════════════════════════════════════════════════
STEP 7: BATCH PROCESSING
═════════════════════════════════════════════════════════════════

7.1 Via Streamlit:
    - Go to "Batch Process" page
    - Upload multiple images
    - Click "PROCESS BATCH"
    - Download CSV results

7.2 Via API:
    curl -X POST "http://localhost:8000/batch-predict" \\
      -F "files=@image1.jpg" \\
      -F "files=@image2.jpg"

═════════════════════════════════════════════════════════════════
STEP 8: UPLOAD DATA & RETRAIN
═════════════════════════════════════════════════════════════════

8.1 Via Streamlit:
    - Go to "Retraining" page
    - Tab 1: Upload new leaf images
    - Tab 2: Configure parameters
    - Click "START RETRAINING"
    - Monitor progress bar

8.2 Via API:
    # Upload data
    curl -X POST "http://localhost:8000/retrain/upload" \\
      -F "files=@newdata1.jpg" \\
      -F "files=@newdata2.jpg"
    
    # Trigger retrain
    curl -X POST "http://localhost:8000/retrain/trigger"

8.3 Check status:
    curl http://localhost:8000/retrain/status

═════════════════════════════════════════════════════════════════
STEP 9: LOAD TESTING
═════════════════════════════════════════════════════════════════

9.1 Install Locust:
    pip install locust

9.2 Start Locust:
    locust -f locust/loadtest.py --host=http://localhost:8000

9.3 Open UI:
    - Visit: http://localhost:8089
    - Set number of users: 100
    - Set spawn rate: 10
    - Click "Start swarming"

9.4 View results:
    - Charts tab: Response times
    - Statistics tab: Requests/sec
    - Failures tab: Error rates

═════════════════════════════════════════════════════════════════
STEP 10: DOCKER DEPLOYMENT
═════════════════════════════════════════════════════════════════

10.1 Build images:
    docker-compose build

10.2 Start containers:
    docker-compose up

10.3 Access services:
    - API: http://localhost:8000
    - UI: http://localhost:8501

10.4 Stop containers:
    docker-compose down

10.5 View logs:
    docker-compose logs -f api

═════════════════════════════════════════════════════════════════
TROUBLESHOOTING
═════════════════════════════════════════════════════════════════

Issue: "Models not found" error
→ Run Jupyter notebook first to train models

Issue: "Connection refused" when accessing API
→ Make sure API is running: python -m uvicorn app:app --reload

Issue: "Port already in use"
→ Change port: streamlit run app.py --server.port 8502

Issue: Out of memory
→ Reduce batch size in config.py

Issue: Slow predictions
→ Use CPU-only TensorFlow or upgrade GPU

═════════════════════════════════════════════════════════════════
SYSTEM REQUIREMENTS
═════════════════════════════════════════════════════════════════

Minimum:
- CPU: 2 cores
- RAM: 4GB
- Disk: 2GB

Recommended:
- CPU: 4+ cores
- RAM: 8GB+
- GPU: NVIDIA (for faster training)
- Disk: 10GB+

═════════════════════════════════════════════════════════════════
PROJECT STRUCTURE
═════════════════════════════════════════════════════════════════

plant_disease_detection/
├── notebook/              # Jupyter notebook
├── src/                   # Source code (ML pipeline)
├── api/                   # FastAPI backend
├── ui/                    # Streamlit frontend
├── data/                  # Datasets
├── models/                # Trained models
├── docker/                # Docker configuration
├── locust/                # Load testing
└── README.md              # Documentation

═════════════════════════════════════════════════════════════════
NEXT STEPS
═════════════════════════════════════════════════════════════════

1. Train models: Jupyter notebook → Run all cells
2. Start API: python -m uvicorn api/app:app --reload
3. Launch UI: streamlit run ui/app.py
4. Make predictions: Upload image → Get results
5. Upload data: Use UI to upload new images
6. Retrain model: Trigger retraining with new data
7. Load test: Use Locust for performance testing
8. Deploy: Use Docker for cloud deployment

═════════════════════════════════════════════════════════════════
SUPPORT & RESOURCES
═════════════════════════════════════════════════════════════════

Documentation: README.md
Issue Tracker: GitHub Issues
Demo Video: [YouTube Link]
API Docs: http://localhost:8000/docs

═════════════════════════════════════════════════════════════════
"""

print(QUICK_START)

# Save to file
with open('SETUP_GUIDE.txt', 'w') as f:
    f.write(QUICK_START)

print("\n✅ Setup guide created: SETUP_GUIDE.txt")
