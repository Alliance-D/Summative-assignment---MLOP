"""
Configuration file for the Plant Disease Detection ML Pipeline
"""
import os
from pathlib import Path
import json

# Detect environment
IS_PRODUCTION = os.getenv("RENDER") is not None

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DATA_DIR = DATA_DIR / "train"
TEST_DATA_DIR = DATA_DIR / "test"
RETRAIN_DATA_DIR = DATA_DIR / "retrain"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for directory in [TRAIN_DATA_DIR, TEST_DATA_DIR, RETRAIN_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Model paths - UPDATED for multi-output model
MULTI_OUTPUT_MODEL = MODELS_DIR / "plant_disease_model.keras"
CLASS_MAPPINGS_FILE = MODELS_DIR / "class_mappings.json"
MODEL_METADATA_FILE = MODELS_DIR / "model_metadata.json"
TRAINING_HISTORY_FILE = MODELS_DIR / "training_history.json"

# Load class mappings if they exist
def load_class_mappings():
    """Load class mappings from file"""
    if CLASS_MAPPINGS_FILE.exists():
        with open(CLASS_MAPPINGS_FILE, 'r') as f:
            return json.load(f)
    return None

# Get class mappings
_MAPPINGS = load_class_mappings()

if _MAPPINGS:
    PLANT_TO_IDX = _MAPPINGS.get('plant_to_idx', {})
    DISEASE_TO_IDX = _MAPPINGS.get('disease_to_idx', {})
    IDX_TO_PLANT = _MAPPINGS.get('idx_to_plant', {})
    IDX_TO_DISEASE = _MAPPINGS.get('idx_to_disease', {})
    NUM_PLANT_CLASSES = len(PLANT_TO_IDX)
    NUM_DISEASE_CLASSES = len(DISEASE_TO_IDX)
    PLANT_TYPES = list(PLANT_TO_IDX.keys())
    DISEASE_TYPES = list(DISEASE_TO_IDX.keys())
else:
    # Default values (will be updated when model is loaded)
    NUM_PLANT_CLASSES = 3
    NUM_DISEASE_CLASSES = 15
    PLANT_TYPES = ["Pepper", "Potato", "Tomato"]
    DISEASE_TYPES = []
    PLANT_TO_IDX = {}
    DISEASE_TO_IDX = {}
    IDX_TO_PLANT = {}
    IDX_TO_DISEASE = {}

# Model Configuration
MODEL_CONFIG = {
    "input_shape": (224, 224, 3),
    "num_plant_classes": NUM_PLANT_CLASSES,
    "num_disease_classes": NUM_DISEASE_CLASSES,
    "batch_size": 64,
    "epochs": 30,
    "learning_rate": 0.001,
    "validation_split": 0.15,
    "test_split": 0.15,
}

# Data Preprocessing
PREPROCESSING_CONFIG = {
    "rescale": 1.0 / 255.0,
    "rotation_range": 15,
    "width_shift_range": 0.15,
    "height_shift_range": 0.15,
    "shear_range": 0.1,
    "zoom_range": 0.15,
    "horizontal_flip": True,
    "vertical_flip": False,  
    "fill_mode": "nearest",
    "brightness_range": [0.8, 1.2],
}

# API Configuration
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": True,
    "workers": 4,
}

# Database Configuration
DATABASE_CONFIG = {
    "db_path": PROJECT_ROOT / "data" / "app.db",
    "connection_string": f"sqlite:///{PROJECT_ROOT / 'data' / 'app.db'}",
}

# Logging
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": LOGS_DIR / "app.log",
}

# Streamlit Configuration
STREAMLIT_CONFIG = {
    "max_upload_size": 200,  # MB
    "theme": "light",
}

# Feature interpretation thresholds
FEATURE_INTERPRETATION = {
    "color_threshold": 0.7,
    "texture_threshold": 0.6,
    "severity_threshold": 0.5,
}

# Retraining Configuration
RETRAINING_CONFIG = {
    "min_samples": 50,
    "backup_old_models": True,
    "auto_deploy": True,
    "improvement_threshold": -0.02,  # Allow up to 2% accuracy decrease
}