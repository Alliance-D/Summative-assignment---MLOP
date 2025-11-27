"""
FastAPI Backend for Multi-Output Plant Disease Detection
Provides REST API endpoints for predictions, retraining, and management
"""

import os
import sys
from pathlib import Path


# Add parent directory to path BEFORE other imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from datetime import datetime
from typing import List, Optional
import io
import time

import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# NOW import your modules (after path is set)
from src.prediction import PredictionEngine
from src.retraining import RetrainingPipeline
from src.database import get_database
from src.config import (
    API_CONFIG,
    RETRAIN_DATA_DIR,
    LOGS_DIR,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Plant Disease Detection API",
    description="Multi-output deep learning API for plant type and disease classification",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://your-app.streamlit.app",  # Your Streamlit URL
        "http://localhost:8501",  # Local development
        "*"  # Remove this in production for security
    ], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
prediction_engine: Optional[PredictionEngine] = None
retraining_pipeline: Optional[RetrainingPipeline] = None

# Global status tracking
api_status = {
    "started_at": datetime.now().isoformat(),
    "model_loaded": False,
    "predictions_count": 0,
    "successful_predictions": 0,
    "failed_predictions": 0,
    "retraining_in_progress": False,
    "last_retrain": None,
}


# PYDANTIC MODELS (Request/Response Schemas)

class PredictionResponse(BaseModel):
    """Response model for single prediction"""
    plant_type: str
    plant_confidence: float
    disease: str
    disease_confidence: float
    overall_confidence: float
    top_3_plants: List[dict]
    top_3_diseases: List[dict]
    recommendation: Optional[str] = None
    status: str


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    results: List[dict]
    total_processed: int
    successful: int
    failed: int


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    model_loaded: bool
    timestamp: str
    uptime_seconds: float


class StatusResponse(BaseModel):
    """Response model for API status"""
    status: str
    model_loaded: bool
    predictions_count: int
    successful_predictions: int
    failed_predictions: int
    uptime: str
    retraining_in_progress: bool


class RetrainDataResponse(BaseModel):
    """Response model for retrain data upload"""
    uploaded: int
    total: int
    failed: int
    errors: List[dict]
    message: str


class RetrainTriggerResponse(BaseModel):
    """Response model for retrain trigger"""
    triggered: bool
    reason: str
    timestamp: str
    new_samples: int


class RetrainStatusResponse(BaseModel):
    """Response model for retrain status"""
    total_samples: int
    ready_to_retrain: bool
    min_required: int
    message: str


# STARTUP & SHUTDOWN EVENTS

@app.on_event("startup")
async def startup_event():
    """Initialize models and services on startup"""
    global prediction_engine, retraining_pipeline, api_status
    
    logger.info("="*70)
    logger.info(" Starting Plant Disease Detection API")
    logger.info("="*70)
    
    try:
        # Initialize database
        logger.info("🗄️ Initializing database...")
        db = get_database()
        logger.info(" Database initialized")
        
        # Initialize prediction engine
        logger.info(" Loading prediction engine...")
        prediction_engine = PredictionEngine()
        
        if prediction_engine.model_loaded:
            api_status["model_loaded"] = True
            logger.info(" Prediction engine loaded successfully")
            
            # Log model info
            model_info = prediction_engine.get_model_info()
            logger.info(f" Model Info:")
            logger.info(f"   - Plant classes: {model_info['num_plant_classes']}")
            logger.info(f"   - Disease classes: {model_info['num_disease_classes']}")
        else:
            logger.error(" Failed to load prediction engine")
            api_status["model_loaded"] = False
        
        # Initialize retraining pipeline
        logger.info(" Initializing retraining pipeline...")
        retraining_pipeline = RetrainingPipeline()
        logger.info(" Retraining pipeline initialized")
        
        logger.info("="*70)
        logger.info(" API startup complete!")
        logger.info("="*70)
        
    except Exception as e:
        logger.error(f" Startup error: {str(e)}")
        api_status["model_loaded"] = False


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info(" Shutting down API...")


# HEALTH & STATUS ENDPOINTS

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - API welcome"""
    return {
        "message": " Plant Disease Detection API",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Detailed health check endpoint"""
    uptime = (datetime.now() - datetime.fromisoformat(api_status["started_at"])).total_seconds()
    
    return HealthResponse(
        status="healthy" if api_status["model_loaded"] else "unhealthy",
        model_loaded=api_status["model_loaded"],
        timestamp=datetime.now().isoformat(),
        uptime_seconds=uptime
    )


@app.get("/status", response_model=StatusResponse, tags=["Status"])
async def get_status():
    """Get comprehensive API and model status"""
    return StatusResponse(
        status="running" if api_status["model_loaded"] else "degraded",
        model_loaded=api_status["model_loaded"],
        predictions_count=api_status["predictions_count"],
        successful_predictions=api_status["successful_predictions"],
        failed_predictions=api_status["failed_predictions"],
        uptime=api_status["started_at"],
        retraining_in_progress=api_status["retraining_in_progress"]
    )


@app.get("/model/info", tags=["Model"])
async def get_model_info():
    """Get detailed model information"""
    if not prediction_engine or not prediction_engine.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return prediction_engine.get_model_info()


# PREDICTION ENDPOINTS

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_single(file: UploadFile = File(...)):
    """
    Make a single prediction on uploaded leaf image
    
    Args:
        file: Image file (JPEG, PNG, BMP)
        
    Returns:
        Prediction results with plant type and disease classification
    """
    if not api_status["model_loaded"] or not prediction_engine:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if file.content_type and not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Additional fallback: check file extension if content_type is None
    if not file.content_type:
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
            )
    
    start_time = time.time()  # Track response time
    db = get_database()
    
    try:
        # Read and decode image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Make prediction
        result = prediction_engine.predict_from_array(image)
        
        # Calculate response time
        response_time_ms = (time.time() - start_time) * 1000
        
        if result.get("status") == "failed" or "error" in result:
            # Log failed prediction
            db.log_prediction(
                plant_type="Unknown",
                plant_confidence=0.0,
                disease="Unknown",
                disease_confidence=0.0,
                overall_confidence=0.0,
                status="failed",
                response_time_ms=response_time_ms,
                error_message=result.get("error", "Unknown error")
            )
            
            api_status["failed_predictions"] += 1
            raise HTTPException(status_code=400, detail=result.get("error", "Prediction failed"))
        
        # Add recommendation
        result["recommendation"] = prediction_engine.get_recommendation(result["disease"])
        
        # Log successful prediction
        db.log_prediction(
            plant_type=result["plant_type"],
            plant_confidence=result["plant_confidence"],
            disease=result["disease"],
            disease_confidence=result["disease_confidence"],
            overall_confidence=result["overall_confidence"],
            status="success",
            response_time_ms=response_time_ms
        )
        
        # Update stats
        api_status["predictions_count"] += 1
        api_status["successful_predictions"] += 1
        
        logger.info(
            f" Prediction #{api_status['predictions_count']}: "
            f"{result['plant_type']} - {result['disease']} "
            f"({response_time_ms:.0f}ms)"
        )
        
        return PredictionResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        response_time_ms = (time.time() - start_time) * 1000
        
        # Log error
        db.log_prediction(
            plant_type="Unknown",
            plant_confidence=0.0,
            disease="Unknown",
            disease_confidence=0.0,
            overall_confidence=0.0,
            status="failed",
            response_time_ms=response_time_ms,
            error_message=str(e)
        )
        
        api_status["failed_predictions"] += 1
        logger.error(f" Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Make batch predictions on multiple images
    
    Args:
        files: List of image files
        
    Returns:
        Batch prediction results
    """
    if not api_status["model_loaded"] or not prediction_engine:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(files) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 images per batch")
    
    results = []
    successful = 0
    failed = 0
    
    for idx, file in enumerate(files):
        try:
            # Read image
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                results.append({
                    "filename": file.filename,
                    "status": "failed",
                    "error": "Invalid image format"
                })
                failed += 1
                continue
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Predict
            result = prediction_engine.predict_from_array(image)
            result["filename"] = file.filename
            result["index"] = idx
            
            if result.get("status") == "success":
                result["recommendation"] = prediction_engine.get_recommendation(result["disease"])
                successful += 1
            else:
                failed += 1
            
            results.append(result)
            api_status["predictions_count"] += 1
            
        except Exception as e:
            logger.error(f" Batch prediction error for {file.filename}: {str(e)}")
            results.append({
                "filename": file.filename,
                "index": idx,
                "status": "failed",
                "error": str(e)
            })
            failed += 1
    
    api_status["successful_predictions"] += successful
    api_status["failed_predictions"] += failed
    
    logger.info(f" Batch prediction: {successful}/{len(files)} successful")
    
    return BatchPredictionResponse(
        results=results,
        total_processed=len(files),
        successful=successful,
        failed=failed
    )


# RETRAINING ENDPOINTS

@app.post("/retrain/upload", response_model=RetrainDataResponse, tags=["Retraining"])
async def upload_retrain_data(files: List[UploadFile] = File(...)):
    """
    Upload images for model retraining
    
    Expected filename format: PlantName___DiseaseName_*.jpg
    Example: Tomato___Early_Blight_001.jpg
    
    Args:
        files: List of labeled image files
        
    Returns:
        Upload status
    """
    if not retraining_pipeline:
        raise HTTPException(status_code=503, detail="Retraining pipeline not initialized")
    
    db = get_database()
    uploaded = 0
    failed = 0
    errors = []
    
    # Ensure retrain directory exists
    RETRAIN_DATA_DIR.mkdir(parents=True, exist_ok=True)
    plant_village_dir = RETRAIN_DATA_DIR / "PlantVillage"
    plant_village_dir.mkdir(exist_ok=True)
    
    for file in files:
        try:
            # Parse filename to get class
            # Expected: PlantName___DiseaseName_*.jpg
            filename = file.filename
            
            # Extract class from filename (before last underscore and number)
            # E.g., "Tomato___Early_Blight_001.jpg" -> "Tomato___Early_Blight"
            parts = filename.rsplit('_', 1)[0] if '_' in filename else filename.split('.')[0]
            
            if '___' not in parts:
                errors.append({
                    "filename": filename,
                    "error": "Invalid filename format. Expected: PlantName___DiseaseName_*.jpg"
                })
                failed += 1
                continue
            
            # Create class directory
            class_dir = plant_village_dir / parts
            class_dir.mkdir(exist_ok=True)
            
            # Save file
            save_path = class_dir / filename
            contents = await file.read()
            
            with open(save_path, "wb") as f:
                f.write(contents)
            
            uploaded += 1
            logger.info(f" Saved: {save_path}")
            
        except Exception as e:
            errors.append({
                "filename": file.filename,
                "error": str(e)
            })
            failed += 1
            logger.error(f" Upload error for {file.filename}: {str(e)}")
    
    message = f"Successfully uploaded {uploaded}/{len(files)} files"
    logger.info(message)
    
    # Log upload to database (optional tracking)
    try:
        db.log_retraining(
            version="upload",
            samples_used=uploaded,
            epochs=0,
            plant_accuracy=0.0,
            disease_accuracy=0.0,
            training_time_seconds=0.0,
            status="upload",
            notes=f"Uploaded {uploaded} files for retraining"
        )
    except:
        pass  # Don't fail if logging fails
    
    return RetrainDataResponse(
        uploaded=uploaded,
        total=len(files),
        failed=failed,
        errors=errors,
        message=message
    )


@app.get("/retrain/status", response_model=RetrainStatusResponse, tags=["Retraining"])
async def get_retrain_status():
    """Check status of retraining data"""
    if not retraining_pipeline:
        raise HTTPException(status_code=503, detail="Retraining pipeline not initialized")
    
    status = retraining_pipeline.check_new_data()
    
    return RetrainStatusResponse(
        total_samples=status["total_samples"],
        ready_to_retrain=status["ready_to_retrain"],
        min_required=status["min_required"],
        message=status["message"]
    )


@app.post("/retrain/trigger", response_model=RetrainTriggerResponse, tags=["Retraining"])
async def trigger_retraining(
    background_tasks: BackgroundTasks,
    min_samples: int = Query(50, description="Minimum samples required for retraining"),
    epochs: int = Query(20, description="Number of training epochs")
):
    """
    Trigger model retraining with uploaded data
    
    Args:
        min_samples: Minimum samples required
        epochs: Number of training epochs
        
    Returns:
        Retraining trigger status
    """
    if not retraining_pipeline or not prediction_engine:
        raise HTTPException(status_code=503, detail="Services not initialized")
    
    if api_status["retraining_in_progress"]:
        raise HTTPException(status_code=409, detail="Retraining already in progress")
    
    # Check if conditions are met
    trigger_status = retraining_pipeline.trigger_retrain(min_samples)
    
    if trigger_status["triggered"]:
        # Add background task
        background_tasks.add_task(
            perform_retraining,
            retraining_pipeline,
            prediction_engine,
            epochs
        )
        api_status["retraining_in_progress"] = True
        logger.info(" Retraining task started in background")
    
    return RetrainTriggerResponse(
        triggered=trigger_status["triggered"],
        reason=trigger_status["reason"],
        timestamp=trigger_status["timestamp"],
        new_samples=trigger_status["new_samples"]
    )


async def perform_retraining(pipeline: RetrainingPipeline, engine: PredictionEngine, epochs: int):
    """Background task for actual retraining"""
    db = get_database()
    start_time = time.time()
    
    try:
        logger.info("="*70)
        logger.info("STARTING MODEL RETRAINING")
        logger.info("="*70)
        
        # Retrain
        results = pipeline.retrain_model(epochs=epochs, fine_tune=True)
        
        training_time = time.time() - start_time
        
        if results["status"] == "success":
            # Save new model
            new_version = pipeline.save_retrained_model()
            logger.info(f"Model saved as {new_version}")
            
            # Log to database
            final_metrics = results.get("final_metrics", {})
            
            db.log_retraining(
                version=new_version,
                samples_used=results.get("samples_used", 0),  # This now shows total uploaded
                epochs=epochs,
                plant_accuracy=final_metrics.get("plant_val_acc", 0.0),
                disease_accuracy=final_metrics.get("disease_val_acc", 0.0),
                training_time_seconds=training_time,
                status="success",
                notes=f"Retraining completed: {results.get('train_samples', 0)} train, {results.get('val_samples', 0)} val samples"
            )
            
            # Reload in prediction engine
            engine.load_model()
            logger.info("Prediction engine reloaded with new model")
            
            # CLEAR RETRAIN DATA after successful retraining
            if pipeline.clear_retrain_data():
                logger.info("Retrain data cleared - ready for next session")
            
            api_status["last_retrain"] = datetime.now().isoformat()
            logger.info("="*70)
            logger.info("RETRAINING COMPLETED SUCCESSFULLY")
            logger.info("="*70)
        else:
            # Log failure
            db.log_retraining(
                version="failed",
                samples_used=results.get("samples_used", 0),
                epochs=epochs,
                plant_accuracy=0.0,
                disease_accuracy=0.0,
                training_time_seconds=training_time,
                status="failed",
                notes=results.get('error', 'Unknown error')
            )
            logger.error(f"Retraining failed: {results.get('error')}")
        
    except Exception as e:
        training_time = time.time() - start_time
        
        # Log error to database
        db.log_retraining(
            version="error",
            samples_used=0,
            epochs=epochs,
            plant_accuracy=0.0,
            disease_accuracy=0.0,
            training_time_seconds=training_time,
            status="error",
            notes=str(e)
        )
        logger.error(f"Retraining error: {str(e)}")
    finally:
        api_status["retraining_in_progress"] = False


# METRICS ENDPOINTS

@app.get("/metrics", tags=["Metrics"])
async def get_metrics():
    """Get API usage metrics from database"""
    db = get_database()
    pred_stats = db.get_prediction_stats()
    uptime_seconds = (datetime.now() - datetime.fromisoformat(api_status["started_at"])).total_seconds()
    
    return {
        "predictions": pred_stats,
        "uptime": {
            "started_at": api_status["started_at"],
            "uptime_seconds": uptime_seconds,
            "uptime_hours": uptime_seconds / 3600
        },
        "model": {
            "loaded": api_status["model_loaded"],
            "last_retrain": api_status["last_retrain"]
        },
        "timestamp": datetime.now().isoformat()
    }


@app.get("/metrics/summary", tags=["Metrics"])
async def get_metrics_summary():
    """Get summarized metrics for dashboard"""
    return {
        "total_predictions": api_status["predictions_count"],
        "model_status": "online" if api_status["model_loaded"] else "offline",
        "uptime": api_status["started_at"],
        "last_retrain": api_status["last_retrain"],
    }


# ANALYTICS ENDPOINTS 


@app.get("/analytics/plants", tags=["Analytics"])
async def get_plant_analytics():
    """Get prediction distribution by plant type"""
    db = get_database()
    distribution = db.get_plant_distribution()
    
    return {
        "plant_distribution": distribution,
        "total_plants": len(distribution),
        "most_common": max(distribution.items(), key=lambda x: x[1])[0] if distribution else None
    }


@app.get("/analytics/diseases", tags=["Analytics"])
async def get_disease_analytics():
    """Get prediction distribution by disease type"""
    db = get_database()
    distribution = db.get_disease_distribution()
    
    return {
        "disease_distribution": distribution,
        "total_diseases": len(distribution),
        "most_common": max(distribution.items(), key=lambda x: x[1])[0] if distribution else None
    }


@app.get("/analytics/confidence", tags=["Analytics"])
async def get_confidence_analytics():
    """Get confidence score statistics"""
    db = get_database()
    stats = db.get_confidence_stats()
    
    return {
        "confidence_stats": stats,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/analytics/response-times", tags=["Analytics"])
async def get_response_time_analytics(limit: int = Query(100, description="Number of recent predictions")):
    """Get response time distribution"""
    db = get_database()
    response_times = db.get_response_time_distribution(limit)
    
    if response_times:
        return {
            "response_times": response_times,
            "count": len(response_times),
            "avg": sum(response_times) / len(response_times),
            "min": min(response_times),
            "max": max(response_times)
        }
    else:
        return {
            "response_times": [],
            "count": 0,
            "message": "No response time data available"
        }


@app.get("/analytics/timeline", tags=["Analytics"])
async def get_timeline_analytics(days: int = Query(7, description="Number of days to analyze")):
    """Get prediction timeline"""
    db = get_database()
    timeline = db.get_predictions_over_time(days)
    
    return {
        "timeline": timeline,
        "days_analyzed": days,
        "total_predictions": sum(item["total"] for item in timeline)
    }


@app.get("/analytics/summary", tags=["Analytics"])
async def get_analytics_summary():
    """Get comprehensive analytics summary"""
    db = get_database()
    
    return {
        "prediction_stats": db.get_prediction_stats(),
        "plant_distribution": db.get_plant_distribution(),
        "disease_distribution": db.get_disease_distribution(),
        "confidence_stats": db.get_confidence_stats(),
        "recent_predictions": db.get_predictions_over_time(7),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/retrain/history", tags=["Retraining"])
async def get_retrain_history(limit: int = Query(10, description="Number of recent retraining sessions")):
    """
    Get retraining history from database
    
    Args:
        limit: Number of records to return
        
        Returns:
        List of retraining sessions
    """
    if not retraining_pipeline:
        raise HTTPException(status_code=503, detail="Retraining pipeline not initialized")
    
    try:
        db = get_database()
        history = db.get_retraining_history(limit)
        
        return history
    
    except Exception as e:
        logger.error(f" Error fetching retraining history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

#endpoint to clear retraining data
    app.delete("/retrain/data", tags=["Retraining"])
async def clear_retrain_data():
    """
    Manually clear all uploaded retraining data
    
    Returns:
        Status of data clearing operation
    """
    if not retraining_pipeline:
        raise HTTPException(status_code=503, detail="Retraining pipeline not initialized")
    
    if api_status["retraining_in_progress"]:
        raise HTTPException(status_code=409, detail="Cannot clear data while retraining is in progress")
    
    try:
        success = retraining_pipeline.clear_retrain_data()
        
        if success:
            return {
                "status": "success",
                "message": "Retrain data cleared successfully",
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to clear retrain data")
    
    except Exception as e:
        logger.error(f"Error clearing retrain data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 


# ERROR HANDLERS

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f" Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error": str(exc)
        }
    )
    


# MAIN


if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment variable (Render provides this)
    port = int(os.getenv("PORT", API_CONFIG["port"]))
    
    # Detect if running in production
    is_production = os.getenv("RENDER") is not None
    
    logger.info(f"Starting FastAPI server on port {port}...")
    
    uvicorn.run(
        app,
        host="0.0.0.0",  # Must be 0.0.0.0 for Render
        port=port,
        reload=not is_production,  # Disable reload in production
        log_level="info"
    )