"""
Database Module for Prediction Logging and Analytics
Uses SQLite for persistent storage of prediction history
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import logging

from src.config import DATABASE_CONFIG

logger = logging.getLogger(__name__)


class PredictionDatabase:
    """Manages prediction history and analytics storage"""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DATABASE_CONFIG["db_path"]
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Predictions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        plant_type TEXT NOT NULL,
                        plant_confidence REAL NOT NULL,
                        disease TEXT NOT NULL,
                        disease_confidence REAL NOT NULL,
                        overall_confidence REAL NOT NULL,
                        status TEXT NOT NULL,
                        response_time_ms REAL,
                        error_message TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Retraining history table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS retraining_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        version TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        samples_used INTEGER,
                        epochs INTEGER,
                        plant_accuracy REAL,
                        disease_accuracy REAL,
                        training_time_seconds REAL,
                        status TEXT NOT NULL,
                        notes TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # System metrics table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS system_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        total_predictions INTEGER,
                        successful_predictions INTEGER,
                        failed_predictions INTEGER,
                        avg_response_time_ms REAL,
                        api_uptime_seconds REAL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.commit()
                logger.info(f" Database initialized at {self.db_path}")
        
        except Exception as e:
            logger.error(f" Database initialization failed: {str(e)}")
            raise
    
    def log_prediction(
        self,
        plant_type: str,
        plant_confidence: float,
        disease: str,
        disease_confidence: float,
        overall_confidence: float,
        status: str = "success",
        response_time_ms: Optional[float] = None,
        error_message: Optional[str] = None
    ) -> int:
        """
        Log a prediction to the database
        
        Returns:
            Prediction ID
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO predictions (
                        timestamp, plant_type, plant_confidence,
                        disease, disease_confidence, overall_confidence,
                        status, response_time_ms, error_message
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now().isoformat(),
                    plant_type,
                    plant_confidence,
                    disease,
                    disease_confidence,
                    overall_confidence,
                    status,
                    response_time_ms,
                    error_message
                ))
                
                conn.commit()
                return cursor.lastrowid
        
        except Exception as e:
            logger.error(f" Failed to log prediction: {str(e)}")
            return -1
    
    def get_prediction_stats(self) -> Dict:
        """Get overall prediction statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Total predictions
                cursor.execute("SELECT COUNT(*) FROM predictions")
                total = cursor.fetchone()[0]
                
                # Successful predictions
                cursor.execute("SELECT COUNT(*) FROM predictions WHERE status = 'success'")
                successful = cursor.fetchone()[0]
                
                # Failed predictions
                failed = total - successful
                
                # Success rate
                success_rate = (successful / total * 100) if total > 0 else 0
                
                # Average response time
                cursor.execute("SELECT AVG(response_time_ms) FROM predictions WHERE response_time_ms IS NOT NULL")
                avg_response = cursor.fetchone()[0] or 0
                
                return {
                    "total": total,
                    "successful": successful,
                    "failed": failed,
                    "success_rate": success_rate,
                    "avg_response_time_ms": avg_response
                }
        
        except Exception as e:
            logger.error(f" Failed to get prediction stats: {str(e)}")
            return {"total": 0, "successful": 0, "failed": 0, "success_rate": 0, "avg_response_time_ms": 0}
    
    def get_plant_distribution(self) -> Dict[str, int]:
        """Get prediction count by plant type"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT plant_type, COUNT(*) 
                    FROM predictions 
                    WHERE status = 'success'
                    GROUP BY plant_type
                    ORDER BY COUNT(*) DESC
                """)
                
                return dict(cursor.fetchall())
        
        except Exception as e:
            logger.error(f" Failed to get plant distribution: {str(e)}")
            return {}
    
    def get_disease_distribution(self) -> Dict[str, int]:
        """Get prediction count by disease type"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT disease, COUNT(*) 
                    FROM predictions 
                    WHERE status = 'success'
                    GROUP BY disease
                    ORDER BY COUNT(*) DESC
                """)
                
                return dict(cursor.fetchall())
        
        except Exception as e:
            logger.error(f" Failed to get disease distribution: {str(e)}")
            return {}
    
    def get_confidence_stats(self) -> Dict:
        """Get confidence score statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT 
                        AVG(plant_confidence) as avg_plant_conf,
                        AVG(disease_confidence) as avg_disease_conf,
                        AVG(overall_confidence) as avg_overall_conf,
                        MIN(plant_confidence) as min_plant_conf,
                        MAX(plant_confidence) as max_plant_conf,
                        MIN(disease_confidence) as min_disease_conf,
                        MAX(disease_confidence) as max_disease_conf
                    FROM predictions
                    WHERE status = 'success'
                """)
                
                result = cursor.fetchone()
                
                if result and result[0] is not None:
                    return {
                        "avg_plant_confidence": result[0],
                        "avg_disease_confidence": result[1],
                        "avg_overall_confidence": result[2],
                        "min_plant_confidence": result[3],
                        "max_plant_confidence": result[4],
                        "min_disease_confidence": result[5],
                        "max_disease_confidence": result[6]
                    }
                else:
                    return {}
        
        except Exception as e:
            logger.error(f" Failed to get confidence stats: {str(e)}")
            return {}
    
    def get_response_time_distribution(self, limit: int = 100) -> List[float]:
        """Get recent response times"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT response_time_ms 
                    FROM predictions 
                    WHERE response_time_ms IS NOT NULL
                    ORDER BY id DESC
                    LIMIT ?
                """, (limit,))
                
                return [row[0] for row in cursor.fetchall()]
        
        except Exception as e:
            logger.error(f" Failed to get response times: {str(e)}")
            return []
    
    def get_predictions_over_time(self, days: int = 7) -> List[Dict]:
        """Get prediction counts over time"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT 
                        DATE(timestamp) as date,
                        COUNT(*) as count,
                        SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successful
                    FROM predictions
                    WHERE DATE(timestamp) >= DATE('now', '-' || ? || ' days')
                    GROUP BY DATE(timestamp)
                    ORDER BY date
                """, (days,))
                
                return [
                    {"date": row[0], "total": row[1], "successful": row[2]}
                    for row in cursor.fetchall()
                ]
        
        except Exception as e:
            logger.error(f" Failed to get predictions over time: {str(e)}")
            return []
    
    def log_retraining(
        self,
        version: str,
        samples_used: int,
        epochs: int,
        plant_accuracy: float,
        disease_accuracy: float,
        training_time_seconds: float,
        status: str = "success",
        notes: Optional[str] = None
    ) -> int:
        """Log a retraining session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO retraining_history (
                        version, timestamp, samples_used, epochs,
                        plant_accuracy, disease_accuracy, training_time_seconds,
                        status, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    version,
                    datetime.now().isoformat(),
                    samples_used,
                    epochs,
                    plant_accuracy,
                    disease_accuracy,
                    training_time_seconds,
                    status,
                    notes
                ))
                
                conn.commit()
                return cursor.lastrowid
        
        except Exception as e:
            logger.error(f" Failed to log retraining: {str(e)}")
            return -1
    
    def get_retraining_history(self, limit: int = 10) -> List[Dict]:
        """Get retraining history"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT 
                        version, timestamp, samples_used, epochs,
                        plant_accuracy, disease_accuracy, training_time_seconds,
                        status, notes
                    FROM retraining_history
                    ORDER BY id DESC
                    LIMIT ?
                """, (limit,))
                
                return [
                    {
                        "version": row[0],
                        "timestamp": row[1],
                        "samples_used": row[2],
                        "epochs": row[3],
                        "plant_accuracy": row[4],
                        "disease_accuracy": row[5],
                        "training_time_seconds": row[6],
                        "status": row[7],
                        "notes": row[8]
                    }
                    for row in cursor.fetchall()
                ]
        
        except Exception as e:
            logger.error(f" Failed to get retraining history: {str(e)}")
            return []
    
    def clear_old_data(self, days: int = 30):
        """Delete predictions older than specified days"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    DELETE FROM predictions 
                    WHERE DATE(timestamp) < DATE('now', '-' || ? || ' days')
                """, (days,))
                
                deleted = cursor.rowcount
                conn.commit()
                
                logger.info(f"🗑️ Deleted {deleted} old predictions")
                return deleted
        
        except Exception as e:
            logger.error(f" Failed to clear old data: {str(e)}")
            return 0


# Singleton instance
_db_instance = None

def get_database() -> PredictionDatabase:
    """Get database singleton instance"""
    global _db_instance
    if _db_instance is None:
        _db_instance = PredictionDatabase()
    return _db_instance


if __name__ == "__main__":
    # Test the database
    db = PredictionDatabase()
    
    print(" Database initialized")
    print(f" Stats: {db.get_prediction_stats()}")