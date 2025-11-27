"""
Locust Load Testing Script for Plant Disease Detection API
"""

from locust import HttpUser, task, between
import random
import os
from pathlib import Path

# Sample image paths (you'll need to provide test images)
SAMPLE_IMAGES = [
    "test_images/sample1.jpg",
    "test_images/sample2.jpg",
    "test_images/sample3.jpg",
]


class PlantDiseaseUser(HttpUser):
    """Simulates a user making predictions"""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    @task(3)
    def health_check(self):
        """Health check endpoint (lightweight)"""
        self.client.get("/health")
    
    @task(2)
    def get_status(self):
        """Get API status"""
        self.client.get("/status")
    
    @task(1)
    def get_metrics(self):
        """Get metrics"""
        self.client.get("/metrics")
    
    @task(5)
    def predict_image(self):
        """Make a prediction (main workload)"""
        # Check if test images exist
        if not SAMPLE_IMAGES or not Path(SAMPLE_IMAGES[0]).exists():
            # Create a dummy image in memory
            import io
            from PIL import Image
            import numpy as np
            
            # Create a random image
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            
            # Save to bytes
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG')
            img_bytes.seek(0)
            
            files = {"file": ("test.jpg", img_bytes, "image/jpeg")}
        else:
            # Use actual test image
            image_path = random.choice(SAMPLE_IMAGES)
            files = {"file": open(image_path, "rb")}
        
        response = self.client.post("/predict", files=files)
        
        # Close file if opened
        if isinstance(files["file"], tuple):
            pass  # BytesIO, no need to close
        else:
            files["file"].close()
    
    @task(1)
    def get_analytics_summary(self):
        """Get analytics summary"""
        self.client.get("/analytics/summary")


class AdminUser(HttpUser):
    """Simulates admin operations"""
    
    wait_time = between(5, 10)
    
    @task(1)
    def check_retrain_status(self):
        """Check retraining status"""
        self.client.get("/retrain/status")
    
    @task(1)
    def get_analytics(self):
        """Get detailed analytics"""
        self.client.get("/analytics/plants")
        self.client.get("/analytics/diseases")
        self.client.get("/analytics/confidence")
        
    """
    Run Locust from command line:
    
    locust -f locust/loadtest.py --host=http://localhost:8000
    
    Then open http://localhost:8089 in your browser to start the load test.
    
    Example scenarios:
    1. Gradual increase: 10 users/sec for 5 minutes
    2. Spike test: Sudden increase to 100 users
    3. Soak test: Run 50 users for 30 minutes
    """