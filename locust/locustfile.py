"""
Locust Load Testing File for Plant Disease Detection API
Tests prediction endpoints under load
"""

from locust import HttpUser, task, between
import random
from pathlib import Path
import io
from PIL import Image
import numpy as np


class PlantDiseaseAPIUser(HttpUser):
    """Simulates a user making requests to the Plant Disease API"""
    
    # Wait between 1-3 seconds between requests
    wait_time = between(1, 3)
    
    def on_start(self):
        """Called when a user starts - setup test data"""
        # Create a dummy test image in memory
        self.test_image = self._create_test_image()
    
    def _create_test_image(self):
        """Create a dummy leaf image for testing"""
        # Create a 224x224 RGB image with random leaf-like colors
        img_array = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array, 'RGB')
        
        # Convert to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        return img_bytes.getvalue()
    
    @task(5)  # Weight: 5 (runs 5x more often than other tasks)
    def predict_single(self):
        """Test single prediction endpoint"""
        files = {
            'file': ('test_leaf.jpg', io.BytesIO(self.test_image), 'image/jpeg')
        }
        
        with self.client.post(
            "/predict",
            files=files,
            catch_response=True,
            name="/predict [single]"
        ) as response:
            if response.status_code == 200:
                result = response.json()
                if result.get('status') == 'success':
                    response.success()
                else:
                    response.failure(f"Prediction failed: {result.get('error')}")
            else:
                response.failure(f"Got status code {response.status_code}")
    
    @task(1)  # Weight: 1
    def check_health(self):
        """Test health endpoint"""
        with self.client.get("/health", catch_response=True, name="/health") as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")
    
    @task(1)
    def get_status(self):
        """Test status endpoint"""
        with self.client.get("/status", catch_response=True, name="/status") as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status check failed: {response.status_code}")
    
    @task(1)
    def get_model_info(self):
        """Test model info endpoint"""
        with self.client.get("/model/info", catch_response=True, name="/model/info") as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Model info failed: {response.status_code}")
    
    @task(1)
    def get_metrics(self):
        """Test metrics endpoint"""
        with self.client.get("/metrics", catch_response=True, name="/metrics") as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Metrics failed: {response.status_code}")
    
    @task(2)
    def get_analytics(self):
        """Test analytics endpoints"""
        endpoints = [
            "/analytics/plants",
            "/analytics/diseases",
            "/analytics/confidence"
        ]
        endpoint = random.choice(endpoints)
        
        with self.client.get(endpoint, catch_response=True, name="/analytics/*") as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Analytics failed: {response.status_code}")


class HeavyLoadUser(HttpUser):
    """Simulates heavy batch prediction users"""
    
    wait_time = between(3, 5)
    
    def on_start(self):
        """Create multiple test images"""
        self.test_images = [self._create_test_image() for _ in range(5)]
    
    def _create_test_image(self):
        """Create a dummy leaf image"""
        img_array = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array, 'RGB')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        return img_bytes.getvalue()
    
    @task
    def batch_predict(self):
        """Test batch prediction with multiple images"""
        files = [
            ('files', (f'test_{i}.jpg', io.BytesIO(img), 'image/jpeg'))
            for i, img in enumerate(self.test_images)
        ]
        
        with self.client.post(
            "/predict/batch",
            files=files,
            catch_response=True,
            name="/predict/batch [5 images]"
        ) as response:
            if response.status_code == 200:
                result = response.json()
                if result.get('successful', 0) > 0:
                    response.success()
                else:
                    response.failure("No successful predictions in batch")
            else:
                response.failure(f"Batch prediction failed: {response.status_code}")


# Custom load shapes (optional - for advanced testing)
from locust import LoadTestShape

class StepLoadShape(LoadTestShape):
    """
    A step load shape that increases users gradually
    
    Step 1: 10 users for 60 seconds
    Step 2: 25 users for 60 seconds
    Step 3: 50 users for 60 seconds
    Step 4: 100 users for 60 seconds
    """
    
    step_time = 60
    step_load = 10
    spawn_rate = 5
    time_limit = 240
    
    def tick(self):
        run_time = self.get_run_time()
        
        if run_time > self.time_limit:
            return None
        
        current_step = int(run_time // self.step_time)
        return (self.step_load * (current_step + 1), self.spawn_rate)