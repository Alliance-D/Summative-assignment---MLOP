"""
Prediction Module for Multi-Output Plant Disease Detection
Handles single and batch predictions using the trained multi-output model
"""

import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Dict, List, Union
import tensorflow as tf
from PIL import Image
import cv2

from config import (
    MULTI_OUTPUT_MODEL,
    CLASS_MAPPINGS_FILE,
    MODEL_METADATA_FILE,
    MODEL_CONFIG,
    PLANT_TO_IDX,
    DISEASE_TO_IDX,
    IDX_TO_PLANT,
    IDX_TO_DISEASE,
)
import json

logger = logging.getLogger(__name__)


class PredictionEngine:
    """Handles multi-output model predictions for plant disease detection"""

    def __init__(self):
        self.model = None
        self.plant_to_idx = {}
        self.disease_to_idx = {}
        self.idx_to_plant = {}
        self.idx_to_disease = {}
        self.model_loaded = False
        
        self.load_model()

    def load_model(self):
        """Load trained multi-output model and class mappings"""
        try:
            # Load model
            if MULTI_OUTPUT_MODEL.exists():
                self.model = tf.keras.models.load_model(str(MULTI_OUTPUT_MODEL))
                logger.info(f" Model loaded from {MULTI_OUTPUT_MODEL}")
            else:
                logger.error(f" Model not found at {MULTI_OUTPUT_MODEL}")
                return False

            # Load class mappings
            if CLASS_MAPPINGS_FILE.exists():
                with open(CLASS_MAPPINGS_FILE, 'r') as f:
                    mappings = json.load(f)
                    self.plant_to_idx = mappings.get('plant_to_idx', {})
                    self.disease_to_idx = mappings.get('disease_to_idx', {})
                    self.idx_to_plant = mappings.get('idx_to_plant', {})
                    self.idx_to_disease = mappings.get('idx_to_disease', {})
                    
                    # Convert string keys to int for idx mappings
                    self.idx_to_plant = {int(k): v for k, v in self.idx_to_plant.items()}
                    self.idx_to_disease = {int(k): v for k, v in self.idx_to_disease.items()}
                    
                logger.info(f" Class mappings loaded: {len(self.plant_to_idx)} plants, {len(self.disease_to_idx)} diseases")
            else:
                logger.warning(f" Class mappings not found at {CLASS_MAPPINGS_FILE}")
                # Use defaults from config
                self.idx_to_plant = IDX_TO_PLANT if IDX_TO_PLANT else {}
                self.idx_to_disease = IDX_TO_DISEASE if IDX_TO_DISEASE else {}

            self.model_loaded = True
            return True

        except Exception as e:
            logger.error(f" Error loading model: {str(e)}")
            self.model_loaded = False
            return False

    def preprocess_image(self, image_input: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Preprocess image for model input
        
        Args:
            image_input: Image path, numpy array, or PIL Image
            
        Returns:
            Preprocessed image array (224, 224, 3)
        """
        try:
            # Load image based on input type
            if isinstance(image_input, str):
                # Load from file path
                image = Image.open(image_input)
                image = image.convert('RGB')
            elif isinstance(image_input, np.ndarray):
                # Already numpy array
                if image_input.dtype == np.uint8:
                    image_input = image_input.astype(np.float32) / 255.0
                image = image_input
            elif isinstance(image_input, Image.Image):
                # PIL Image
                image = image_input.convert('RGB')
                image = np.array(image).astype(np.float32) / 255.0
            else:
                raise ValueError(f"Unsupported image input type: {type(image_input)}")

            # Convert PIL to numpy if needed
            if isinstance(image, Image.Image):
                image = np.array(image).astype(np.float32) / 255.0

            # Resize to model input size
            target_size = (MODEL_CONFIG["input_shape"][0], MODEL_CONFIG["input_shape"][1])
            if image.shape[:2] != target_size:
                image = cv2.resize(image, target_size)

            # Ensure correct shape
            if image.shape != MODEL_CONFIG["input_shape"]:
                logger.warning(f"Image shape mismatch: {image.shape} vs {MODEL_CONFIG['input_shape']}")

            return image

        except Exception as e:
            logger.error(f" Error preprocessing image: {str(e)}")
            return None

    def predict_single(self, image_input: Union[str, np.ndarray, Image.Image]) -> Dict:
        """
        Make prediction on a single image
        
        Args:
            image_input: Image path, numpy array, or PIL Image
            
        Returns:
            Dictionary with predictions and confidence scores
        """
        if not self.model_loaded or self.model is None:
            return {"error": "Model not loaded"}

        try:
            # Preprocess image
            image = self.preprocess_image(image_input)
            if image is None:
                return {"error": "Failed to preprocess image"}

            # Add batch dimension
            image_batch = np.expand_dims(image, axis=0)

            # Make prediction
            predictions = self.model.predict(image_batch, verbose=0)

            # Extract plant predictions
            plant_preds = predictions['plant_output'][0]
            plant_idx = int(np.argmax(plant_preds))
            plant_confidence = float(plant_preds[plant_idx])
            plant_name = self.idx_to_plant.get(plant_idx, f"Unknown_Plant_{plant_idx}")

            # Extract disease predictions
            disease_preds = predictions['disease_output'][0]
            disease_idx = int(np.argmax(disease_preds))
            disease_confidence = float(disease_preds[disease_idx])
            disease_name = self.idx_to_disease.get(disease_idx, f"Unknown_Disease_{disease_idx}")

            # Get top 3 predictions for each
            top_3_plants = self._get_top_n_predictions(
                plant_preds, 
                self.idx_to_plant, 
                n=min(3, len(self.idx_to_plant))
            )
            top_3_diseases = self._get_top_n_predictions(
                disease_preds, 
                self.idx_to_disease, 
                n=min(3, len(self.idx_to_disease))
            )

            # Compile result
            result = {
                "plant_type": plant_name,
                "plant_confidence": plant_confidence,
                "disease": disease_name,
                "disease_confidence": disease_confidence,
                "overall_confidence": (plant_confidence + disease_confidence) / 2,
                "top_3_plants": top_3_plants,
                "top_3_diseases": top_3_diseases,
                "status": "success"
            }

            logger.info(
                f" Prediction: {plant_name} ({plant_confidence:.2%}) - "
                f"{disease_name} ({disease_confidence:.2%})"
            )

            return result

        except Exception as e:
            logger.error(f" Prediction error: {str(e)}")
            return {"error": str(e), "status": "failed"}

    def predict_batch(self, image_inputs: List[Union[str, np.ndarray, Image.Image]]) -> List[Dict]:
        """
        Make predictions on multiple images
        
        Args:
            image_inputs: List of images (paths, arrays, or PIL Images)
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for idx, image_input in enumerate(image_inputs):
            logger.info(f"Processing image {idx + 1}/{len(image_inputs)}")
            
            result = self.predict_single(image_input)
            
            # Add index/identifier
            if isinstance(image_input, str):
                result["image_path"] = image_input
                result["image_name"] = Path(image_input).name
            else:
                result["image_index"] = idx
            
            results.append(result)
        
        logger.info(f" Batch prediction complete: {len(results)} images processed")
        return results

    def predict_from_array(self, image_array: np.ndarray) -> Dict:
        """
        Make prediction from numpy array (for API usage)
        
        Args:
            image_array: Image as numpy array (RGB, uint8 or float32)
            
        Returns:
            Prediction dictionary
        """
        return self.predict_single(image_array)

    @staticmethod
    def _get_top_n_predictions(
        predictions: np.ndarray,
        idx_to_label: Dict[int, str],
        n: int = 3
    ) -> List[Dict]:
        """
        Get top N predictions with labels and confidences
        
        Args:
            predictions: Prediction array
            idx_to_label: Mapping from index to label
            n: Number of top predictions
            
        Returns:
            List of top predictions
        """
        top_indices = np.argsort(predictions)[-n:][::-1]
        
        return [
            {
                "label": idx_to_label.get(int(idx), f"Unknown_{idx}"),
                "confidence": float(predictions[idx])
            }
            for idx in top_indices
        ]

    def get_recommendation(self, disease_name: str) -> str:
        """
        Get treatment recommendation based on disease
        
        Args:
            disease_name: Name of the disease
            
        Returns:
            Recommendation text
        """
        disease_lower = disease_name.lower()
        
        recommendations = {
            "healthy": " The plant appears healthy! Continue regular care and monitoring.",
            "bacterial_spot": " Bacterial spot detected! Apply copper-based bactericide. Remove affected leaves and improve air circulation.",
            "early_blight": " Early blight detected! Apply fungicide (chlorothalonil or mancozeb). Remove infected leaves and avoid overhead watering.",
            "late_blight": " Late blight detected! This is serious - apply fungicide immediately (mefenoxam or chlorothalonil). Remove all infected plants.",
            "leaf_mold": " Leaf mold detected! Improve ventilation, reduce humidity. Apply fungicide if severe.",
            "septoria_leaf_spot": " Septoria leaf spot detected! Remove infected leaves, apply fungicide, and mulch to prevent soil splash.",
            "target_spot": " Target spot detected! Apply fungicide and improve air circulation.",
            "mosaic_virus": " Viral infection detected! Remove infected plants immediately. Control aphids to prevent spread.",
            "yellow_leaf_curl": " Yellow leaf curl virus! Remove infected plants and control whiteflies.",
        }
        
        # Check for keyword matches
        for key, recommendation in recommendations.items():
            if key in disease_lower:
                return recommendation
        
        # Default recommendation
        return " For specific treatment recommendations, consult with a local agricultural extension officer."

    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        info = {
            "model_loaded": self.model_loaded,
            "model_path": str(MULTI_OUTPUT_MODEL) if MULTI_OUTPUT_MODEL.exists() else None,
            "num_plant_classes": len(self.idx_to_plant),
            "num_disease_classes": len(self.idx_to_disease),
            "plant_classes": list(self.idx_to_plant.values()),
            "disease_classes": list(self.idx_to_disease.values()),
        }
        
        # Load metadata if available
        if MODEL_METADATA_FILE.exists():
            with open(MODEL_METADATA_FILE, 'r') as f:
                metadata = json.load(f)
                info.update(metadata)
        
        return info


# Singleton instance for reuse
_prediction_engine_instance = None

def get_prediction_engine() -> PredictionEngine:
    """Get singleton prediction engine instance"""
    global _prediction_engine_instance
    if _prediction_engine_instance is None:
        _prediction_engine_instance = PredictionEngine()
    return _prediction_engine_instance


if __name__ == "__main__":
    # Test the prediction engine
    engine = PredictionEngine()
    
    if engine.model_loaded:
        print(" Prediction engine initialized successfully!")
        info = engine.get_model_info()
        print(f"\n Model Info:")
        print(f"  - Plant classes: {info['num_plant_classes']}")
        print(f"  - Disease classes: {info['num_disease_classes']}")
        print(f"  - Plants: {info['plant_classes']}")
        print(f"  - Diseases: {info['disease_classes'][:5]}...")
    else:
        print(" Failed to initialize prediction engine")