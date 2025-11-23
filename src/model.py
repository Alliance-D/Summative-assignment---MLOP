"""
Multi-Output Model for Plant Disease Detection
Single model with dual outputs: Plant Type + Disease Classification
"""

import numpy as np
import logging
import json
from pathlib import Path
from typing import Tuple, Dict
from datetime import datetime

import tensorflow as tf
from tensorflow.keras import layers, models, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    TensorBoard,
)

from config import (
    MODEL_CONFIG,
    MULTI_OUTPUT_MODEL,
    CLASS_MAPPINGS_FILE,
    MODEL_METADATA_FILE,
    TRAINING_HISTORY_FILE,
    LOGS_DIR,
)

logger = logging.getLogger(__name__)


class MultiOutputPlantDiseaseDetector:
    """
    Multi-output CNN for simultaneous plant type and disease classification
    Uses shared MobileNetV2 backbone with two classification heads
    """

    def __init__(
        self,
        input_shape=MODEL_CONFIG["input_shape"],
        num_plant_classes=MODEL_CONFIG["num_plant_classes"],
        num_disease_classes=MODEL_CONFIG["num_disease_classes"],
    ):
        self.input_shape = input_shape
        self.num_plant_classes = num_plant_classes
        self.num_disease_classes = num_disease_classes
        self.model = None
        self.history = None
        self.plant_to_idx = {}
        self.disease_to_idx = {}

    def build_model(self):
        """
        Build multi-output model with shared backbone
        
        Returns:
            Compiled Keras model with dual outputs
        """
        logger.info("Building multi-output model with MobileNetV2 backbone")

        # Input layer
        input_layer = layers.Input(shape=self.input_shape, name='input')

        # Shared backbone - MobileNetV2
        base_model = MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False  # Freeze initially

        # Shared feature extraction
        x = base_model(input_layer)
        x = layers.GlobalAveragePooling2D(name='global_pool')(x)

        # Shared dense layers
        x = layers.Dense(512, activation='relu', kernel_regularizer=l2(0.01), name='shared_dense1')(x)
        x = layers.BatchNormalization(name='shared_bn1')(x)
        x = layers.Dropout(0.5, name='shared_dropout1')(x)

        x = layers.Dense(256, activation='relu', kernel_regularizer=l2(0.01), name='shared_dense2')(x)
        x = layers.BatchNormalization(name='shared_bn2')(x)
        x = layers.Dropout(0.4, name='shared_dropout2')(x)

        # PLANT CLASSIFICATION HEAD
        plant_branch = layers.Dense(128, activation='relu', name='plant_dense')(x)
        plant_branch = layers.Dropout(0.3, name='plant_dropout')(plant_branch)
        plant_output = layers.Dense(
            self.num_plant_classes, 
            activation='softmax', 
            name='plant_output'
        )(plant_branch)

        # DISEASE CLASSIFICATION HEAD
        disease_branch = layers.Dense(128, activation='relu', name='disease_dense')(x)
        disease_branch = layers.Dropout(0.3, name='disease_dropout')(disease_branch)
        disease_output = layers.Dense(
            self.num_disease_classes, 
            activation='softmax', 
            name='disease_output'
        )(disease_branch)

        # Create model with dual outputs
        self.model = Model(
            inputs=input_layer,
            outputs={'plant_output': plant_output, 'disease_output': disease_output},
            name='multi_output_plant_disease_detector'
        )

        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=MODEL_CONFIG["learning_rate"]),
            loss={
                'plant_output': 'categorical_crossentropy',
                'disease_output': 'categorical_crossentropy'
            },
            loss_weights={'plant_output': 1.0, 'disease_output': 1.0},
            metrics={
                'plant_output': ['accuracy'],
                'disease_output': ['accuracy']
            }
        )

        logger.info(f"Multi-output model built successfully")
        logger.info(f"  - Plant classes: {self.num_plant_classes}")
        logger.info(f"  - Disease classes: {self.num_disease_classes}")
        logger.info(f"  - Total parameters: {self.model.count_params():,}")

        return self.model

    def train(self, train_gen, val_gen, epochs=None):
        """
        Train the multi-output model
        
        Args:
            train_gen: Training data generator
            val_gen: Validation data generator
            epochs: Number of epochs
            
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()

        if epochs is None:
            epochs = MODEL_CONFIG["epochs"]

        logger.info(f"Training multi-output model for {epochs} epochs")

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                verbose=1,
                min_lr=1e-7
            ),
            ModelCheckpoint(
                str(MULTI_OUTPUT_MODEL),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            TensorBoard(log_dir=str(LOGS_DIR / "multi_output_model"))
        ]

        # Train
        self.history = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

        logger.info("Training completed")
        return self.history

    def save_model(self, save_mappings=True):
        """Save model and metadata"""
        if self.model is None:
            logger.warning("No model to save")
            return

        # Save model
        self.model.save(str(MULTI_OUTPUT_MODEL))
        logger.info(f"Model saved to {MULTI_OUTPUT_MODEL}")

        # Save class mappings
        if save_mappings and (self.plant_to_idx or self.disease_to_idx):
            mappings = {
                'plant_to_idx': self.plant_to_idx,
                'disease_to_idx': self.disease_to_idx,
                'idx_to_plant': {v: k for k, v in self.plant_to_idx.items()},
                'idx_to_disease': {v: k for k, v in self.disease_to_idx.items()}
            }
            with open(CLASS_MAPPINGS_FILE, 'w') as f:
                json.dump(mappings, f, indent=2)
            logger.info(f"Class mappings saved to {CLASS_MAPPINGS_FILE}")

        # Save metadata
        metadata = {
            'model_name': 'Multi-Output Plant Disease Detector',
            'version': '1.0.0',
            'created_at': datetime.now().isoformat(),
            'architecture': 'MobileNetV2 with dual output heads',
            'input_shape': list(self.input_shape),
            'num_plant_classes': self.num_plant_classes,
            'num_disease_classes': self.num_disease_classes,
            'total_parameters': int(self.model.count_params())
        }
        with open(MODEL_METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata saved to {MODEL_METADATA_FILE}")

    def load_model(self):
        """Load saved model"""
        if not MULTI_OUTPUT_MODEL.exists():
            logger.error(f"Model not found at {MULTI_OUTPUT_MODEL}")
            return False

        try:
            self.model = tf.keras.models.load_model(str(MULTI_OUTPUT_MODEL))
            logger.info(f"Model loaded from {MULTI_OUTPUT_MODEL}")

            # Load class mappings
            if CLASS_MAPPINGS_FILE.exists():
                with open(CLASS_MAPPINGS_FILE, 'r') as f:
                    mappings = json.load(f)
                    self.plant_to_idx = mappings['plant_to_idx']
                    self.disease_to_idx = mappings['disease_to_idx']
                logger.info("Class mappings loaded")

            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

    def set_class_mappings(self, plant_to_idx: Dict, disease_to_idx: Dict):
        """Set class mappings from data generator"""
        self.plant_to_idx = plant_to_idx
        self.disease_to_idx = disease_to_idx
        logger.info(f"Class mappings set: {len(plant_to_idx)} plants, {len(disease_to_idx)} diseases")

    def summary(self):
        """Print model summary"""
        if self.model:
            self.model.summary()
        else:
            logger.warning("No model to summarize")


if __name__ == "__main__":
    detector = MultiOutputPlantDiseaseDetector()
    detector.build_model()
    detector.summary()