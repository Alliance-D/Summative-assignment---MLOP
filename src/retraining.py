"""
Retraining Pipeline for Multi-Output Model 
Handles automated model retraining with new data
"""

import numpy as np
import logging
from pathlib import Path
from datetime import datetime
import json
import shutil
from typing import Tuple, Dict, Optional, List
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

from src.model import MultiOutputPlantDiseaseDetector
from src.config import (
    MULTI_OUTPUT_MODEL,
    CLASS_MAPPINGS_FILE,
    MODEL_METADATA_FILE,
    RETRAIN_DATA_DIR,
    MODELS_DIR,
    MODEL_CONFIG,
    PREPROCESSING_CONFIG,
    LOGS_DIR,
    RETRAINING_CONFIG,
)

logger = logging.getLogger(__name__)


class MultiLabelDataGenerator(tf.keras.utils.Sequence):
    """
    Custom generator that returns images with dual labels (plant + disease)
    """
    def __init__(self, directory, datagen, plant_to_idx, disease_to_idx, 
                 img_size=(224, 224), batch_size=32, shuffle=True):
        self.directory = directory
        self.datagen = datagen
        self.img_size = img_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.plant_to_idx = plant_to_idx
        self.disease_to_idx = disease_to_idx
        
        # Collect samples
        self.samples = []
        self._collect_samples()
        
        self.n = len(self.samples)
        self.num_plants = len(plant_to_idx)
        self.num_diseases = len(disease_to_idx)
        
        self.indexes = np.arange(self.n)
        if self.shuffle:
            np.random.shuffle(self.indexes)
        
        logger.info(f"Generator initialized: {self.n} samples, {self.num_plants} plants, {self.num_diseases} diseases")
    
    def _collect_samples(self):
        """Collect image samples from directory"""
        logger.info(f"Collecting samples from: {self.directory}")
        
        if not self.directory.exists():
            logger.error(f"Directory does not exist: {self.directory}")
            return
        
        # Get available classes
        available_plants = set(self.plant_to_idx.keys())
        available_diseases = set(self.disease_to_idx.keys())
        
        logger.info(f"Looking for plants: {available_plants}")
        logger.info(f"Looking for diseases: {available_diseases}")
        
        for class_dir in self.directory.iterdir():
            if not class_dir.is_dir():
                continue
            
            # Parse class name: "Plant___Disease" or "Plant__variety___Disease"
            class_name = class_dir.name
            
            if '___' not in class_name:
                logger.warning(f"Skipping malformed class (no ___): {class_name}")
                continue
            
            parts = class_name.split('___')
            if len(parts) != 2:
                logger.warning(f"Skipping malformed class (wrong format): {class_name}")
                continue
            
            # Extract plant and disease
            plant_raw = parts[0]
            disease_raw = parts[1]
            
            # Remove variety suffix (e.g., "Tomato__Cherry" -> "Tomato")
            plant = plant_raw.split('__')[0] if '__' in plant_raw else plant_raw
            disease = disease_raw
            
            logger.info(f"Found class dir: {class_name} -> Plant: {plant}, Disease: {disease}")
            
            # Check if in mappings
            if plant not in self.plant_to_idx:
                logger.warning(f"Plant '{plant}' not in model classes. Available: {available_plants}")
                continue
            
            if disease not in self.disease_to_idx:
                logger.warning(f"Disease '{disease}' not in model classes. Available: {available_diseases}")
                continue
            
            # Collect images (handle both lowercase and uppercase extensions)
            image_count = 0
            for img_path in class_dir.glob('*'):
                ext = img_path.suffix.lower()
                if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.samples.append({
                        'path': img_path,
                        'plant': plant,
                        'disease': disease
                    })
                    image_count += 1
            
            if image_count > 0:
                logger.info(f"  Added {image_count} images from {class_name}")
            else:
                logger.warning(f"  No valid images found in {class_name}")
        
        logger.info(f"TOTAL COLLECTED: {len(self.samples)} samples")
    
    def __len__(self):
        """Number of batches per epoch"""
        if self.n == 0:
            return 0
        return int(np.ceil(self.n / self.batch_size))
    
    def __getitem__(self, index):
        """Generate one batch of data"""
        if self.n == 0:
            # Return empty arrays with correct shape
            return (
                np.zeros((0, *self.img_size, 3), dtype=np.float32),
                {
                    'plant_output': np.zeros((0, self.num_plants), dtype=np.float32),
                    'disease_output': np.zeros((0, self.num_diseases), dtype=np.float32)
                }
            )
        
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        batch_images = []
        batch_plant_labels = []
        batch_disease_labels = []
        
        for idx in batch_indexes:
            sample = self.samples[idx]
            
            # Load and preprocess image
            try:
                img = Image.open(sample['path'])
                img = img.convert('RGB')
                img = img.resize(self.img_size)
                img_array = np.array(img).astype(np.float32)
                
                # Apply augmentation
                img_array = self.datagen.random_transform(img_array)
                img_array = self.datagen.standardize(img_array)
                
                batch_images.append(img_array)
                
                # Create one-hot encoded labels
                plant_label = np.zeros(self.num_plants)
                plant_label[self.plant_to_idx[sample['plant']]] = 1
                batch_plant_labels.append(plant_label)
                
                disease_label = np.zeros(self.num_diseases)
                disease_label[self.disease_to_idx[sample['disease']]] = 1
                batch_disease_labels.append(disease_label)
                
            except Exception as e:
                logger.error(f"Error loading {sample['path']}: {str(e)}")
                continue
        
        # Ensure we have data
        if len(batch_images) == 0:
            logger.error(f"Batch {index} has 0 valid images!")
            return (
                np.zeros((0, *self.img_size, 3), dtype=np.float32),
                {
                    'plant_output': np.zeros((0, self.num_plants), dtype=np.float32),
                    'disease_output': np.zeros((0, self.num_diseases), dtype=np.float32)
                }
            )
        
        return (
            np.array(batch_images),
            {
                'plant_output': np.array(batch_plant_labels),
                'disease_output': np.array(batch_disease_labels)
            }
        )
    
    def on_epoch_end(self):
        """Shuffle indexes after each epoch"""
        if self.shuffle and self.n > 0:
            np.random.shuffle(self.indexes)


class RetrainingPipeline:
    """Manages multi-output model retraining with new data"""

    def __init__(self):
        self.detector = MultiOutputPlantDiseaseDetector()
        self.training_history = {}
        self.model_versions = self._load_model_versions()
        self.plant_to_idx = {}
        self.disease_to_idx = {}

    def _load_model_versions(self) -> Dict:
        """Load model version history"""
        versions_file = MODELS_DIR / "versions.json"
        if versions_file.exists():
            with open(versions_file, "r") as f:
                return json.load(f)
        return {"versions": [], "current_version": "v1.0.0"}

    def _save_model_versions(self):
        """Save model version history"""
        versions_file = MODELS_DIR / "versions.json"
        with open(versions_file, "w") as f:
            json.dump(self.model_versions, f, indent=2)

    def _backup_current_model(self) -> Optional[Path]:
        """
        Backup current model before retraining
        
        Returns:
            Path to backup directory
        """
        if not MULTI_OUTPUT_MODEL.exists():
            logger.warning("No existing model to backup")
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = MODELS_DIR / f"backup_{timestamp}"
        backup_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Backup model
            if MULTI_OUTPUT_MODEL.exists():
                backup_model_dir = backup_dir / MULTI_OUTPUT_MODEL.name
                shutil.copytree(str(MULTI_OUTPUT_MODEL), str(backup_model_dir))
                logger.info(f"Model backed up to {backup_model_dir}")

            # Backup mappings
            if CLASS_MAPPINGS_FILE.exists():
                shutil.copy2(CLASS_MAPPINGS_FILE, backup_dir / CLASS_MAPPINGS_FILE.name)

            # Backup metadata
            if MODEL_METADATA_FILE.exists():
                shutil.copy2(MODEL_METADATA_FILE, backup_dir / MODEL_METADATA_FILE.name)

            logger.info(f"Backup created at {backup_dir}")
            return backup_dir

        except Exception as e:
            logger.error(f"Backup failed: {str(e)}")
            return None

    def check_new_data(self) -> Dict:
        """
        Check if new training data is available
        
        Returns:
            Dictionary with data statistics (returns total files before train/val split)
        """
        logger.info(f"Checking for new data in: {RETRAIN_DATA_DIR}")
        
        # Count images in PlantVillage subdirectory
        plant_village_dir = RETRAIN_DATA_DIR / "PlantVillage"
        
        if not plant_village_dir.exists():
            logger.warning(f"PlantVillage directory not found: {plant_village_dir}")
            return {
                "total_samples": 0,
                "ready_to_retrain": False,
                "min_required": RETRAINING_CONFIG.get("min_samples", 50),
                "data_dir": str(RETRAIN_DATA_DIR),
                "message": "PlantVillage directory not found. Upload some images first."
            }
        
        # Load class mappings to validate
        self.load_class_mappings()
        
        # Count all images in class subdirectories
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        total_images = 0
        valid_images = 0
        class_counts = {}
        invalid_classes = []
        
        for class_dir in plant_village_dir.iterdir():
            if class_dir.is_dir():
                count = 0
                for ext in image_extensions:
                    count += len(list(class_dir.glob(f'*{ext}')))
                    count += len(list(class_dir.glob(f'*{ext.upper()}')))
                
                if count > 0:
                    total_images += count
                    
                    # Check if class is valid
                    class_name = class_dir.name
                    if '___' in class_name:
                        parts = class_name.split('___')
                        plant = parts[0].split('__')[0]
                        disease = parts[1]
                        
                        if plant in self.plant_to_idx and disease in self.disease_to_idx:
                            class_counts[class_name] = count
                            valid_images += count
                        else:
                            invalid_classes.append({
                                'class': class_name,
                                'count': count,
                                'plant_valid': plant in self.plant_to_idx,
                                'disease_valid': disease in self.disease_to_idx
                            })
        
        logger.info(f"Found {total_images} total images, {valid_images} valid for model classes")
        
        if invalid_classes:
            logger.warning(f"Found {len(invalid_classes)} invalid class directories:")
            for inv in invalid_classes:
                logger.warning(f"  - {inv['class']}: {inv['count']} images (Plant valid: {inv['plant_valid']}, Disease valid: {inv['disease_valid']})")
        
        for class_name, count in class_counts.items():
            logger.info(f"  Valid: {class_name}: {count} images")

        # Check if ready for retraining (use valid images only)
        min_samples = RETRAINING_CONFIG.get("min_samples", 50)
        ready = valid_images >= min_samples

        message = f"Found {valid_images} valid samples across {len(class_counts)} classes"
        if invalid_classes:
            message += f" ({total_images - valid_images} images in invalid classes)"
        message += f". {'Ready' if ready else f'Need {min_samples - valid_images} more valid samples'} for retraining."

        stats = {
            "total_samples": valid_images,  # Return total valid files (before split)
            "total_files": total_images,
            "ready_to_retrain": ready,
            "min_required": min_samples,
            "data_dir": str(RETRAIN_DATA_DIR),
            "class_counts": class_counts,
            "invalid_classes": invalid_classes,
            "message": message
        }

        return stats
    
    def clear_retrain_data(self):
        """Clear all data in the retrain directory after successful retraining"""
        plant_village_dir = RETRAIN_DATA_DIR / "PlantVillage"
        
        if plant_village_dir.exists():
            try:
                shutil.rmtree(plant_village_dir)
                plant_village_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Cleared retrain data directory")
                return True
            except Exception as e:
                logger.error(f"Failed to clear retrain data: {str(e)}")
                return False
        return True

    def load_class_mappings(self) -> bool:
        """
        Load existing class mappings from trained model
        
        Returns:
            True if successful
        """
        if CLASS_MAPPINGS_FILE.exists():
            with open(CLASS_MAPPINGS_FILE, 'r') as f:
                mappings = json.load(f)
                self.plant_to_idx = mappings['plant_to_idx']
                self.disease_to_idx = mappings['disease_to_idx']
            logger.info(f"Loaded class mappings: {len(self.plant_to_idx)} plants, {len(self.disease_to_idx)} diseases")
            logger.info(f"Valid plants: {list(self.plant_to_idx.keys())}")
            logger.info(f"Valid diseases: {list(self.disease_to_idx.keys())}")
            return True
        else:
            logger.error(f"Class mappings not found at {CLASS_MAPPINGS_FILE}")
            return False

    def prepare_data_generators(self, val_split: float = 0.15) -> Tuple[Optional[MultiLabelDataGenerator], Optional[MultiLabelDataGenerator]]:
        """
        Prepare training and validation data generators
        
        Args:
            val_split: Fraction for validation set
            
        Returns:
            Tuple of (train_generator, val_generator)
        """
        logger.info(f"Preparing data generators from {RETRAIN_DATA_DIR}")

        # Load class mappings
        if not self.load_class_mappings():
            logger.error("Cannot prepare generators without class mappings")
            return None, None

        # Check for PlantVillage subdirectory
        plant_village_dir = RETRAIN_DATA_DIR / "PlantVillage"
        if not plant_village_dir.exists():
            logger.error(f"PlantVillage directory not found in {RETRAIN_DATA_DIR}")
            return None, None

        # Create ImageDataGenerator for augmentation
        train_datagen = ImageDataGenerator(
            rescale=PREPROCESSING_CONFIG['rescale'],
            rotation_range=PREPROCESSING_CONFIG['rotation_range'],
            width_shift_range=PREPROCESSING_CONFIG['width_shift_range'],
            height_shift_range=PREPROCESSING_CONFIG['height_shift_range'],
            shear_range=PREPROCESSING_CONFIG['shear_range'],
            zoom_range=PREPROCESSING_CONFIG['zoom_range'],
            horizontal_flip=PREPROCESSING_CONFIG['horizontal_flip'],
            vertical_flip=PREPROCESSING_CONFIG['vertical_flip'],
            fill_mode=PREPROCESSING_CONFIG['fill_mode'],
            brightness_range=PREPROCESSING_CONFIG.get('brightness_range'),
        )

        val_datagen = ImageDataGenerator(rescale=PREPROCESSING_CONFIG['rescale'])

        # Create temporary train/val split directories
        train_dir = RETRAIN_DATA_DIR / "temp_train" / "PlantVillage"
        val_dir = RETRAIN_DATA_DIR / "temp_val" / "PlantVillage"
        
        # Clear if exists
        if train_dir.parent.exists():
            shutil.rmtree(train_dir.parent)
        if val_dir.parent.exists():
            shutil.rmtree(val_dir.parent)
        
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        # Split data - only valid classes
        total_samples = 0
        valid_classes = 0
        
        for class_dir in plant_village_dir.iterdir():
            if not class_dir.is_dir():
                continue
            
            # Validate class
            class_name = class_dir.name
            if '___' not in class_name:
                logger.warning(f"Skipping {class_name}: invalid format")
                continue
            
            parts = class_name.split('___')
            plant = parts[0].split('__')[0]
            disease = parts[1]
            
            # Skip if not in model classes
            if plant not in self.plant_to_idx or disease not in self.disease_to_idx:
                logger.warning(f"Skipping {class_name}: not in model classes")
                continue
            
            # Get all images
            images = [f for f in class_dir.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
            
            if len(images) == 0:
                logger.warning(f"No images found in {class_dir.name}")
                continue
            
            logger.info(f"Processing {len(images)} images from {class_dir.name}")
            
            # Shuffle
            np.random.shuffle(images)
            
            # Split
            split_idx = max(1, int(len(images) * (1 - val_split)))  # At least 1 train image
            train_images = images[:split_idx]
            val_images = images[split_idx:] if split_idx < len(images) else images[-1:]
            
            # Create class directories
            train_class_dir = train_dir / class_dir.name
            val_class_dir = val_dir / class_dir.name
            train_class_dir.mkdir(exist_ok=True)
            val_class_dir.mkdir(exist_ok=True)
            
            # Copy files
            for img in train_images:
                shutil.copy2(img, train_class_dir / img.name)
            for img in val_images:
                shutil.copy2(img, val_class_dir / img.name)
            
            total_samples += len(images)
            valid_classes += 1
            logger.info(f"  Split: {len(train_images)} train, {len(val_images)} val")

        if total_samples == 0:
            logger.error("No valid samples found for training!")
            return None, None

        logger.info(f"Total split: {total_samples} samples from {valid_classes} valid classes")

        # Create generators
        try:
            train_gen = MultiLabelDataGenerator(
                directory=train_dir,
                datagen=train_datagen,
                plant_to_idx=self.plant_to_idx,
                disease_to_idx=self.disease_to_idx,
                img_size=(MODEL_CONFIG['input_shape'][0], MODEL_CONFIG['input_shape'][1]),
                batch_size=MODEL_CONFIG['batch_size'],
                shuffle=True
            )

            val_gen = MultiLabelDataGenerator(
                directory=val_dir,
                datagen=val_datagen,
                plant_to_idx=self.plant_to_idx,
                disease_to_idx=self.disease_to_idx,
                img_size=(MODEL_CONFIG['input_shape'][0], MODEL_CONFIG['input_shape'][1]),
                batch_size=MODEL_CONFIG['batch_size'],
                shuffle=False
            )

            logger.info(f"Train generator: {len(train_gen)} batches, {train_gen.n} samples")
            logger.info(f"Val generator: {len(val_gen)} batches, {val_gen.n} samples")
            
            if train_gen.n == 0:
                logger.error("Training generator has 0 samples!")
                return None, None
            
            return train_gen, val_gen

        except Exception as e:
            logger.error(f"Error creating generators: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None, None

    def retrain_model(
        self,
        epochs: int = 20,
        fine_tune: bool = True
    ) -> Dict:
        """
        Retrain the multi-output model with new data
        
        Args:
            epochs: Number of epochs
            fine_tune: Whether to unfreeze and fine-tune base layers
            
        Returns:
            Training results and metrics
        """
        logger.info("=" * 70)
        logger.info("STARTING MODEL RETRAINING")
        logger.info("=" * 70)

        # Get total samples BEFORE train/val split
        data_check = self.check_new_data()
        total_samples_uploaded = data_check.get("total_samples", 0)
        
        # Backup current model
        if RETRAINING_CONFIG.get("backup_old_models", True):
            backup_dir = self._backup_current_model()

        # Prepare data
        train_gen, val_gen = self.prepare_data_generators()
        if train_gen is None or val_gen is None:
            error_msg = "Failed to prepare data generators - no valid training data found"
            logger.error(error_msg)
            return {
                "status": "failed",
                "error": error_msg,
                "samples_used": 0
            }
        
        if train_gen.n == 0:
            error_msg = "Training generator has 0 samples - check your uploaded images match model classes"
            logger.error(error_msg)
            return {
                "status": "failed",
                "error": error_msg,
                "samples_used": 0
            }

        # Load existing model or build new one
        if not self.detector.load_model():
            logger.warning("No existing model found, building new one")
            self.detector.num_plant_classes = len(self.plant_to_idx)
            self.detector.num_disease_classes = len(self.disease_to_idx)
            self.detector.build_model()

        # Set class mappings
        self.detector.set_class_mappings(self.plant_to_idx, self.disease_to_idx)

        # Fine-tuning: unfreeze some layers
        if fine_tune and self.detector.model:
            logger.info("Unfreezing layers for fine-tuning")
            # Unfreeze last 20 layers
            for layer in self.detector.model.layers[-20:]:
                if not isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = True

            # Recompile with lower learning rate
            self.detector.model.compile(
                optimizer=Adam(learning_rate=MODEL_CONFIG["learning_rate"] / 10),
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
            logger.info("Model recompiled for fine-tuning")

        # Setup callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                verbose=1,
                min_lr=1e-7
            ),
            ModelCheckpoint(
                str(MULTI_OUTPUT_MODEL),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]

        # Train
        try:
            logger.info(f"Starting training: {epochs} epochs with {total_samples_uploaded} total samples ({train_gen.n} train, {val_gen.n} val)")
            history = self.detector.model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )

            self.training_history = history.history
            logger.info("Retraining completed successfully")

            # Clean up temp directories
            if (RETRAIN_DATA_DIR / "temp_train").exists():
                shutil.rmtree(RETRAIN_DATA_DIR / "temp_train")
            if (RETRAIN_DATA_DIR / "temp_val").exists():
                shutil.rmtree(RETRAIN_DATA_DIR / "temp_val")

            # Compile results - use total uploaded samples
            results = {
                "status": "success",
                "epochs_trained": len(history.history['loss']),
                "samples_used": total_samples_uploaded,  # Total uploaded (before split)
                "train_samples": train_gen.n,  # Actual training samples
                "val_samples": val_gen.n,  # Actual validation samples
                "final_metrics": {
                    "plant_train_acc": float(history.history['plant_output_accuracy'][-1]),
                    "plant_val_acc": float(history.history['val_plant_output_accuracy'][-1]),
                    "disease_train_acc": float(history.history['disease_output_accuracy'][-1]),
                    "disease_val_acc": float(history.history['val_disease_output_accuracy'][-1]),
                },
                "timestamp": datetime.now().isoformat()
            }

            logger.info(f"Final metrics: {results['final_metrics']}")
            logger.info("=" * 70)
            logger.info("RETRAINING COMPLETED SUCCESSFULLY")
            logger.info("=" * 70)
            return results

        except Exception as e:
            logger.error(f"Retraining failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "status": "failed",
                "error": str(e),
                "samples_used": total_samples_uploaded
            }

    def save_retrained_model(self, version: Optional[str] = None) -> str:
        """
        Save retrained model with version info
        
        Args:
            version: Version string (auto-generated if None)
            
        Returns:
            New version string
        """
        if version is None:
            # Auto-increment version
            current = self.model_versions.get("current_version", "v1.0.0")
            major, minor, patch = map(int, current.replace('v', '').split('.'))
            version = f"v{major}.{minor + 1}.0"

        logger.info(f"Saving retrained model as {version}")

        # Save model
        self.detector.save_model(save_mappings=True)

        # Update version history
        version_info = {
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "metrics": self.training_history,
        }
        self.model_versions["versions"].append(version_info)
        self.model_versions["current_version"] = version
        self._save_model_versions()

        logger.info(f"Model saved as {version}")
        return version

    def trigger_retrain(self, min_samples: Optional[int] = None) -> Dict:
        """
        Check if retraining should be triggered
        
        Args:
            min_samples: Minimum samples required (uses config default if None)
            
        Returns:
            Status dictionary
        """
        if min_samples is None:
            min_samples = RETRAINING_CONFIG.get("min_samples", 50)

        # Check data availability
        data_check = self.check_new_data()

        status = {
            "triggered": False,
            "reason": "",
            "timestamp": datetime.now().isoformat(),
            "new_samples": data_check["total_samples"]
        }

        if data_check["total_samples"] < min_samples:
            status["reason"] = f"Insufficient data: {data_check['total_samples']} < {min_samples}"
            logger.info(f"{status['reason']}")
            return status

        status["triggered"] = True
        status["reason"] = f"Retraining conditions met: {data_check['total_samples']} samples available"
        logger.info(f"{status['reason']}")

        return status


if __name__ == "__main__":
    # Test the retraining pipeline
    pipeline = RetrainingPipeline()
    
    print("Checking for new training data...")
    data_check = pipeline.check_new_data()
    print(f"\nData Status:")
    print(f"  - Total samples: {data_check['total_samples']}")
    print(f"  - Ready to retrain: {data_check['ready_to_retrain']}")
    print(f"  - Message: {data_check['message']}")
    
    if data_check.get('invalid_classes'):
        print(f"\nInvalid classes found:")
        for inv in data_check['invalid_classes']:
            print(f"  - {inv}")