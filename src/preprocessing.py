"""
Data Preprocessing module for Plant Disease Detection
Handles image loading, augmentation, and normalization
"""

import numpy as np
import cv2
from pathlib import Path
import logging
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from config import (
    PREPROCESSING_CONFIG,
    MODEL_CONFIG,
    DATA_DIR,
)

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles all data preprocessing operations"""

    def __init__(self, input_shape=MODEL_CONFIG["input_shape"]):
        self.input_shape = input_shape
        self.label_encoder_plant = LabelEncoder()
        self.label_encoder_disease = LabelEncoder()
        self.augmenter = ImageDataGenerator(**PREPROCESSING_CONFIG)

    def load_image(self, image_path, target_size=None):
        """
        Load and preprocess a single image
        
        Args:
            image_path: Path to the image file
            target_size: Target size (height, width)
            
        Returns:
            Preprocessed image as numpy array
        """
        if target_size is None:
            target_size = self.input_shape[:2]

        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning(f"Could not load image: {image_path}")
                return None

            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Resize
            image = cv2.resize(image, (target_size[1], target_size[0]))

            # Normalize
            image = image / 255.0

            return image

        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            return None

    def load_images_from_directory(self, directory, recursive=True):
        """
        Load all images from a directory
        
        Args:
            directory: Path to directory containing images
            recursive: Whether to search subdirectories
            
        Returns:
            List of loaded images
        """
        images = []
        directory = Path(directory)

        # Valid image extensions
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

        # Find all image files
        if recursive:
            image_files = [
                f for f in directory.rglob("*")
                if f.is_file() and f.suffix.lower() in valid_extensions
            ]
        else:
            image_files = [
                f for f in directory.iterdir()
                if f.is_file() and f.suffix.lower() in valid_extensions
            ]

        logger.info(f"Found {len(image_files)} images in {directory}")

        for image_file in image_files:
            image = self.load_image(image_file)
            if image is not None:
                images.append((image, image_file))

        return images

    def augment_image(self, image, num_augmentations=5):
        """
        Generate augmented versions of an image
        
        Args:
            image: Input image as numpy array
            num_augmentations: Number of augmented versions to create
            
        Returns:
            List of augmented images
        """
        augmented_images = []
        
        # Reshape for augmentation
        image = np.expand_dims(image, axis=0)

        # Generate augmented images
        for _ in range(num_augmentations):
            aug_image = self.augmenter.random_transform(image[0])
            augmented_images.append(aug_image)

        return augmented_images

    def extract_color_features(self, image):
        """
        Extract color distribution features from image
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary with color statistics
        """
        # Convert to HSV for better color analysis
        image_hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)

        features = {
            "hue_mean": np.mean(image_hsv[:, :, 0]),
            "hue_std": np.std(image_hsv[:, :, 0]),
            "saturation_mean": np.mean(image_hsv[:, :, 1]),
            "saturation_std": np.std(image_hsv[:, :, 1]),
            "value_mean": np.mean(image_hsv[:, :, 2]),
            "value_std": np.std(image_hsv[:, :, 2]),
        }

        return features

    def extract_texture_features(self, image):
        """
        Extract texture features using edge detection
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary with texture statistics
        """
        # Convert to grayscale
        image_gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

        # Apply Sobel edge detection
        edges_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
        edges_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)
        edges_magnitude = np.sqrt(edges_x**2 + edges_y**2)

        features = {
            "edge_density": np.sum(edges_magnitude > 50) / edges_magnitude.size,
            "edge_mean": np.mean(edges_magnitude),
            "edge_std": np.std(edges_magnitude),
            "gradient_magnitude_mean": np.mean(np.sqrt(edges_x**2 + edges_y**2)),
        }

        return features

    def normalize_image(self, image):
        """
        Normalize image to [0, 1] range
        
        Args:
            image: Input image
            
        Returns:
            Normalized image
        """
        image = np.asarray(image, dtype=np.float32)
        if np.max(image) > 1.0:
            image = image / 255.0
        return image

    def preprocess_batch(self, image_list):
        """
        Preprocess a batch of images
        
        Args:
            image_list: List of image paths or arrays
            
        Returns:
            Batch of preprocessed images
        """
        batch = []

        for item in image_list:
            if isinstance(item, (str, Path)):
                image = self.load_image(item)
            else:
                image = item

            if image is not None:
                image = self.normalize_image(image)
                batch.append(image)

        return np.array(batch)


def create_data_generators():
    """
    Create training and validation data generators
    
    Returns:
        Tuple of (train_generator, validation_generator)
    """
    train_datagen = ImageDataGenerator(**PREPROCESSING_CONFIG)

    # Validation data should only be rescaled, not augmented
    val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    return train_datagen, val_datagen


def split_data_70_15_15(X, y_plant, y_disease, random_state=42):
    """
    Split dataset into Train (70%) / Validation (15%) / Test (15%)
    
    Args:
        X: Input features (images)
        y_plant: Plant labels
        y_disease: Disease labels
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train_plant, y_val_plant, y_test_plant,
                  y_train_disease, y_val_disease, y_test_disease)
    """
    # Step 1: Split into train (70%) and temp (30%)
    X_train, X_temp, y_train_plant, y_temp_plant, y_train_disease, y_temp_disease = train_test_split(
        X, y_plant, y_disease,
        test_size=0.30,
        random_state=random_state,
        stratify=y_plant
    )
    
    # Step 2: Split temp into validation (50% of temp = 15% total) and test (50% of temp = 15% total)
    X_val, X_test, y_val_plant, y_test_plant, y_val_disease, y_test_disease = train_test_split(
        X_temp, y_temp_plant, y_temp_disease,
        test_size=0.50,
        random_state=random_state,
        stratify=y_temp_plant
    )
    
    return (
        X_train, X_val, X_test,
        y_train_plant, y_val_plant, y_test_plant,
        y_train_disease, y_val_disease, y_test_disease
    )



if __name__ == "__main__":
    # Test preprocessing
    preprocessor = DataPreprocessor()
    print("DataPreprocessor initialized successfully")
    print(f"Input shape: {preprocessor.input_shape}")
