# Plant Disease Detection Model Export

## Files Included:
- `plant_disease_model.keras` - Trained TensorFlow/Keras model
- `class_mappings.json` - Class indices for plant and disease labels
- `model_metadata.json` - Model configuration and performance metrics
- `training_history.json` - Training history for visualization

## Model Details:
- Architecture: MobileNetV2 with dual output heads
- Input: 224x224x3 RGB images
- Outputs: 
  - Plant classification (3 classes)
  - Disease classification (15 classes)

## Usage:
Place these files in your `models/` directory and load using TensorFlow/Keras.
