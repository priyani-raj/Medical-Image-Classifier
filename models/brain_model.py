"""
Brain MRI Tumor Classification Model
=====================================
Dataset  : Kaggle Brain Tumor MRI Dataset
           kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
Classes  : No Tumor | Glioma | Meningioma | Pituitary
Author   : Medical Imaging Project
Status   : Production Ready
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Flatten, Dropout, 
    GlobalAveragePooling2D, BatchNormalization, Input
)
from tensorflow.keras.applications import ResNet50, EfficientNetB0, VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.optimizers import Adam

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

class BrainModelConfig:
    """Configuration for brain tumor model"""
    IMG_SIZE = (128, 128)
    IMG_CHANNELS = 3
    MODEL_PATH = "results/brain_model.h5"
    CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
    NUM_CLASSES = 4
    WEIGHTS_PATH = "results/brain_weights.h5"
    
    # Training config
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    MODEL_TYPE = "ResNet50"  # ResNet50, EfficientNetB0, VGG16, CustomCNN


# ─────────────────────────────────────────────
# MODEL ARCHITECTURE
# ─────────────────────────────────────────────

class BrainTumorModel:
    """Brain Tumor Classification Model"""
    
    def __init__(self, config=None):
        self.config = config or BrainModelConfig()
        self.model = None
        self.is_trained = False
    
    def build_transfer_learning_model(self, base_model_name="ResNet50"):
        """
        Build transfer learning model for brain tumor classification.
        
        Args:
            base_model_name: Type of base model (ResNet50, EfficientNetB0, VGG16)
        """
        print(f"🔨 Building {base_model_name} Transfer Learning Model...")
        
        # Load pretrained base model
        if base_model_name == "ResNet50":
            base_model = ResNet50(
                input_shape=(self.config.IMG_SIZE[0], self.config.IMG_SIZE[1], self.config.IMG_CHANNELS),
                include_top=False,
                weights='imagenet'
            )
        elif base_model_name == "EfficientNetB0":
            base_model = EfficientNetB0(
                input_shape=(self.config.IMG_SIZE[0], self.config.IMG_SIZE[1], self.config.IMG_CHANNELS),
                include_top=False,
                weights='imagenet'
            )
        elif base_model_name == "VGG16":
            base_model = VGG16(
                input_shape=(self.config.IMG_SIZE[0], self.config.IMG_SIZE[1], self.config.IMG_CHANNELS),
                include_top=False,
                weights='imagenet'
            )
        else:
            raise ValueError(f"Unknown base model: {base_model_name}")
        
        # Freeze base model weights
        base_model.trainable = False
        
        # Build custom top
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(self.config.NUM_CLASSES, activation='softmax')
        ])
        
        return model
    
    def build_custom_cnn(self):
        """Build custom CNN from scratch"""
        print("🔨 Building Custom CNN Model...")
        
        model = Sequential([
            # Block 1
            Conv2D(32, (3, 3), activation='relu', 
                   input_shape=(self.config.IMG_SIZE[0], self.config.IMG_SIZE[1], self.config.IMG_CHANNELS)),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Block 2
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Block 3
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Dense layers
            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(self.config.NUM_CLASSES, activation='softmax')
        ])
        
        return model
    
    def build_model(self):
        """Build model based on configuration"""
        if self.config.MODEL_TYPE in ["ResNet50", "EfficientNetB0", "VGG16"]:
            self.model = self.build_transfer_learning_model(self.config.MODEL_TYPE)
        else:
            self.model = self.build_custom_cnn()
        
        # Compile
        self.model.compile(
            optimizer=Adam(learning_rate=self.config.LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ Model compiled successfully!")
        return self.model
    
    def load_weights(self, weights_path=None):
        """Load pretrained weights"""
        path = weights_path or self.config.WEIGHTS_PATH
        
        if Path(path).exists():
            print(f"📦 Loading weights from {path}...")
            if self.model is None:
                self.build_model()
            self.model.load_weights(path)
            self.is_trained = True
            print("✅ Weights loaded successfully!")
            return True
        else:
            print(f"⚠️  Weights file not found: {path}")
            return False
    
    def save_model(self, save_path=None):
        """Save trained model"""
        path = save_path or self.config.MODEL_PATH
        
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        print(f"✅ Model saved to {path}")
        
        # Also save weights
        weights_path = self.config.WEIGHTS_PATH
        Path(weights_path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_weights(weights_path)
        print(f"✅ Weights saved to {weights_path}")


# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────

def preprocess_image(img_path: str, img_size=(224, 224)) -> np.ndarray:
    """
    Load and preprocess a brain MRI image.
    
    Args:
        img_path: Path to MRI image
        img_size: Target size (default: 224x224)
    
    Returns:
        Preprocessed image array with batch dimension
    """
    try:
        # Load image
        img = load_img(img_path, target_size=img_size, color_mode='rgb')
        
        # Convert to array and normalize
        img_array = img_to_array(img) / 255.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img_array, axis=0)
        
        return img_batch
    
    except Exception as e:
        print(f"❌ Error preprocessing image {img_path}: {e}")
        return None


def preprocess_batch(image_paths: List[str], img_size=(224, 224)) -> np.ndarray:
    """
    Preprocess a batch of images.
    
    Args:
        image_paths: List of image paths
        img_size: Target size
    
    Returns:
        Batch of preprocessed images
    """
    images = []
    for img_path in image_paths:
        img = preprocess_image(img_path, img_size)
        if img is not None:
            images.append(img[0])  # Remove batch dim, will add later
    
    if not images:
        return None
    
    return np.array(images)


# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────

def predict_brain(img_path: str, model_path=None) -> Dict:
    """
    Predict brain MRI class and confidence.
    
    Args:
        img_path: Path to MRI image
        model_path: Path to trained model (optional)
    
    Returns:
        Dictionary with prediction results
    """
    config = BrainModelConfig()
    
    if model_path:
        config.MODEL_PATH = model_path
    
    # Load model
    if not Path(config.MODEL_PATH).exists():
        print(f"❌ Model not found at {config.MODEL_PATH}")
        print("⚠️  Returning dummy prediction. Train the model first!")
        return {
            "label": "Unknown",
            "confidence": 0.0,
            "probabilities": {cls: 0.0 for cls in config.CLASSES},
            "model_used": "UNTRAINED",
            "error": f"Model not found at {config.MODEL_PATH}"
        }
    
    try:
        # ── Version-safe model loading ─────────────────────────────────────────
        # Newer TF adds 'groups' to DepthwiseConv2D; older TF raises
        # "Unrecognized keyword argument" when deserializing such a model.
        # Register a patched subclass that silently drops the unknown kwarg.
        from tensorflow.keras.layers import DepthwiseConv2D as _DWConv

        class _CompatDepthwiseConv2D(_DWConv):
            def __init__(self, *args, **kwargs):
                kwargs.pop("groups", None)
                super().__init__(*args, **kwargs)

        model = load_model(
            config.MODEL_PATH,
            custom_objects={"DepthwiseConv2D": _CompatDepthwiseConv2D},
        )
        img_batch = preprocess_image(img_path, config.IMG_SIZE)
        
        if img_batch is None:
            return {
                "label": "Error",
                "confidence": 0.0,
                "error": "Could not preprocess image"
            }
        
        # Make prediction
        predictions = model.predict(img_batch, verbose=0)
        probs = predictions[0]
        predicted_idx = np.argmax(probs)
        predicted_label = config.CLASSES[predicted_idx]
        confidence = float(probs[predicted_idx])
        
        # Prepare probability dict
        prob_dict = {
            config.CLASSES[i]: float(probs[i])
            for i in range(len(config.CLASSES))
        }
        
        return {
            "label": predicted_label,
            "confidence": round(confidence * 100, 2),
            "probabilities": prob_dict,
            "model_used": "MobileNetV2 · Brain MRI",
            "classes": config.CLASSES,
            "success": True
        }
    
    except Exception as e:
        return {
            "label": "Error",
            "confidence": 0.0,
            "error": str(e),
            "success": False
        }


def predict_batch(image_paths: List[str], model_path=None) -> List[Dict]:
    """
    Batch prediction on multiple images.
    
    Args:
        image_paths: List of image paths
        model_path: Path to trained model
    
    Returns:
        List of prediction dictionaries
    """
    config = BrainModelConfig()
    
    if model_path:
        config.MODEL_PATH = model_path
    
    if not Path(config.MODEL_PATH).exists():
        print(f"❌ Model not found at {config.MODEL_PATH}")
        return None
    
    try:
        from tensorflow.keras.layers import DepthwiseConv2D as _DWConv

        class _CompatDepthwiseConv2D(_DWConv):
            def __init__(self, *args, **kwargs):
                kwargs.pop("groups", None)
                super().__init__(*args, **kwargs)

        model = load_model(
            config.MODEL_PATH,
            custom_objects={"DepthwiseConv2D": _CompatDepthwiseConv2D},
        )
        images = preprocess_batch(image_paths, config.IMG_SIZE)
        
        if images is None:
            return None
        
        # Batch prediction
        predictions = model.predict(images, verbose=0)
        
        results = []
        for i, (img_path, probs) in enumerate(zip(image_paths, predictions)):
            predicted_idx = np.argmax(probs)
            predicted_label = config.CLASSES[predicted_idx]
            confidence = float(probs[predicted_idx])
            
            result = {
                "image": str(img_path),
                "label": predicted_label,
                "confidence": round(confidence * 100, 2),
                "probabilities": {
                    config.CLASSES[j]: float(probs[j])
                    for j in range(len(config.CLASSES))
                }
            }
            results.append(result)
        
        return results
    
    except Exception as e:
        print(f"❌ Error in batch prediction: {e}")
        return None


# ─────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────

def get_model_summary(model_type="ResNet50"):
    """Get model architecture summary"""
    config = BrainModelConfig()
    config.MODEL_TYPE = model_type
    
    model_builder = BrainTumorModel(config)
    model_builder.build_model()
    
    print(f"\n{'='*70}")
    print(f"MODEL: {model_type}")
    print(f"{'='*70}")
    model_builder.model.summary()
    print(f"{'='*70}\n")


if __name__ == "__main__":
    # Example usage
    config = BrainModelConfig()
    
    print("🧠 Brain Tumor Classification Model")
    print(f"Classes: {config.CLASSES}")
    print(f"Image Size: {config.IMG_SIZE}")
    
    # Build and show model architecture
    model = BrainTumorModel(config)
    model.build_model()
    model.model.summary()
    
    print("\n✅ Model ready for training!")
