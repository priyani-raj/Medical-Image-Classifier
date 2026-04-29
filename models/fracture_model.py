"""
Bone Fracture Model (Model 3)
------------------------------
Dataset  : Kaggle Bone Fracture Detection
           kaggle.com/datasets/pkdarabi/bone-fracture-detection-computer-vision-project
Classes  : Normal | Fractured
Model    : ResNet50 (pretrained ImageNet, fine-tuned)
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

# ─── CONFIG ──────────────────────────────────────────────────────────────────
IMG_SIZE   = (224, 224)
MODEL_PATH = "results/fracture_model.h5"
CLASSES    = ["Normal", "Fractured"]


# ─── MODEL ───────────────────────────────────────────────────────────────────
def build_fracture_model():
    """
    ResNet50-based binary classifier for fracture detection.
    ResNet50 handles edge/structural features well — good for bone X-rays.
    """
    base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base.trainable = False

    model = Sequential([
        base,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dense(256, activation='relu'),
        Dropout(0.4),
        Dense(1, activation='sigmoid')   # binary output
    ])

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss      = 'binary_crossentropy',
        metrics   = ['accuracy']
    )
    return model


def unfreeze_fracture(model, unfreeze_last_n=10):
    """Fine-tune last N layers of ResNet50 with a low LR."""
    base = model.layers[0]
    for layer in base.layers:
        layer.trainable = False
    for layer in base.layers[-unfreeze_last_n:]:
        layer.trainable = True

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss      = 'binary_crossentropy',
        metrics   = ['accuracy']
    )
    return model


# ─── PREPROCESSING ───────────────────────────────────────────────────────────
def preprocess_image(img_path):
    """Load and preprocess a bone X-ray image for inference."""
    import cv2
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.resize(img, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    arr = img.astype(np.float32)
    arr = preprocess_input(arr)           # ResNet50 preprocessing
    return np.expand_dims(arr, axis=0)


# ─── PREDICTION ──────────────────────────────────────────────────────────────
def predict_fracture(img_path: str) -> dict:
    """Predict fracture from a bone X-ray image."""
    if not os.path.exists(MODEL_PATH):
        return {
            "label"      : "Model not trained yet",
            "confidence" : 0,
            "model_used" : "FractureCNN_ResNet50",
            "classes"    : CLASSES,
            "error"      : f"Train first. Expected at: {MODEL_PATH}"
        }

    model = load_model(MODEL_PATH)
    img   = preprocess_image(img_path)
    prob  = float(model.predict(img, verbose=0)[0][0])
    label = CLASSES[1] if prob > 0.5 else CLASSES[0]
    confidence = round((prob if prob > 0.5 else 1 - prob) * 100, 2)

    return {
        "label"      : label,
        "confidence" : confidence,
        "model_used" : "FractureCNN_ResNet50",
        "classes"    : CLASSES
    }