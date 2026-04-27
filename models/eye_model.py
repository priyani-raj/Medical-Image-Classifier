"""
Eye / Retinal Model  —  Diabetic Retinopathy Grading
======================================================
Dataset  : Kaggle — Diabetic Retinopathy 2015 (Resized / Colored)
           kaggle.com/datasets/sovitrath/diabetic-retinopathy-2015-data-colored-resized

Classes  : 0 — No DR
           1 — Mild DR
           2 — Moderate DR
           3 — Severe DR
           4 — Proliferative DR

Model    : EfficientNetB3 (ImageNet pretrained, fine-tuned)
           Two-phase training:
             Phase 1 — frozen backbone, train head  (LR = 1e-3)
             Phase 2 — unfreeze last 30 layers       (LR = 1e-5)

KEY FIX  : Preprocessing is now IDENTICAL in training and inference.
           Both use EfficientNet's preprocess_input (scales to [-1, 1]).
           The old code used CLAHE + /255 for inference but preprocess_input
           during training — this distribution mismatch caused the model to
           collapse and predict Grade 0 for every image.

Author   : Ishita Arora
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
)
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input

# ─── CONFIG ──────────────────────────────────────────────────────────────────
IMG_SIZE   = (224, 224)
MODEL_PATH = "results/eye_model.h5"

CLASSES = [
    "No DR (Grade 0)",
    "Mild DR (Grade 1)",
    "Moderate DR (Grade 2)",
    "Severe DR (Grade 3)",
    "Proliferative DR (Grade 4)",
]

GRADE_INFO = {
    0: {
        "short":  "No DR",
        "advice": "No signs of diabetic retinopathy detected. Continue routine annual check-ups.",
    },
    1: {
        "short":  "Mild DR",
        "advice": "Microaneurysms present. Manage blood sugar carefully. Follow up in 12 months.",
    },
    2: {
        "short":  "Moderate DR",
        "advice": "Moderate vascular changes visible. Ophthalmology referral recommended within 6 months.",
    },
    3: {
        "short":  "Severe DR",
        "advice": "Significant retinal changes. Urgent ophthalmology referral required.",
    },
    4: {
        "short":  "Proliferative DR",
        "advice": "Advanced stage — new abnormal blood vessel growth. Immediate treatment needed.",
    },
}


# ─── MODEL DEFINITION ────────────────────────────────────────────────────────
def build_eye_model():
    """
    EfficientNetB3 backbone + lightweight classification head.
    BatchNormalization after pooling helps with the heavy class imbalance.
    Phase 1: backbone frozen, only the head is trained.
    """
    base = EfficientNetB3(
        weights     = "imagenet",
        include_top = False,
        input_shape = (224, 224, 3),
    )
    base.trainable = False  # freeze for Phase 1

    model = Sequential([
        base,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dense(256, activation="relu"),
        Dropout(0.4),
        Dense(5, activation="softmax"),
    ])

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss      = "categorical_crossentropy",
        metrics   = ["accuracy"],
    )
    return model


def unfreeze_for_finetuning(model, unfreeze_last_n=30):
    """
    Phase 2: unfreeze the last N EfficientNetB3 layers with a very low LR.
    Low LR is critical — a large LR would destroy the pretrained features.
    """
    base = model.layers[0]
    for layer in base.layers:
        layer.trainable = False
    for layer in base.layers[-unfreeze_last_n:]:
        layer.trainable = True

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss      = "categorical_crossentropy",
        metrics   = ["accuracy"],
    )
    return model


# ─── PREPROCESSING ───────────────────────────────────────────────────────────
def preprocess_image(img_path: str):
    """
    Single preprocessing function used for BOTH training and inference.

    Uses EfficientNet's built-in preprocess_input which scales pixel values
    from [0, 255] → [-1, 1].  This is the same function passed as
    `preprocessing_function` to ImageDataGenerator during training.

    *** Do NOT swap this for CLAHE + manual /255 normalization.
        That caused a training/inference distribution mismatch which
        made the model predict Grade 0 for every single image. ***
    """
    import cv2

    img_cv = cv2.imread(img_path)
    if img_cv is None:
        return None

    img_cv  = cv2.resize(img_cv, IMG_SIZE)
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    arr = img_rgb.astype(np.float32)
    arr = preprocess_input(arr)          # scales to [-1, 1]  (matches training)

    return np.expand_dims(arr, axis=0)  # → (1, 224, 224, 3)


# ─── CLASS WEIGHTS ───────────────────────────────────────────────────────────
def get_class_weights(train_labels=None):
    """
    Sqrt-dampened class weights to handle the heavy class imbalance
    without overwhelming the gradient signal.

    Formula: weight_i = sqrt(max_count / count_i), normalised so min = 1.

    Raw ratios (used in older code) caused the model to collapse onto the
    most-weighted class.  The sqrt dampens the extremes.
    """
    if train_labels is not None and len(train_labels) > 0:
        counts    = np.bincount(train_labels, minlength=5).astype(float)
        counts    = np.where(counts == 0, 1, counts)
        max_count = counts.max()
        weights   = np.sqrt(max_count / counts)
        weights   = weights / weights.min()           # normalise: min = 1.0
        return {i: round(float(w), 2) for i, w in enumerate(weights)}

    # Fallback based on approximate dataset distribution:
    # No DR ~73% | Mild ~7% | Moderate ~15% | Severe ~2.5% | Prolif ~2%
    return {0: 1.0, 1: 2.3, 2: 1.6, 3: 3.0, 4: 3.3}


# ─── PREDICTION ──────────────────────────────────────────────────────────────
def predict_eye(img_path: str) -> dict:
    """
    Run inference on a single retinal fundus image.

    Returns
    -------
    dict with keys:
        label       — human-readable DR class name
        grade       — integer 0-4
        confidence  — top-class probability as a percentage
        all_probs   — list[5] of probabilities (%) for all grades
        advice      — clinical guidance note for this grade
        model_used  — backbone identifier
        classes     — full list of class names
        note        — quick legend string
    """
    if not os.path.exists(MODEL_PATH):
        return {
            "label"      : "Model not trained yet",
            "grade"      : -1,
            "confidence" : 0.0,
            "model_used" : "EyeCNN_EfficientNetB3",
            "classes"    : CLASSES,
            "error"      : f"Model file not found: {MODEL_PATH}",
        }

    model = load_model(MODEL_PATH)
    img   = preprocess_image(img_path)

    if img is None:
        return {
            "label"      : "Image load failed",
            "grade"      : -1,
            "confidence" : 0.0,
            "model_used" : "EyeCNN_EfficientNetB3",
            "classes"    : CLASSES,
            "error"      : f"Could not read image: {img_path}",
        }

    probs      = model.predict(img, verbose=0)[0]
    grade      = int(np.argmax(probs))
    confidence = round(float(probs[grade]) * 100, 2)
    all_probs  = [round(float(p) * 100, 2) for p in probs]

    return {
        "label"      : CLASSES[grade],
        "grade"      : grade,
        "confidence" : confidence,
        "all_probs"  : all_probs,
        "advice"     : GRADE_INFO[grade]["advice"],
        "model_used" : "EyeCNN_EfficientNetB3",
        "classes"    : CLASSES,
        "note"       : "Grade 0 = Healthy  |  Grade 4 = Most Severe",
    }