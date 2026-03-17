"""
Bone Fracture Model (Model 3)
------------------------------
Dataset  : Kaggle Bone Fracture Detection
           kaggle.com/datasets/pkdarabi/bone-fracture-detection-computer-vision-project
Classes  : Normal | Fractured
Status   : DUMMY (replace with trained model when ready)
"""

import random

# ─────────────────────────────────────────────
# STEP 1: Imports (uncomment when training)
# ─────────────────────────────────────────────
# import numpy as np
# from tensorflow.keras.models import load_model, Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
# from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ─────────────────────────────────────────────
# STEP 2: Config
# ─────────────────────────────────────────────
IMG_SIZE    = (224, 224)
MODEL_PATH  = "results/fracture_model.h5"
CLASSES     = ["Normal", "Fractured"]


# ─────────────────────────────────────────────
# STEP 3: Model Architecture (fill when training)
# ─────────────────────────────────────────────
def build_fracture_model():
    """
    CNN architecture for fracture detection.
    TODO: Train this with the Kaggle fracture dataset.
    """
    # from tensorflow.keras.applications import ResNet50
    # base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    # base.trainable = False
    # model = Sequential([
    #     base,
    #     Flatten(),
    #     Dense(128, activation='relu'),
    #     Dropout(0.3),
    #     Dense(1, activation='sigmoid')   # Binary: Normal vs Fractured
    # ])
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # return model
    pass


# ─────────────────────────────────────────────
# STEP 4: Preprocessing (fill when training)
# ─────────────────────────────────────────────
def preprocess_image(img_path):
    """
    Load and preprocess a bone X-ray image.
    TODO: Uncomment when real model is ready.
    """
    # img = load_img(img_path, target_size=IMG_SIZE, color_mode='rgb')
    # arr = img_to_array(img) / 255.0
    # return np.expand_dims(arr, axis=0)
    pass


# ─────────────────────────────────────────────
# STEP 5: Prediction
# ─────────────────────────────────────────────
def predict_fracture(img_path: str) -> dict:
    """
    Predict fracture in bone X-ray.
    Currently returns DUMMY output — replace with real model inference.
    """

    # --- DUMMY (remove after training) ---
    label      = random.choice(CLASSES)
    confidence = round(random.uniform(70, 99), 2)
    # --------------------------------------

    # TODO: Real inference (uncomment after training)
    # model = load_model(MODEL_PATH)
    # img   = preprocess_image(img_path)
    # prob  = model.predict(img)[0][0]
    # label = CLASSES[1] if prob > 0.5 else CLASSES[0]
    # confidence = round(float(prob if prob > 0.5 else 1 - prob) * 100, 2)

    return {
        "label"      : label,
        "confidence" : confidence,
        "model_used" : "FractureCNN (Dummy)",
        "classes"    : CLASSES
    }