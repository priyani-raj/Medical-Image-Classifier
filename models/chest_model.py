"""
Chest X-Ray Model (Model 1)
---------------------------
Dataset  : Kaggle Chest X-Ray Pneumonia
           kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
Classes  : Normal | Pneumonia
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
MODEL_PATH  = "results/chest_model.h5"
CLASSES     = ["Normal", "Pneumonia"]


# ─────────────────────────────────────────────
# STEP 3: Model Architecture (fill when training)
# ─────────────────────────────────────────────
def build_chest_model():
    """
    CNN architecture for chest X-ray classification.
    TODO: Train this with the Kaggle pneumonia dataset.
    """
    # from tensorflow.keras.applications import VGG16
    # base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    # base.trainable = False
    # model = Sequential([
    #     base,
    #     Flatten(),
    #     Dense(256, activation='relu'),
    #     Dropout(0.5),
    #     Dense(1, activation='sigmoid')   # Binary: Normal vs Pneumonia
    # ])
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # return model
    pass


# ─────────────────────────────────────────────
# STEP 4: Preprocessing (fill when training)
# ─────────────────────────────────────────────
def preprocess_image(img_path):
    """
    Load and preprocess a chest X-ray image.
    TODO: Uncomment when real model is ready.
    """
    # img = load_img(img_path, target_size=IMG_SIZE, color_mode='rgb')
    # arr = img_to_array(img) / 255.0
    # return np.expand_dims(arr, axis=0)
    pass


# ─────────────────────────────────────────────
# STEP 5: Prediction
# ─────────────────────────────────────────────
def predict_chest(img_path: str) -> dict:
    """
    Predict chest X-ray class.
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
        "model_used" : "ChestCNN (Dummy)",
        "classes"    : CLASSES
    }