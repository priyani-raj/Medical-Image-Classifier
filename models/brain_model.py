"""
Brain MRI Model (Model 2)
--------------------------
Dataset  : Kaggle Brain Tumor MRI Dataset
           kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
Classes  : No Tumor | Glioma | Meningioma | Pituitary
Status   : DUMMY (replace with trained model when ready)
"""

import random

# ─────────────────────────────────────────────
# STEP 1: Imports (uncomment when training)
# ─────────────────────────────────────────────
# import numpy as np
# from tensorflow.keras.models import load_model, Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, GlobalAveragePooling2D
# from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ─────────────────────────────────────────────
# STEP 2: Config
# ─────────────────────────────────────────────

##sample data insert krna h uski dimesnions
IMG_SIZE    = (224, 224)
MODEL_PATH  = "results/brain_model.h5"
CLASSES     = ["No Tumor", "Glioma", "Meningioma", "Pituitary"]


# ─────────────────────────────────────────────
# STEP 3: Model Architecture (fill when training)
# ─────────────────────────────────────────────
def build_brain_model():
    """
    CNN architecture for brain MRI classification.
    TODO: Train this with the Kaggle brain tumor dataset.
    Tip: Use transfer learning with EfficientNetB0 for better accuracy.
    """
    # from tensorflow.keras.applications import EfficientNetB0
    # base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    # base.trainable = False
    # model = Sequential([
    #     base,
    #     GlobalAveragePooling2D(),
    #     Dense(256, activation='relu'),
    #     Dropout(0.4),
    #     Dense(4, activation='softmax')   # 4 classes
    # ])
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # return model
    pass


# ─────────────────────────────────────────────
# STEP 4: Preprocessing (fill when training)
# ─────────────────────────────────────────────
def preprocess_image(img_path):
    """
    Load and preprocess a brain MRI image.
    TODO: Uncomment when real model is ready.
    """
    # img = load_img(img_path, target_size=IMG_SIZE, color_mode='rgb')
    # arr = img_to_array(img) / 255.0
    # return np.expand_dims(arr, axis=0)
    pass


# ─────────────────────────────────────────────
# STEP 5: Prediction
# ─────────────────────────────────────────────
def predict_brain(img_path: str) -> dict:
    """
    Predict brain MRI class.
    Currently returns DUMMY output — replace with real model inference.
    """

    # --- DUMMY (remove after training) ---
    label      = random.choice(CLASSES)
    confidence = round(random.uniform(70, 99), 2)
    # --------------------------------------

    # TODO: Real inference (uncomment after training)
    # model  = load_model(MODEL_PATH)
    # img    = preprocess_image(img_path)
    # probs  = model.predict(img)[0]
    # idx    = np.argmax(probs)
    # label  = CLASSES[idx]
    # confidence = round(float(probs[idx]) * 100, 2)

    return {
        "label"      : label,
        "confidence" : confidence,
        "model_used" : "BrainMRI_CNN (Dummy)",
        "classes"    : CLASSES
    }
