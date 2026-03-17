"""
Eye / Retinal Model (Model 4)
------------------------------
Dataset  : Kaggle - Diabetic Retinopathy 2015 (Resized Version)
           kaggle.com/datasets/sovitrath/diabetic-retinopathy-2015-data-colored-resized
Classes  : 0 - No DR
           1 - Mild
           2 - Moderate
           3 - Severe
           4 - Proliferative DR
Status   : DUMMY (replace with trained model when ready)

Note     : This is a 5-class classification problem.
           Dataset is imbalanced — class 0 (No DR) has the most images.
           Use class_weight during training to handle this.
"""

import random

# ─────────────────────────────────────────────
# STEP 1: Imports (uncomment when training)
# ─────────────────────────────────────────────
# import numpy as np
# from tensorflow.keras.models import load_model, Sequential
# from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from tensorflow.keras.applications import EfficientNetB3


# ─────────────────────────────────────────────
# STEP 2: Config
# ─────────────────────────────────────────────
IMG_SIZE    = (224, 224)
MODEL_PATH  = "results/eye_model.h5"
CLASSES     = [
    "No DR (Grade 0)",
    "Mild DR (Grade 1)",
    "Moderate DR (Grade 2)",
    "Severe DR (Grade 3)",
    "Proliferative DR (Grade 4)"
]


# ─────────────────────────────────────────────
# STEP 3: Model Architecture
# ─────────────────────────────────────────────
def build_eye_model():
    """
    CNN architecture for diabetic retinopathy classification.

    Why EfficientNetB3?
    - Retinal images are detailed and high-resolution
    - EfficientNetB3 is better at capturing fine-grained features
      compared to VGG16 or ResNet (used in chest/fracture models)
    - Pretrained on ImageNet → good base for medical images

    TODO: Uncomment and run after downloading dataset.
    """

    # base = EfficientNetB3(
    #     weights     = 'imagenet',
    #     include_top = False,
    #     input_shape = (224, 224, 3)
    # )
    # base.trainable = False   # Freeze base layers first (Phase 1)
    #
    # model = Sequential([
    #     base,
    #     GlobalAveragePooling2D(),
    #     Dense(512, activation='relu'),
    #     Dropout(0.5),
    #     Dense(256, activation='relu'),
    #     Dropout(0.3),
    #     Dense(5, activation='softmax')    # 5 classes (Grade 0 to 4)
    # ])
    #
    # model.compile(
    #     optimizer = 'adam',
    #     loss      = 'categorical_crossentropy',
    #     metrics   = ['accuracy']
    # )
    # return model
    pass


# ─────────────────────────────────────────────
# STEP 4: Preprocessing
# ─────────────────────────────────────────────
def preprocess_image(img_path):
    """
    Load and preprocess a retinal fundus image.

    Note: Retinal images benefit from CLAHE (contrast enhancement)
    before feeding into the model — improves feature visibility.
    TODO: Uncomment when real model is ready.
    """

    # img = load_img(img_path, target_size=IMG_SIZE, color_mode='rgb')
    # arr = img_to_array(img) / 255.0
    #
    # Optional: Apply CLAHE for better contrast
    # import cv2
    # img_cv  = cv2.imread(img_path)
    # img_cv  = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
    # l, a, b = cv2.split(img_cv)
    # clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # l       = clahe.apply(l)
    # img_cv  = cv2.merge((l, a, b))
    # img_cv  = cv2.cvtColor(img_cv, cv2.COLOR_LAB2BGR)
    # arr     = img_cv / 255.0
    #
    # return np.expand_dims(arr, axis=0)
    pass


# ─────────────────────────────────────────────
# STEP 5: Handle Class Imbalance
# ─────────────────────────────────────────────
def get_class_weights():
    """
    Diabetic retinopathy dataset is heavily imbalanced.
    Grade 0 (No DR) dominates — without this, model just predicts 0 for everything.

    TODO: Compute from your actual dataset distribution.
    Use sklearn's compute_class_weight for exact values.
    """

    # from sklearn.utils.class_weight import compute_class_weight
    # weights = compute_class_weight(
    #     class_weight = 'balanced',
    #     classes      = np.array([0, 1, 2, 3, 4]),
    #     y            = train_labels    # your actual label array
    # )
    # return dict(enumerate(weights))

    # Approximate manual weights based on known dataset distribution
    return {
        0: 1.0,    # No DR       — most common, least weight
        1: 3.5,    # Mild        
        2: 2.0,    # Moderate    
        3: 6.0,    # Severe      — rare, high weight
        4: 5.0     # Proliferative DR — rare, high weight
    }


# ─────────────────────────────────────────────
# STEP 6: Prediction
# ─────────────────────────────────────────────
def predict_eye(img_path: str) -> dict:
    """
    Predict diabetic retinopathy grade from retinal image.
    Currently returns DUMMY output — replace with real model inference.
    """

    # --- DUMMY (remove after training) ---
    label      = random.choice(CLASSES)
    grade      = CLASSES.index(label)
    confidence = round(random.uniform(70, 99), 2)
    # --------------------------------------

    # TODO: Real inference (uncomment after training)
    # model  = load_model(MODEL_PATH)
    # img    = preprocess_image(img_path)
    # probs  = model.predict(img)[0]
    # grade  = int(np.argmax(probs))
    # label  = CLASSES[grade]
    # confidence = round(float(probs[grade]) * 100, 2)

    return {
        "label"       : label,
        "grade"       : grade,
        "confidence"  : confidence,
        "model_used"  : "EyeCNN_EfficientNetB3 (Dummy)",
        "classes"     : CLASSES,
        "note"        : "Grade 0 = Healthy | Grade 4 = Most Severe"
    }