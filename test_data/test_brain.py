import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load model
model = load_model("results/brain/brain_tumor_model.h5")  # or .keras

# Classes
classes = ["No Tumor", "Glioma", "Meningioma", "Pituitary"]

# Load image
img_path = "test.jpg"   # <-- your test image
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (128, 128))
img = img / 255.0

# Predict
img = np.expand_dims(img, axis=0)
pred = model.predict(img)

class_idx = np.argmax(pred)
confidence = np.max(pred)

print("Prediction:", classes[class_idx])
print("Confidence:", round(confidence * 100, 2), "%")