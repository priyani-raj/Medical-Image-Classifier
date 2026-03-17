"""
Router Module
-------------
Decides which specialist model to call based on image type.
In a real deployment, this can auto-detect image type using metadata or a small classifier.
For now, image_type is passed manually.
"""

from models.chest_model import predict_chest
from models.brain_model import predict_brain
from models.fracture_model import predict_fracture
from models.eye_model import predict_eye


def predict_image(img_path: str, image_type: str) -> dict:
    """
    Routes an image to the correct specialist model.

    Args:
        img_path   : Path to the image file
        image_type : One of 'chest', 'brain', 'fracture', 'eye'

    Returns:
        dict with keys: label, confidence, model_used
    """

    image_type = image_type.lower().strip()

    if image_type == "chest":
        return predict_chest(img_path)

    elif image_type == "brain":
        return predict_brain(img_path)

    elif image_type == "fracture":
        return predict_fracture(img_path)

    elif image_type == "eye":
        return predict_eye(img_path)

    else:
        return {
            "label": "Unknown",
            "confidence": 0,
            "model_used": "None",
            "error": f"Invalid image_type '{image_type}'. Use: chest | brain | fracture | eye"
        }