"""
router.py — Model Router
-------------------------
Routes an image to the correct specialist model based on image_type.
Each model is independent — only the relevant one is loaded into memory.

image_type options:  'chest' | 'fracture' | 'eye' | 'brain'
"""

from models.chest_model    import predict_chest
from models.fracture_model import predict_fracture
from models.eye_model      import predict_eye
from models.brain_model    import predict_brain   # still dummy — train later


def predict_image(img_path: str, image_type: str) -> dict:
    """
    Routes an image to the correct specialist model.

    Args:
        img_path   : Path to the image file
        image_type : One of 'chest', 'fracture', 'eye', 'brain'

    Returns:
        dict with keys: label, confidence, model_used, classes
        (eye model also returns: grade, note)
    """
    image_type = image_type.lower().strip()

    if image_type == "chest":
        return predict_chest(img_path)

    elif image_type == "fracture":
        return predict_fracture(img_path)

    elif image_type == "eye":
        return predict_eye(img_path)

    elif image_type == "brain":
        return predict_brain(img_path)

    else:
        return {
            "label"      : "Unknown",
            "confidence" : 0,
            "model_used" : "None",
            "error"      : f"Invalid image_type '{image_type}'. Use: chest | fracture | eye | brain"
        }