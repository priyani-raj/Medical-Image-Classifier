"""
Medical Image Classification System
JIIT Noida - AI & ML Lab Project
Team: Ishita, Priyani, Utkarsh, Samman
"""

import os
import sys
from router import predict_image

def main():
    print("=" * 50)
    print("  Medical Image Classification System")
    print("  JIIT Noida | AI & ML Lab")
    print("=" * 50)

    # --- DEMO MODE (before real training) ---
    # Once models are trained, replace this with actual image path input

    demo_images = [
        ("data/chest_xray/test/pneumonia/sample.jpg",   "chest"),
        ("data/brain_mri/test/glioma/sample.jpg",       "brain"),
        ("data/fracture/test/fractured/sample.jpg",     "fracture"),
        ("data/retina/test/moderate/sample.jpg",        "eye"),
    ]

    print("\n[DEMO] Running predictions on dummy samples...\n")

    for img_path, img_type in demo_images:
        result = predict_image(img_path, img_type)
        print(f"Image : {img_path}")
        print(f"Type  : {img_type.upper()}")
        print(f"Result: {result['label']} ({result['confidence']}% confidence)")
        print("-" * 40)

if __name__ == "__main__":
    main()