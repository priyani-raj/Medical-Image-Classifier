"""
main.py — Entry Point
======================
Medical Image Classification System
JIIT Noida | AI & ML Lab | 3rd Semester
Team: Ishita Arora | Priyani Rajvanshi | Hrishita Raj Singh | Samman Singh

Run:
    python main.py                          # interactive mode
    python main.py --img path/to/img.jpg --type eye
"""

import os
import argparse
from router import predict_image

BANNER = """
╔══════════════════════════════════════════════╗
║   Medical Image Classification System       ║
║   JIIT Noida | AI & ML Lab | 3rd Sem        ║
╚══════════════════════════════════════════════╝
"""

def run_single(img_path, img_type):
    if not os.path.exists(img_path):
        print(f"[Error] Image not found: {img_path}")
        return

    print(f"\nImage : {img_path}")
    print(f"Type  : {img_type.upper()}")
    print("-" * 44)

    result = predict_image(img_path, img_type)

    if "error" in result:
        print(f"[Error] {result['error']}")
        return

    print(f"Label      : {result['label']}")
    print(f"Confidence : {result['confidence']}%")
    print(f"Model      : {result['model_used']}")

    if "grade" in result:
        print(f"DR Grade   : {result['grade']}")
    if "note" in result:
        print(f"Note       : {result['note']}")


def interactive_mode():
    print(BANNER)
    print("Supported types: chest | fracture | eye | brain\n")

    while True:
        img_path = input("Image path (or 'q' to quit): ").strip()
        if img_path.lower() == 'q':
            break

        img_type = input("Image type [chest/fracture/eye/brain]: ").strip()
        run_single(img_path, img_type)
        print()


def main():
    parser = argparse.ArgumentParser(description="Medical Image Classifier")
    parser.add_argument("--img",  type=str, help="Path to image file")
    parser.add_argument("--type", type=str, help="Image type: chest | fracture | eye | brain")
    args = parser.parse_args()

    if args.img and args.type:
        print(BANNER)
        run_single(args.img, args.type)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()