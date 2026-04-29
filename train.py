"""
train.py — Master Training Script
===================================
Trains chest, fracture, brain and eye (Diabetic Retinopathy) models.

Usage:
    python train.py --model chest
    python train.py --model fracture
    python train.py --model eye
    python train.py --model all

Outputs saved to results/:
    chest_model.h5          fracture_model.h5       eye_model.h5
    eye_model_phase1.h5     (phase-1 checkpoint, useful for debugging)
    training_plot_chest.png
    training_plot_fracture.png
    training_plot_eye.png

KEY FIX (eye model):
    ImageDataGenerator now uses EfficientNet's preprocess_input as its
    preprocessing_function — exactly what eye_model.py uses at inference
    time.  The old version used preprocess_input in training but CLAHE
    + /255 in inference, causing a distribution mismatch that made the
    model predict Grade 0 for every image.
"""

import os
import shutil
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)
from sklearn.model_selection import train_test_split

# ── import only what belongs to each specialist model ──────────────────────
from models.chest_model    import build_chest_model, unfreeze_chest
from models.fracture_model import build_fracture_model, unfreeze_fracture
from models.eye_model      import (
    build_eye_model, unfreeze_for_finetuning, get_class_weights
)

os.makedirs("results", exist_ok=True)

# ─── CONFIG ──────────────────────────────────────────────────────────────────
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32


# ─── HELPERS ─────────────────────────────────────────────────────────────────
def plot_history(h1, h2, name):
    """Plot train/val accuracy and loss across both training phases."""
    acc   = h1.history["accuracy"]     + (h2.history["accuracy"]     if h2 else [])
    val   = h1.history["val_accuracy"] + (h2.history["val_accuracy"] if h2 else [])
    loss  = h1.history["loss"]         + (h2.history["loss"]         if h2 else [])
    vloss = h1.history["val_loss"]     + (h2.history["val_loss"]     if h2 else [])
    p1    = len(h1.history["accuracy"])
    ep    = range(1, len(acc) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{name} — Training History", fontsize=13, fontweight="bold")

    axes[0].plot(ep, acc,  label="Train Accuracy",  linewidth=2)
    axes[0].plot(ep, val,  label="Val Accuracy",    linewidth=2)
    if h2:
        axes[0].axvline(p1, color="gray", linestyle="--", label="Phase 2 start")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(ep, loss,  label="Train Loss",  linewidth=2)
    axes[1].plot(ep, vloss, label="Val Loss",    linewidth=2)
    if h2:
        axes[1].axvline(p1, color="gray", linestyle="--", label="Phase 2 start")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = f"results/training_plot_{name.lower()}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {path}")


def standard_callbacks(model_path, monitor="val_accuracy"):
    return [
        ModelCheckpoint(model_path, monitor=monitor,
                        save_best_only=True, verbose=1),
        EarlyStopping(monitor=monitor, patience=5,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                         patience=3, min_lr=1e-7, verbose=1),
    ]


# ─── BINARY GENERATORS (chest / fracture — unchanged) ────────────────────────
def binary_generators(train_dir, test_dir, preprocess_fn):
    """
    For binary classification models (chest, fracture).
    Uses each backbone's own preprocess_input, not manual /255.
    """
    train_gen = ImageDataGenerator(
        preprocessing_function = preprocess_fn,
        rotation_range         = 15,
        zoom_range             = 0.1,
        horizontal_flip        = True,
        fill_mode              = "reflect",
    )
    test_gen = ImageDataGenerator(preprocessing_function=preprocess_fn)

    train_data = train_gen.flow_from_directory(
        train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="binary", shuffle=True,
    )
    test_data = test_gen.flow_from_directory(
        test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="binary", shuffle=False,
    )
    return train_data, test_data


# ─── TRAIN CHEST (unchanged) ─────────────────────────────────────────────────
def train_chest():
    """
    Dataset expected at:
        data/chest_data/train/NORMAL/
        data/chest_data/train/PNEUMONIA/
        data/chest_data/test/NORMAL/
        data/chest_data/test/PNEUMONIA/
    """
    from tensorflow.keras.applications.vgg16 import preprocess_input

    print("\n" + "=" * 55)
    print("  CHEST MODEL TRAINING")
    print("=" * 55)

    train_data, test_data = binary_generators(
        "data/chest_data/train", "data/chest_data/test", preprocess_input
    )

    print("\n[Chest] Phase 1 — frozen VGG16, training head (LR=1e-3)...")
    model = build_chest_model()
    h1 = model.fit(
        train_data, validation_data=test_data, epochs=10,
        callbacks=standard_callbacks("results/chest_model_p1.h5"),
    )
    print(f"  Best val_acc: {max(h1.history['val_accuracy'])*100:.2f}%")

    print("\n[Chest] Phase 2 — fine-tuning last 4 VGG16 layers (LR=1e-5)...")
    model = unfreeze_chest(model, unfreeze_last_n=4)
    h2 = model.fit(
        train_data, validation_data=test_data, epochs=15,
        callbacks=standard_callbacks("results/chest_model.h5"),
    )
    print(f"  Best val_acc: {max(h2.history['val_accuracy'])*100:.2f}%")
    print("  Saved → results/chest_model.h5")
    plot_history(h1, h2, "Chest")



# ─── TRAIN FRACTURE MODEL ─────────────────────
def train_fracture():
    """
    Dataset: Kaggle Bone Fracture Detection (YOLO layout).
    prepare_fracture_keras_dirs() converts it to flow_from_directory format
    on the first run; subsequent runs reuse the already-converted data.
    """
    from tensorflow.keras.applications.resnet50 import preprocess_input

    print("\n" + "=" * 55)
    print("  FRACTURE MODEL TRAINING")
    print("=" * 55)

    train_dir, test_dir = prepare_fracture_keras_dirs()

    train_data, test_data = binary_generators(train_dir, test_dir, preprocess_input)

    print("\n[Fracture] Phase 1 — frozen ResNet50, training head (LR=1e-3)...")
    model = build_fracture_model()
    h1 = model.fit(
        train_data, validation_data=test_data, epochs=10,
        callbacks=standard_callbacks("results/fracture_model_p1.h5"),
    )
    print(f"  Best val_acc: {max(h1.history['val_accuracy'])*100:.2f}%")

    print("\n[Fracture] Phase 2 — fine-tuning last 10 ResNet50 layers (LR=1e-5)...")
    model = unfreeze_fracture(model, unfreeze_last_n=10)
    h2 = model.fit(
        train_data, validation_data=test_data, epochs=20,
        callbacks=standard_callbacks("results/fracture_model.h5"),
    )
    print(f"  Best val_acc: {max(h2.history['val_accuracy'])*100:.2f}%")
    print("  Saved → results/fracture_model.h5")
    plot_history(h1, h2, "Fracture")


# ─── TRAIN EYE (DR only) ─────────────────────────────────────────────────────
def prepare_eye_split(source_dir, output_dir, test_size=0.2, seed=42):
    """
    Splits the flat Kaggle colored_images folder into:
        output_dir/train/0/  …  output_dir/train/4/
        output_dir/test/0/   …  output_dir/test/4/

    Kaggle folder names → DR Grade mapping:
        No_DR          → 0
        Mild           → 1
        Moderate       → 2
        Severe         → 3
        Proliferate_DR → 4
    """
    train_dir = os.path.join(output_dir, "train")
    test_dir  = os.path.join(output_dir, "test")

    if os.path.exists(train_dir):
        print("[Eye Split] Train/test split already exists — skipping.\n")
        return train_dir, test_dir

    class_map = {
        "No_DR"          : "0",
        "Mild"           : "1",
        "Moderate"       : "2",
        "Severe"         : "3",
        "Proliferate_DR" : "4",
    }

    print("[Eye Split] Creating 80/20 stratified split...")
    for folder, grade in class_map.items():
        src = os.path.join(source_dir, folder)
        if not os.path.exists(src):
            print(f"  [!] Missing: {src} — skipping grade {grade}")
            continue
        imgs = [
            f for f in os.listdir(src)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        train_imgs, test_imgs = train_test_split(
            imgs, test_size=test_size, random_state=seed
        )
        for split, subset in [("train", train_imgs), ("test", test_imgs)]:
            dest = os.path.join(output_dir, split, grade)
            os.makedirs(dest, exist_ok=True)
            for img in subset:
                shutil.copy2(os.path.join(src, img), os.path.join(dest, img))
        print(
            f"  Grade {grade} ({folder:20s}): "
            f"{len(train_imgs)} train | {len(test_imgs)} test"
        )
    print("[Eye Split] Done.\n")
    return train_dir, test_dir


def train_eye():
    """
    Trains the Diabetic Retinopathy grading model (5-class).

    Expected dataset layout (Kaggle download):
        data/eye_data/colored_images/No_DR/
        data/eye_data/colored_images/Mild/
        data/eye_data/colored_images/Moderate/
        data/eye_data/colored_images/Severe/
        data/eye_data/colored_images/Proliferate_DR/

    The script auto-splits into train/ and test/ subfolders on first run.

    PREPROCESSING (critical):
        Uses EfficientNet's preprocess_input as the ImageDataGenerator's
        preprocessing_function.  This is identical to what eye_model.py
        uses during inference, so there is NO distribution mismatch.
    """
    # ── import AFTER the model module sets things up ──────────────────────
    from tensorflow.keras.applications.efficientnet import preprocess_input

    print("\n" + "=" * 55)
    print("  EYE MODEL TRAINING  (Diabetic Retinopathy — 5 grades)")
    print("=" * 55)

    train_dir, test_dir = prepare_eye_split(
        source_dir = "data/eye_data/colored_images",
        output_dir = "data/eye_data",
    )

    # ── Data generators ───────────────────────────────────────────────────
    # preprocessing_function = preprocess_input  ← same as inference
    train_gen = ImageDataGenerator(
        preprocessing_function = preprocess_input,   # ← KEY FIX
        rotation_range         = 20,
        zoom_range             = 0.15,
        horizontal_flip        = True,
        vertical_flip          = True,
        brightness_range       = [0.85, 1.15],
        fill_mode              = "reflect",
    )
    test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_data = train_gen.flow_from_directory(
        train_dir,
        target_size = IMG_SIZE,
        batch_size  = BATCH_SIZE,
        classes     = ["0", "1", "2", "3", "4"],
        class_mode  = "categorical",
        shuffle     = True,
    )
    test_data = test_gen.flow_from_directory(
        test_dir,
        target_size = IMG_SIZE,
        batch_size  = BATCH_SIZE,
        classes     = ["0", "1", "2", "3", "4"],
        class_mode  = "categorical",
        shuffle     = False,
    )

    class_weights = get_class_weights(train_data.classes)
    print(f"[Eye] Class weights (sqrt-dampened): {class_weights}\n")

    # ── Phase 1: train head only (backbone frozen) ────────────────────────
    print("[Eye] Phase 1 — frozen EfficientNetB3 backbone, training head (LR=1e-3)...")
    model = build_eye_model()
    h1 = model.fit(
        train_data,
        validation_data = test_data,
        epochs          = 15,
        class_weight    = class_weights,
        callbacks       = standard_callbacks("results/eye_model_phase1.h5"),
    )
    print(f"  Best val_acc (Phase 1): {max(h1.history['val_accuracy'])*100:.2f}%")

    # ── Phase 2: fine-tune last 30 EfficientNetB3 layers ─────────────────
    print("\n[Eye] Phase 2 — fine-tuning last 30 backbone layers (LR=1e-5)...")
    model = unfreeze_for_finetuning(model, unfreeze_last_n=30)
    h2 = model.fit(
        train_data,
        validation_data = test_data,
        epochs          = 25,
        class_weight    = class_weights,
        callbacks       = [
            ModelCheckpoint(
                "results/eye_model.h5",
                monitor        = "val_accuracy",
                save_best_only = True,
                verbose        = 1,
            ),
            EarlyStopping(
                monitor               = "val_accuracy",
                patience              = 7,
                restore_best_weights  = True,
                verbose               = 1,
            ),
            ReduceLROnPlateau(
                monitor  = "val_loss",
                factor   = 0.5,
                patience = 3,
                min_lr   = 1e-7,
                verbose  = 1,
            ),
        ],
    )
    print(f"  Best val_acc (Phase 2): {max(h2.history['val_accuracy'])*100:.2f}%")
    print("  Saved → results/eye_model.h5")
    plot_history(h1, h2, "Eye")


# ─── MAIN ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medical Image Classifier — Training")
    parser.add_argument(
        "--model",
        type    = str,
        default = "all",
        choices = ["chest", "fracture", "eye", "all"],
        help    = "Which model to train",
    )
    args = parser.parse_args()

    if args.model == "chest":
        train_chest()
    elif args.model == "fracture":
        train_fracture()
    elif args.model == "eye":
        train_eye()
    elif args.model == "all":
        train_chest()
        train_fracture()
        train_eye()

    print("\n[Done] Check the results/ folder for .h5 files and training plots.")