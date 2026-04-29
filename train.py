"""
Training Script
---------------
Run this file to train all 3 models separately.
Usage: python train.py --model chest
       python train.py --model brain
       python train.py --model fracture
       python train.py --model all
"""

import argparse
import os

# ─────────────────────────────────────────────
# Uncomment these when ready to train
# ─────────────────────────────────────────────
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
# from models.chest_model    import build_chest_model
# from models.brain_model    import build_brain_model
# from models.fracture_model import build_fracture_model


# ─── CONFIG ───────────────────────────────────
EPOCHS      = 20
BATCH_SIZE  = 32
IMG_SIZE    = (224, 224)


# ─── DATA GENERATORS ──────────────────────────
def get_generators(train_dir, test_dir):
    """
    Creates train/test data generators with augmentation.
    TODO: Uncomment when ready to train.
    """
    # train_gen = ImageDataGenerator(
    #     rescale=1./255,
    #     rotation_range=15,
    #     zoom_range=0.1,
    #     horizontal_flip=True
    # )
    # test_gen = ImageDataGenerator(rescale=1./255)
    #
    # train_data = train_gen.flow_from_directory(train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE)
    # test_data  = test_gen.flow_from_directory(test_dir,  target_size=IMG_SIZE, batch_size=BATCH_SIZE)
    # return train_data, test_data
    pass


# ─── TRAIN CHEST MODEL ────────────────────────
def train_chest():
    print("[Chest] Training started...")
    # model = build_chest_model()
    # train_data, test_data = get_generators(
    #     "data/chest_xray/train",
    #     "data/chest_xray/test"
    # )
    # callbacks = [
    #     ModelCheckpoint("results/chest_model.h5", save_best_only=True),
    #     EarlyStopping(patience=5, restore_best_weights=True)
    # ]
    # model.fit(train_data, validation_data=test_data, epochs=EPOCHS, callbacks=callbacks)
    # print("[Chest] Model saved to results/chest_model.h5")
    print("[Chest] TODO: Uncomment training code above")


# ─── TRAIN BRAIN MODEL ────────────────────────
def train_brain():
    import subprocess
    print("[Brain] Running full training pipeline...")
    import sys
    subprocess.run([sys.executable, "src/train_brain.py", "--quick-train"])

# ─── TRAIN FRACTURE MODEL ─────────────────────
def train_fracture():
    print("[Fracture] Training started...")
    # model = build_fracture_model()
    # train_data, test_data = get_generators(
    #     "data/fracture/train",
    #     "data/fracture/test"
    # )
    # callbacks = [
    #     ModelCheckpoint("results/fracture_model.h5", save_best_only=True),
    #     EarlyStopping(patience=5, restore_best_weights=True)
    # ]
    # model.fit(train_data, validation_data=test_data, epochs=EPOCHS, callbacks=callbacks)
    # print("[Fracture] Model saved to results/fracture_model.h5")
    print("[Fracture] TODO: Uncomment training code above")


# ─── MAIN ─────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all",
                        choices=["chest", "brain", "fracture", "all"])
    args = parser.parse_args()

    if args.model == "chest"    : train_chest()
    elif args.model == "brain"  : train_brain()
    elif args.model == "fracture": train_fracture()
    elif args.model == "all":
        train_chest()
        train_brain()
        train_fracture()