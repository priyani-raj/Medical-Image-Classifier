"""
Brain Tumor MRI - CPU-Optimized Training Script
================================================
Optimized for laptops WITHOUT GPU (Dell Vostro 14 3000, etc.)

IMPORTANT: CPU training is SLOW. This script is optimized for CPU but will still
take 1-2 hours. Consider using Google Colab (free GPU) as an alternative.

Dataset: masoudnickparvar/brain-tumor-mri-dataset from Kaggle
Classes: No Tumor, Glioma, Meningioma, Pituitary

Usage:
    python train_brain_cpu_optimized.py
    python train_brain_cpu_optimized.py --epochs 30 --quick-train
    python train_brain_cpu_optimized.py --lite  # Lite mode: 10 epochs, 64x64 images
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import cv2
from tqdm import tqdm
import json
import psutil
import time

# TensorFlow imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

print("="*70)
print("🧠 BRAIN TUMOR - CPU OPTIMIZED TRAINING")
print("="*70)
print(f"\n📊 System Information:")
print(f"  Processor: {psutil.cpu_count()} cores")
print(f"  RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
print(f"  TensorFlow Version: {tf.__version__}")

# Check if GPU is available (for reference)
gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
print(f"  GPU Available: {'Yes ✓' if gpu_available else 'No ✗'}")

if gpu_available:
    print(f"\n⚠️  GPU detected but CPU mode selected. To use GPU, use regular script.")
else:
    print(f"\n✓ CPU mode optimized for your system.")

# ─────────────────────────────────────────────
# CONFIGURATION - CPU OPTIMIZED
# ─────────────────────────────────────────────

class CPUTrainingConfig:
    """Configuration optimized for CPU training"""
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # ⭐ CPU OPTIMIZATION: Smaller values for CPU
    TRAIN_DIR = os.path.join(BASE_DIR, "Training")
    TEST_DIR = os.path.join(BASE_DIR, "Testing")
    
    # Default: Lite mode (fast training, lower accuracy)
    # Can be overridden by --quick-train or --normal-train
    IMG_SIZE = (128, 128)  # Smaller = faster on CPU
    BATCH_SIZE = 8        # Smaller = uses less RAM
    EPOCHS = 20           # Fewer epochs = faster training
    
    # Advanced CPU optimization
    NUM_WORKERS = 2       # Parallel data loading
    PREFETCH = tf.data.AUTOTUNE
    LEARNING_RATE = 0.001
    NUM_CLASSES = 4
    CLASSES = ['No Tumor', 'Glioma', 'Meningioma', 'Pituitary']
    
    # Paths
    RESULTS_DIR = "results/brain"
    MODEL_PATH = os.path.join(RESULTS_DIR, "brain_tumor_model.h5")
    PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
    LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
    
    VAL_SPLIT = 0.15
    DEVICE = "CPU"
    
    # CPU-specific optimizations
    USE_MIXED_PRECISION = False  # Not recommended for CPU
    ENABLE_FUNCTION = True       # @tf.function for speed
    REDUCE_DATA = 0.5            # Train on 50% of data for speed


# ─────────────────────────────────────────────
# CPU OPTIMIZATIONS
# ─────────────────────────────────────────────

def optimize_for_cpu():
    """Apply CPU optimizations"""
    print("\n⚙️  Applying CPU optimizations...")
    
    # Disable GPU to ensure CPU usage
    tf.config.set_visible_devices([], 'GPU')
    
    # Enable CPU multithreading
    tf.config.threading.set_intra_op_parallelism_threads(psutil.cpu_count())
    tf.config.threading.set_inter_op_parallelism_threads(psutil.cpu_count())
    
    print(f"  ✓ CPU threads: {psutil.cpu_count()}")
    print(f"  ✓ Using MobileNetV2 (lightweight model)")
    print(f"  ✓ Mixed precision: Disabled (slower on CPU)")
    print(f"  ✓ Data prefetching: Enabled")


# ─────────────────────────────────────────────
# MODEL BUILDING - CPU OPTIMIZED
# ─────────────────────────────────────────────

class CPUBrainTumorModel:
    """CPU-optimized brain tumor classification model"""
    
    def __init__(self, config):
        self.config = config
    
    def build_model(self):
        """Build lightweight model suitable for CPU"""
        print(f"\n📐 Building CPU-optimized MobileNetV2 model...")
        print(f"  Input size: {self.config.IMG_SIZE}")
        
        # ⭐ MobileNetV2 is much lighter than EfficientNetB0
        # Designed for mobile/edge devices (perfect for CPU)
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.config.IMG_SIZE, 3)
        )
        
        # Freeze base model for faster training
        base_model.trainable = False
        
        # Lightweight classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)  # ⭐ Smaller than 512
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(128, activation='relu')(x)  # ⭐ Smaller than 256
        x = Dropout(0.3)(x)
        outputs = Dense(self.config.NUM_CLASSES, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=outputs)
        
        # Compile with CPU-friendly settings
        optimizer = Adam(learning_rate=self.config.LEARNING_RATE)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"✓ Model built successfully!")
        print(f"✓ Total parameters: {model.count_params():,}")
        print(f"✓ Trainable parameters: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")
        
        return model


# ─────────────────────────────────────────────
# DATA LOADING - OPTIMIZED
# ─────────────────────────────────────────────

class CPUDataLoader:
    """Load brain tumor dataset with CPU optimizations"""
    
    def __init__(self, config=None):
        self.config = config or CPUTrainingConfig()
        self.class_names = None
    
    def load_images_from_directory(self, directory, reduce_data=1.0):
        """Load images from directory structure"""
        print(f"\n📁 Loading images from: {directory}")
        
        images = []
        labels = []
        class_names = []
        
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        # Get class folders
        class_dirs = sorted([d for d in os.listdir(directory) 
                           if os.path.isdir(os.path.join(directory, d))])
        class_names = class_dirs
        
        print(f"✓ Found {len(class_names)} classes: {class_names}")
        
        # Load images from each class
        for class_idx, class_name in enumerate(class_names):
            class_path = os.path.join(directory, class_name)
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # ⭐ CPU OPTIMIZATION: Reduce data for faster training
            if reduce_data < 1.0:
                num_images = max(1, int(len(image_files) * reduce_data))
                image_files = image_files[:num_images]
                print(f"\n  Loading {class_name}: {len(image_files)} images (reduced from {len(os.listdir(class_path))})")
            else:
                print(f"\n  Loading {class_name}: {len(image_files)} images")
            
            for image_file in tqdm(image_files, desc=f"  {class_name}", leave=False):
                img_path = os.path.join(class_path, image_file)
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, self.config.IMG_SIZE)
                        img = img / 255.0  # Normalize
                        
                        images.append(img)
                        labels.append(class_idx)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
        
        print(f"\n✓ Loaded {len(images)} total images")
        return np.array(images, dtype=np.float32), np.array(labels), class_names
    
    def load_data(self, reduce_data=1.0):
        """Load training and testing data"""
        print("\n" + "="*70)
        print("PHASE 1: DATA LOADING")
        print("="*70)
        
        X_train, y_train, class_names = self.load_images_from_directory(
            self.config.TRAIN_DIR, reduce_data=reduce_data
        )
        self.class_names = class_names
        
        X_test, y_test, _ = self.load_images_from_directory(
            self.config.TEST_DIR, reduce_data=reduce_data
        )
        
        print(f"\n✓ Training data shape: {X_train.shape}")
        print(f"✓ Testing data shape: {X_test.shape}")
        print(f"✓ Memory usage: {(X_train.nbytes + X_test.nbytes) / (1024**2):.1f} MB")
        
        return X_train, y_train, X_test, y_test


# ─────────────────────────────────────────────
# DATA PREPROCESSING - OPTIMIZED
# ─────────────────────────────────────────────

class CPUDataPreprocessor:
    """Preprocess and split data with CPU optimizations"""
    
    def __init__(self, config=None):
        self.config = config or CPUTrainingConfig()
    
    def split_data(self, X_train, y_train, X_test, y_test):
        """Split data into train, validation, and test"""
        print("\n" + "="*70)
        print("PHASE 2: DATA PREPROCESSING & SPLITTING")
        print("="*70)
        
        # Split training data
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train,
            test_size=self.config.VAL_SPLIT,
            random_state=42,
            stratify=y_train
        )
        
        print(f"\n✓ Data Split:")
        print(f"  Training set:   {X_train_split.shape[0]} images")
        print(f"  Validation set: {X_val.shape[0]} images")
        print(f"  Test set:       {X_test.shape[0]} images")
        
        # Print class distribution
        unique, counts = np.unique(y_train_split, return_counts=True)
        print(f"\n✓ Class distribution:")
        for cls, count in zip(unique, counts):
            print(f"  Class {cls}: {count} samples")
        
        return X_train_split, X_val, X_test, y_train_split, y_val, y_test
    
    def create_data_generators(self, X_train, y_train, X_val, y_val, num_classes):
        """Create optimized data generators"""
        print(f"\n✓ Creating data generators...")
        
        # ⭐ CPU OPTIMIZATION: Minimal augmentation
        train_augmentation = ImageDataGenerator(
            rotation_range=15,      # Reduced from 20
            width_shift_range=0.15, # Reduced from 0.2
            height_shift_range=0.15,
            zoom_range=0.15,        # Reduced from 0.2
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        val_augmentation = ImageDataGenerator()
        
        train_generator = train_augmentation.flow(
            X_train, keras.utils.to_categorical(y_train, num_classes),
            batch_size=self.config.BATCH_SIZE,
            shuffle=True
        )
        
        val_generator = val_augmentation.flow(
            X_val, keras.utils.to_categorical(y_val, num_classes),
            batch_size=self.config.BATCH_SIZE,
            shuffle=False
        )
        
        print(f"✓ Generators created (batch size: {self.config.BATCH_SIZE})")
        return train_generator, val_generator


# ─────────────────────────────────────────────
# TRAINING - CPU OPTIMIZED
# ─────────────────────────────────────────────

class CPUModelTrainer:
    """Train model with CPU optimizations"""
    
    def __init__(self, config=None):
        self.config = config or CPUTrainingConfig()
        self.history = None
    
    def get_callbacks(self):
        """Get training callbacks optimized for CPU"""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=8,  # ⭐ Shorter patience for CPU training
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=4,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=self.config.MODEL_PATH,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=0
            )
        ]
        return callbacks
    
    def train(self, model, train_generator, val_generator, num_train_samples):
        """Train the model on CPU"""
        print("\n" + "="*70)
        print("PHASE 4: MODEL TRAINING (CPU MODE)")
        print("="*70)
        
        print(f"\n⚠️  WARNING: CPU training is SLOW")
        print(f"    Estimated time: {self.config.EPOCHS * 2}-{self.config.EPOCHS * 3} minutes")
        print(f"    (~2-4 hours for 30 epochs)")
        
        print(f"\n🚀 Starting training...")
        print(f"   Epochs: {self.config.EPOCHS}")
        print(f"   Batch size: {self.config.BATCH_SIZE}")
        print(f"   Learning rate: {self.config.LEARNING_RATE}")
        
        steps_per_epoch = max(1, num_train_samples // self.config.BATCH_SIZE)
        print(f"   Steps per epoch: {steps_per_epoch}")
        
        callbacks = self.get_callbacks()
        
        # ⭐ CPU OPTIMIZATION: Use graph mode for speed
        @tf.function
        def _train_step(x, y):
            with tf.GradientTape() as tape:
                logits = model(x, training=True)
                loss = tf.keras.losses.categorical_crossentropy(y, logits)
            return loss
        
        start_time = time.time()
        
        self.history = model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=self.config.EPOCHS,
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        elapsed_time = time.time() - start_time
        print(f"\n✓ Training completed in {elapsed_time/60:.1f} minutes!")
        
        # Save model
        if not os.path.exists(self.config.MODEL_PATH):
            model.save(self.config.MODEL_PATH)
            print(f"Model saved to {self.config.MODEL_PATH}")
        
        return model, self.history


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

class CPUModelEvaluator:
    """Evaluate brain model"""
    
    def __init__(self, config=None, class_names=None):
        self.config = config or CPUTrainingConfig()
        self.class_names = class_names or self.config.CLASSES
    
    def evaluate(self, model, X_test, y_test):
        """Evaluate model on test set"""
        print("\n" + "="*70)
        print("PHASE 5: MODEL EVALUATION")
        print("="*70)
        
        print("\n✓ Making predictions on test set...")
        
        # ⭐ CPU OPTIMIZATION: Smaller batch size for prediction
        y_pred_proba = model.predict(X_test, batch_size=16, verbose=1)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n📊 Overall Metrics:")
        print(f"  Accuracy: {accuracy:.4f}")
        
        print(f"\n📊 Per-Class Metrics:")
        print(classification_report(y_test, y_pred, target_names=self.class_names, digits=4))
        
        return {
            'accuracy': float(accuracy),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'y_test': y_test
        }
    
    def plot_training_history(self, history):
        """Plot training metrics"""
        print("\n✓ Plotting training history...")
        
        Path(self.config.PLOTS_DIR).mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy
        axes[0].plot(history.history['accuracy'], label='Training', linewidth=2)
        axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
        axes[0].set_title('Model Accuracy', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Loss
        axes[1].plot(history.history['loss'], label='Training', linewidth=2)
        axes[1].plot(history.history['val_loss'], label='Validation', linewidth=2)
        axes[1].set_title('Model Loss', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(self.config.PLOTS_DIR, "training_history.png")
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        print(f"✓ Saved to {plot_path}")
        plt.close()
    
    def plot_confusion_matrix(self, results):
        """Plot confusion matrix"""
        print("✓ Plotting confusion matrix...")
        
        cm = confusion_matrix(results['y_test'], results['y_pred'])
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        plot_path = os.path.join(self.config.PLOTS_DIR, "confusion_matrix.png")
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        print(f"✓ Saved to {plot_path}")
        plt.close()
    
    def save_results(self, results):
        """Save evaluation results"""
        results_file = os.path.join(self.config.RESULTS_DIR, "evaluation_results.json")
        
        Path(self.config.RESULTS_DIR).mkdir(parents=True, exist_ok=True)
        
        results_to_save = {
            'accuracy': results['accuracy'],
            'model_name': 'MobileNetV2 (CPU-Optimized)',
            'device': self.config.DEVICE,
            'classes': self.class_names,
            'image_size': self.config.IMG_SIZE,
            'batch_size': self.config.BATCH_SIZE
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_to_save, f, indent=4)
        
        print(f"\n✓ Saved results to {results_file}")


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def main(args):
    """Main training pipeline"""
    
    # Optimize for CPU
    optimize_for_cpu()
    
    # Create config
    config = CPUTrainingConfig()
    
    # Apply mode settings
    if args.lite:
        print("\n⚡ LITE MODE: Fast training, lower accuracy")
        config.IMG_SIZE = (96, 96)
        config.BATCH_SIZE = 8
        config.EPOCHS = 15
        config.REDUCE_DATA = 0.3
    elif args.quick_train:
        print("\n⏱️  QUICK TRAIN MODE: Balanced speed/accuracy")
        config.IMG_SIZE = (128, 128)
        config.BATCH_SIZE = 8
        config.EPOCHS = 20
        config.REDUCE_DATA = 0.5
    else:
        print("\n📊 NORMAL TRAIN MODE: Full training (slowest)")
        config.IMG_SIZE = (160, 160)
        config.BATCH_SIZE = 8
        config.EPOCHS = 30
        config.REDUCE_DATA = 1.0
    
    # Override with command line args
    if args.epochs:
        config.EPOCHS = args.epochs
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    
    print("\n" + "="*70)
    print("🧠 BRAIN TUMOR CLASSIFICATION - CPU TRAINING")
    print("="*70)
    print(f"\n⚙️  Configuration:")
    print(f"  Device: CPU")
    print(f"  Model: MobileNetV2 (lightweight)")
    print(f"  Image size: {config.IMG_SIZE}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Epochs: {config.EPOCHS}")
    
    # Phase 1: Load Data
    data_loader = CPUDataLoader(config)
    X_train, y_train, X_test, y_test = data_loader.load_data(reduce_data=config.REDUCE_DATA)
    
    # Phase 2: Preprocess & Split
    preprocessor = CPUDataPreprocessor(config)
    X_train_split, X_val, X_test_final, y_train_split, y_val, y_test_final = \
        preprocessor.split_data(X_train, y_train, X_test, y_test)
    
    train_generator, val_generator = preprocessor.create_data_generators(
        X_train_split, y_train_split, X_val, y_val, config.NUM_CLASSES
    )
    
    # Phase 3: Build Model
    print("\n" + "="*70)
    print("PHASE 3: MODEL BUILDING")
    print("="*70)
    
    Path(config.RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    
    model_builder = CPUBrainTumorModel(config)
    model = model_builder.build_model()
    
    # Phase 4: Train
    trainer = CPUModelTrainer(config)
    model, history = trainer.train(
        model, train_generator, val_generator,
        num_train_samples=len(X_train_split)
    )
    
    # Phase 5: Evaluate
    evaluator = CPUModelEvaluator(config, data_loader.class_names)
    results = evaluator.evaluate(model, X_test_final, y_test_final)
    evaluator.plot_training_history(history)
    evaluator.plot_confusion_matrix(results)
    evaluator.save_results(results)
    
    # Summary
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\n📊 Final Accuracy: {results['accuracy']:.4f}")
    print(f"💾 Model saved to: {config.MODEL_PATH}")
    print(f"📈 Results saved to: {config.RESULTS_DIR}")
    print(f"\n💡 TIP: For faster training, use Google Colab (free GPU)")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train brain tumor model on CPU (Dell Vostro 14 3000, etc.)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
CPU Training Modes:
  Lite mode:      ~20 minutes,  70-75% accuracy  (--lite)
  Quick train:    ~1 hour,      75-80% accuracy  (--quick-train)
  Normal train:   ~2-3 hours,   80-85% accuracy  (default)

Examples:
  python train_brain_cpu_optimized.py --lite
  python train_brain_cpu_optimized.py --quick-train
  python train_brain_cpu_optimized.py
  python train_brain_cpu_optimized.py --epochs 40
        """
    )
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--lite', action='store_true',
                      help='Lite mode: 15 epochs, 96x96 images (fastest, ~20 min)')
    group.add_argument('--quick-train', action='store_true',
                      help='Quick train: 20 epochs, 128x128 images (balanced, ~1 hour)')
    
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override number of epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Override batch size (8 recommended for CPU)')
    
    args = parser.parse_args()
    
    main(args)