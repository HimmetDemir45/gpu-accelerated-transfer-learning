# %% LIBRARY IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import json
import subprocess


from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras import mixed_precision
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras.applications import ResNet50

from pathlib import Path
import os.path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# save as gpu_check.py

import sys

def check_system():
    print("="*40)
    print("STARTING SYSTEM CHECK")
    print("="*40)

    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("ERROR: GPU not found! Training with CPU will be very slow.")
        sys.exit(1) # Stop training

    print(f"{len(gpus)} GPU active.")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    print("Memory Growth activated.")
    print("="*40 + "\n")
    return True
# %% GPU SETTINGS AND CHECK
print("="*60)
print("GPU CHECK AND SETTINGS")
print("="*60)

# GPU check
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Memory growth setting (prevents OOM error)
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        # GPU info
        print(f"\n{len(gpus)} GPU found!")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu.name}")

        # CUDA info
        print(f"\nTensorFlow Version: {tf.__version__}")
        print(f"CUDA Support: {tf.test.is_built_with_cuda()}")
        print(f"GPU Device: {tf.test.gpu_device_name()}")

    except RuntimeError as e:
        print(f"Error while setting up GPU: {e}")
else:
    print("GPU not found! CPU will be used.")
    print("   To use GPU: pip install tensorflow[and-cuda]")

# Mixed Precision (2x speedup!)
mixed_precision.set_global_policy('mixed_float16')
print("\nMixed Precision (FP16) active - 2x speedup expected!")

print("="*60 + "\n")

# %% GPU TEMPERATURE MONITOR (Callback)
class GPUMonitor(Callback):
    """GPU temperature and usage tracking"""

    def __init__(self, check_every_n_batches=50):
        super().__init__()
        self.check_every_n_batches = check_every_n_batches
        self.batch_count = 0

    def on_batch_end(self, batch, logs=None):
        self.batch_count += 1
        if self.batch_count % self.check_every_n_batches == 0:
            try:
                # get temperature and usage with nvidia-smi
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=temperature.gpu,utilization.gpu,memory.used',
                     '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=2
                )
                if result.returncode == 0:
                    temp, util, mem = result.stdout.strip().split(',')
                    temp, util, mem = int(temp), int(util), int(mem)

                    # Warning check
                    if temp > 85:
                        print(f"\nGPU Temperature High: {temp}°C")
                    elif temp > 80:
                        print(f"\nGPU: {temp}°C | Usage: {util}% | VRAM: {mem}MB")
            except Exception:
                pass  # continue silently if nvidia-smi does not work

    def on_epoch_end(self, epoch, logs=None):
        """Detailed report at the end of each epoch"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                temp, util, mem_used, mem_total = result.stdout.strip().split(',')
                print(f"\nGPU Stats - Epoch {epoch+1}:")
                print(f"   Temperature: {temp}°C")
                print(f"   Usage: {util}%")
                print(f"   VRAM: {mem_used}/{mem_total} MB")
        except Exception:
            pass

# %% CONFIGURATION
CONFIG = {
    'dataset_path': "Dataset",
    'img_size': (224, 224),
    'batch_size': 64,  # Optimized for GPU (ideal for 8GB)
    'initial_epochs': 20,
    'additional_epochs': 30,
    'initial_lr': 1e-4,
    'checkpoint_path': "model_checkpoint.keras",
    'history_path': "training_history.json",
    'fine_tune_at': 100,
}

print("CONFIGURATION:")
print(f"  Batch Size: {CONFIG['batch_size']} (GPU optimized)")
print(f"  Image Size: {CONFIG['img_size']}")
print(f"  Initial Epochs: {CONFIG['initial_epochs']}")
print(f"  Learning Rate: {CONFIG['initial_lr']}")
print()

# %% DATA LOADING AND PREPARATION
print("="*60)
print("DATA LOADING")
print("="*60 + "\n")

dataset = CONFIG['dataset_path']
image_dir = Path(dataset)

filepaths = list(image_dir.glob(r"**/*.jpg")) + list(image_dir.glob(r"**/*.png"))
labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

filepaths = pd.Series(filepaths, name="filepath").astype("str")
labels = pd.Series(labels, name="labels").astype("str")
image_df = pd.concat([filepaths, labels], axis=1)

print(f"Total number of images: {len(image_df)}")
print(f"Number of classes: {len(image_df.labels.unique())}")
print(f"Classes: {image_df.labels.unique()}\n")

# %% DATA SPLITTING
train_df, test_df = train_test_split(
    image_df,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=labels
)

print(f"Train set: {len(train_df)} images")
print(f"Test set: {len(test_df)} images\n")

# %% DATA AUGMENTATION AND GENERATOR
print("="*60)
print("CREATING DATA GENERATOR")
print("="*60 + "\n")

train_generator = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_generator = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input
)

train_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col="filepath",
    y_col="labels",
    target_size=CONFIG['img_size'],
    color_mode="rgb",
    class_mode="categorical",
    batch_size=CONFIG['batch_size'],
    shuffle=True,
    seed=42,
    subset="training"
)

val_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col="filepath",
    y_col="labels",
    target_size=CONFIG['img_size'],
    color_mode="rgb",
    class_mode="categorical",
    batch_size=CONFIG['batch_size'],
    shuffle=True,
    seed=42,
    subset="validation"
)

test_images = test_generator.flow_from_dataframe(
    dataframe=test_df,
    x_col="filepath",
    y_col="labels",
    target_size=CONFIG['img_size'],
    color_mode="rgb",
    class_mode="categorical",
    batch_size=CONFIG['batch_size'],
    shuffle=False
)

print(f"Training batches: {len(train_images)}")
print(f"Validation batches: {len(val_images)}")
print(f"Test batches: {len(test_images)}\n")

# %% MODEL CREATION FUNCTION
def create_model(num_classes=10, fine_tune=False):
    """Model creation function"""

    pretrained_model = ResNet50(
        input_shape=(*CONFIG['img_size'], 3),
        include_top=False,
        weights="imagenet",
        pooling=None
    )

    if fine_tune:
        pretrained_model.trainable = True
        for layer in pretrained_model.layers[:-CONFIG['fine_tune_at']]:
            layer.trainable = False
    else:
        pretrained_model.trainable = False

    # Model architecture
    inputs = pretrained_model.input
    x = pretrained_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax", dtype='float32')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

# %% CALLBACK FUNCTIONS
checkpoint_callback = ModelCheckpoint(
    CONFIG['checkpoint_path'],
    save_best_only=True,
    monitor="val_accuracy",
    mode='max',
    verbose=1
)

early_stopping = EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

# Add GPU monitor
gpu_monitor = GPUMonitor(check_every_n_batches=50)

# %% HISTORY SAVING FUNCTIONS
def save_history(history, path):
    """Save training history as JSON"""
    hist_dict = history.history
    for key in hist_dict.keys():
        hist_dict[key] = [float(x) for x in hist_dict[key]]

    with open(path, 'w') as f:
        json.dump(hist_dict, f)

def load_history(path):
    """Load training history"""
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None

# %% MODEL TRAINING
print("\n" + "="*60)
print("MODEL TRAINING")
print("="*60 + "\n")

# Checkpoint check
if os.path.exists(CONFIG['checkpoint_path']):
    print(f"Checkpoint found: {CONFIG['checkpoint_path']}")
    print("Loading model...")
    model = tf.keras.models.load_model(CONFIG['checkpoint_path'])
    print("Model loaded successfully!")

    old_history = load_history(CONFIG['history_path'])
    initial_epoch = len(old_history['loss']) if old_history else 0
    print(f"Training will resume from epoch {initial_epoch}\n")
else:
    print("Checkpoint not found. Creating a new model...")
    model = create_model(num_classes=len(train_images.class_indices))
    model.compile(
        optimizer=Adam(CONFIG['initial_lr']),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    initial_epoch = 0
    old_history = None
    print("New model created!\n")

# Model summary
print("\n" + "="*60)
print("MODEL ARCHITECTURE")
print("="*60)
model.summary()

# %% GPU ACCELERATION TEST (UPDATED)
print("\n" + "="*60)
print("GPU ACCELERATION TEST")
print("="*60 + "\n")

print("Running a short test...")
import time

# Use a small batch for the test (Just to see the speed difference)
test_batch_size = 8

# CPU test
print("   Running CPU Test (Be patient)...")
with tf.device('/CPU:0'):
    start = time.time()
    # We reduced the batch size: 8 instead of 64
    x_test = tf.random.normal([test_batch_size, 224, 224, 3])
    # We reduced the loop: 1 instead of 10
    for _ in range(1):
        _ = model(x_test, training=False)
    cpu_time = time.time() - start

# GPU test
if gpus:
    print("   Running GPU Test...")
    with tf.device('/GPU:0'):
        start = time.time()
        x_test = tf.random.normal([test_batch_size, 224, 224, 3])
        # Since the GPU is very fast, we can run it more to be fair, but it's not necessary
        for _ in range(1):
            _ = model(x_test, training=False)
        gpu_time = time.time() - start

    speedup = cpu_time / gpu_time
    print(f"\nTest Results (1 Batch, {test_batch_size} Image):")
    print(f"   CPU: {cpu_time:.3f} seconds")
    print(f"   GPU: {gpu_time:.3f} seconds")
    print(f"   Speedup: {speedup:.1f}x times faster!\n")
else:
    print("GPU not found.\n")

# %% START TRAINING
print("="*60)
print(f"PHASE 1: TRANSFER LEARNING")
print("="*60)
print(f"Total Epochs: {CONFIG['initial_epochs']}")
print(f"Starting Epoch: {initial_epoch}")
print(f"Batch Size: {CONFIG['batch_size']}")
print(f"Steps per Epoch: {len(train_images)}")
print("="*60 + "\n")

# Training start time
training_start = time.time()

history = model.fit(
    train_images,
    steps_per_epoch=len(train_images),
    validation_data=val_images,
    validation_steps=len(val_images),
    epochs=CONFIG['initial_epochs'],
    initial_epoch=initial_epoch,
    callbacks=[checkpoint_callback, early_stopping, reduce_lr, gpu_monitor],
    verbose=1
)

training_time = time.time() - training_start

# Training time report
print("\n" + "="*60)
print("TRAINING COMPLETED")
print("="*60)
print(f"Total Time: {training_time/60:.2f} minutes ({training_time:.1f} seconds)")
print(f"Average per Epoch: {training_time/len(history.history['loss']):.1f} seconds")
print("="*60 + "\n")

# Save history
save_history(history, CONFIG['history_path'])
print(f"Training history saved: {CONFIG['history_path']}\n")

# %% FINE-TUNING PHASE
print("="*60)
print("PHASE 2: FINE-TUNING (OPTIONAL)")
print("="*60 + "\n")

response = input("Do you want to perform fine-tuning? (y/n): ")

if response.lower() == 'y':
    print("\nStarting fine-tuning...\n")

    model = create_model(num_classes=len(train_images.class_indices), fine_tune=True)
    model.load_weights(CONFIG['checkpoint_path'])

    model.compile(
        optimizer=Adam(CONFIG['initial_lr'] / 10),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    fine_tune_checkpoint = CONFIG['checkpoint_path'].replace('.keras', '_finetuned.keras')
    checkpoint_callback_ft = ModelCheckpoint(
        fine_tune_checkpoint,
        save_best_only=True,
        monitor="val_accuracy",
        mode='max',
        verbose=1
    )

    current_epoch = len(history.history['loss'])
    total_epochs = current_epoch + CONFIG['additional_epochs']

    print(f"Fine-tuning: Epoch {current_epoch} -> {total_epochs}\n")

    ft_start = time.time()

    history_fine = model.fit(
        train_images,
        steps_per_epoch=len(train_images),
        validation_data=val_images,
        validation_steps=len(val_images),
        epochs=total_epochs,
        initial_epoch=current_epoch,
        callbacks=[checkpoint_callback_ft, early_stopping, reduce_lr, gpu_monitor],
        verbose=1
    )

    ft_time = time.time() - ft_start

    print(f"\nFine-tuning Time: {ft_time/60:.2f} minutes\n")

    fine_tune_history_path = CONFIG['history_path'].replace('.json', '_finetuned.json')
    save_history(history_fine, fine_tune_history_path)
    print(f"Fine-tuning history saved: {fine_tune_history_path}\n")
    """100/100 [==============================] - 52s 522ms/step - loss: 0.0109 - accuracy: 0.9964 - val_loss: 0.0399 - val_accuracy: 0.9869 - lr: 1.2500e-06
Epoch 41: early stopping
Fine-tuning Time: 19.06 minutes"""

# %% MODEL EVALUATION
print("="*60)
print("MODEL EVALUATION")
print("="*60 + "\n")

best_model = tf.keras.models.load_model(CONFIG['checkpoint_path'])
loss, accuracy = best_model.evaluate(test_images, verbose=1)

print(f"\nTEST RESULTS:")
print(f"   Loss: {loss:.4f}")
print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print("="*60 + "\n")

# %% HISTORY VISUALIZATION
def plot_history(history_path, title_prefix=""):
    """Visualize training history"""
    history_dict = load_history(history_path)
    if history_dict is None:
        print(f"History file not found: {history_path}")
        return

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history_dict["accuracy"], marker="o", label="Training Accuracy", linewidth=2)
    plt.plot(history_dict["val_accuracy"], marker="o", label="Validation Accuracy", linewidth=2)
    plt.title(f"{title_prefix}Model Accuracy", fontsize=14, fontweight='bold')
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history_dict["loss"], marker="o", label="Training Loss", linewidth=2)
    plt.plot(history_dict["val_loss"], marker="o", label="Validation Loss", linewidth=2)
    plt.title(f"{title_prefix}Model Loss", fontsize=14, fontweight='bold')
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()

print("Visualizing training history...\n")
plot_history(CONFIG['history_path'], "Transfer Learning - ")

# %% PREDICTION AND METRICS
print("="*60)
print("PREDICTION AND METRIC CALCULATION")
print("="*60 + "\n")

print("Predicting on the test set...")
pred = best_model.predict(test_images, verbose=1)
pred_classes = np.argmax(pred, axis=1)

labels_dict = train_images.class_indices
labels_dict = dict((v, k) for k, v in labels_dict.items())
pred_labels = [labels_dict[k] for k in pred_classes]

y_true = test_df.labels.values

print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
print(classification_report(y_true, pred_labels))
"""============================================================
CLASSIFICATION REPORT
============================================================
              precision    recall  f1-score   support
      Alaxan       0.83      0.93      0.87       200
    Bactidol       0.89      0.83      0.86       200
      Bioflu       0.93      0.89      0.91       200
    Biogesic       0.92      0.80      0.85       200
     DayZinc       0.92      0.95      0.94       200
    Decolgen       0.94      0.92      0.93       200
    Fish Oil       0.86      0.95      0.91       200
     Kremil S       0.85      0.93      0.89       200
     Medicol       0.99      0.97      0.98       200
      Neozep       0.88      0.82      0.85       200
    accuracy                           0.90      2000
   macro avg       0.90      0.90      0.90      2000
weighted avg       0.90      0.90      0.90      2000
"""
# %% CONFUSION MATRIX
print("Creating Confusion Matrix...\n")
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_true, pred_labels, labels=list(labels_dict.values()))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels_dict.values(),
            yticklabels=labels_dict.values())
plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# %% RANDOM PREDICTIONS
print("Visualizing random predictions...\n")
random_index = np.random.randint(0, len(test_df), min(15, len(test_df)))
fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(12, 15))

for i, ax in enumerate(axes.flat):
    if i < len(random_index):
        img_path = test_df.iloc[random_index[i]].filepath
        ax.imshow(plt.imread(img_path))

        true_label = test_df.iloc[random_index[i]].labels
        pred_label = pred_labels[random_index[i]]
        confidence = pred[random_index[i]][pred_classes[random_index[i]]] * 100

        color = "green" if true_label == pred_label else "red"
        ax.set_title(f"True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%",
                     color=color, fontsize=9)
        ax.axis('off')
    else:
        ax.axis('off')

plt.tight_layout()
plt.show()

# %% SUMMARY REPORT
print("\n" + "="*60)
print("FINAL REPORT")
print("="*60)
print(f"Model: ResNet50 (Transfer Learning)")
print(f"Dataset: {len(image_df)} images, {len(labels_dict)} classes")
print(f"Train/Val/Test Split: {len(train_df)}/{len(val_images)*CONFIG['batch_size']}/{len(test_df)}")
print(f"Training Time: {training_time/60:.2f} minutes")
print(f"Final Test Accuracy: {accuracy*100:.2f}%")
print(f"Model Saved: {CONFIG['checkpoint_path']}")
print(f"GPU Used: {'Yes' if gpus else 'No'}")
if gpus:
    print(f"Mixed Precision: Active (FP16)")
    print(f"Batch Size: {CONFIG['batch_size']} (GPU optimized)")
print("="*60)

print("\nTraining completed successfully!")
print("All files saved.")
print("\nHint: If you run the model again, it will continue from where it left off!")
"""============================================================
FINAL REPORT
============================================================
Model: ResNet50 (Transfer Learning)
Dataset: 10000 images, 10 classes
Train/Val/Test Split: 8000/1600/2000
Training Time: 18.18 minutes
Final Test Accuracy: 89.90%
Model Saved: model_checkpoint.keras
GPU Used: Yes
Mixed Precision: Active (FP16)
Batch Size: 64 (GPU optimized)
============================================================
Training completed successfully!
All files saved.
Hint: If you run the model again, it will continue from where it left off!
"""