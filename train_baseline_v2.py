import os
import time
import json

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# CONFIGURATION
# ============================================================

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 20

DATASET_DIR = "dataset/processed"

# IMPORTANT:
# This is a NEW filename.
# Your original model remains untouched.
MODEL_PATH = "models/baseline_cnn_v2.keras"

HISTORY_PATH = "models/baseline_cnn_v2_history.json"

CONFUSION_MATRIX_PATH = "models/baseline_cnn_v2_confusion_matrix.png"

TRAINING_PLOT_PATH = "models/baseline_cnn_v2_training.png"

RANDOM_SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs("models", exist_ok=True)


# ============================================================
# PRINT ENVIRONMENT INFORMATION
# ============================================================

print("=" * 70)
print("SIGN LANGUAGE DETECTION - CNN BASELINE V2")
print("=" * 70)

print(f"TensorFlow version : {tf.__version__}")
print(f"Keras version      : {tf.keras.__version__}")
print(f"Image size         : {IMG_SIZE}x{IMG_SIZE}")
print(f"Batch size         : {BATCH_SIZE}")
print(f"Maximum epochs     : {EPOCHS}")
print(f"Dataset directory  : {DATASET_DIR}")
print(f"Model output       : {MODEL_PATH}")
print("=" * 70)


# ============================================================
# CHECK DATASET
# ============================================================

if not os.path.isdir(DATASET_DIR):
    raise FileNotFoundError(
        f"Dataset directory not found: {DATASET_DIR}"
    )


required_classes = ["A", "B", "C", "D"]

for class_name in required_classes:
    class_path = os.path.join(DATASET_DIR, class_name)

    if not os.path.isdir(class_path):
        raise FileNotFoundError(
            f"Required class directory not found: {class_path}"
        )


# ============================================================
# TRAINING DATA GENERATOR
# ============================================================
#
# IMPORTANT:
# Augmentation is applied ONLY to training images.
#
# Validation data will use a separate generator below
# without random augmentation.
# ============================================================

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.2,

    rotation_range=15,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.7, 1.3],

    horizontal_flip=False
)


# ============================================================
# VALIDATION DATA GENERATOR
# ============================================================
#
# Only normalization is applied.
# No random augmentation.
# ============================================================

val_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.2
)


# ============================================================
# CREATE TRAINING GENERATOR
# ============================================================

train_gen = train_datagen.flow_from_directory(
    DATASET_DIR,

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    subset="training",

    shuffle=True,

    seed=RANDOM_SEED
)


# ============================================================
# CREATE VALIDATION GENERATOR
# ============================================================

val_gen = val_datagen.flow_from_directory(
    DATASET_DIR,

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    subset="validation",

    shuffle=False
)


# ============================================================
# DATASET INFORMATION
# ============================================================

num_classes = len(train_gen.class_indices)

class_indices = train_gen.class_indices

class_labels = list(class_indices.keys())


print("\nClass mapping:")
for class_name, class_index in class_indices.items():
    print(f"  {class_name} -> {class_index}")

print(f"\nNumber of classes     : {num_classes}")
print(f"Training images       : {train_gen.samples}")
print(f"Validation images     : {val_gen.samples}")
print(f"Total images          : {train_gen.samples + val_gen.samples}")
print(f"Training batches      : {len(train_gen)}")
print(f"Validation batches    : {len(val_gen)}")


# ============================================================
# VERIFY CLASS COUNT
# ============================================================

if num_classes != 4:
    raise ValueError(
        f"Expected 4 classes (A, B, C, D), but found {num_classes}."
    )


# ============================================================
# BUILD CNN MODEL
# ============================================================
#
# This keeps the architecture of your original CNN baseline.
#
# Original architecture:
#
# 128x128x3
#      ↓
# Conv2D 32
# BatchNorm
# MaxPool
#      ↓
# Conv2D 64
# BatchNorm
# MaxPool
#      ↓
# Conv2D 128
# BatchNorm
# MaxPool
#      ↓
# Flatten
# Dense 128
# Dropout 0.5
# Dense 4 Softmax
#
# The Input() layer is used explicitly for Keras 3 compatibility.
# ============================================================

model = Sequential(
    [
        Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

        Conv2D(
            32,
            (3, 3),
            activation="relu"
        ),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(
            64,
            (3, 3),
            activation="relu"
        ),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(
            128,
            (3, 3),
            activation="relu"
        ),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Flatten(),

        Dense(
            128,
            activation="relu"
        ),

        Dropout(0.5),

        Dense(
            num_classes,
            activation="softmax"
        )
    ],
    name="baseline_cnn_v2"
)


# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="categorical_crossentropy",

    metrics=[
        "accuracy",

        tf.keras.metrics.Precision(
            name="precision"
        ),

        tf.keras.metrics.Recall(
            name="recall"
        )
    ]
)


# ============================================================
# DISPLAY MODEL
# ============================================================

print("\nModel architecture:")
model.summary()


# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),

    ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]


# ============================================================
# TRAIN MODEL
# ============================================================

print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)

training_start = time.time()

history = model.fit(
    train_gen,

    validation_data=val_gen,

    epochs=EPOCHS,

    callbacks=callbacks,

    verbose=1
)

training_time = time.time() - training_start

print("\nTraining completed.")

print(
    f"Total training time: "
    f"{training_time / 60:.2f} minutes"
)


# ============================================================
# LOAD BEST MODEL
# ============================================================
#
# ModelCheckpoint saves the best validation-accuracy model.
# Loading it ensures evaluation uses the best saved model.
# ============================================================

print("\nLoading best saved model...")

best_model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Best model loaded successfully.")


# ============================================================
# FINAL VALIDATION EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL VALIDATION EVALUATION")
print("=" * 70)

val_gen.reset()

evaluation_results = best_model.evaluate(
    val_gen,
    verbose=1,
    return_dict=True
)

print("\nValidation metrics:")

for metric_name, metric_value in evaluation_results.items():
    print(
        f"{metric_name:12s}: "
        f"{metric_value:.4f}"
    )


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating validation predictions...")

val_gen.reset()

y_pred_prob = best_model.predict(
    val_gen,
    verbose=1
)

y_pred = np.argmax(
    y_pred_prob,
    axis=1
)

y_true = val_gen.classes


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    y_true,
    y_pred,
    target_names=class_labels,
    digits=4
)

print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


fig, ax = plt.subplots(
    figsize=(7, 7)
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_labels
)

disp.plot(
    ax=ax,
    values_format="d"
)

ax.set_title(
    "Baseline CNN V2 - Validation Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    CONFUSION_MATRIX_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"\nConfusion matrix saved to:"
    f"\n{CONFUSION_MATRIX_PATH}"
)


# ============================================================
# TRAINING HISTORY
# ============================================================

history_data = {
    key: [float(value) for value in values]
    for key, values in history.history.items()
}

with open(
    HISTORY_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        history_data,
        f,
        indent=4
    )

print(
    f"Training history saved to:"
    f"\n{HISTORY_PATH}"
)


# ============================================================
# TRAINING CURVES
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(13, 5)
)


# Accuracy

axes[0].plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

axes[0].plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

axes[0].set_title(
    "Baseline CNN V2 - Accuracy"
)

axes[0].set_xlabel(
    "Epoch"
)

axes[0].set_ylabel(
    "Accuracy"
)

axes[0].legend()

axes[0].grid(
    True,
    alpha=0.3
)


# Loss

axes[1].plot(
    history.history["loss"],
    label="Training Loss"
)

axes[1].plot(
    history.history["val_loss"],
    label="Validation Loss"
)

axes[1].set_title(
    "Baseline CNN V2 - Loss"
)

axes[1].set_xlabel(
    "Epoch"
)

axes[1].set_ylabel(
    "Loss"
)

axes[1].legend()

axes[1].grid(
    True,
    alpha=0.3
)


plt.tight_layout()

plt.savefig(
    TRAINING_PLOT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Training curves saved to:"
    f"\n{TRAINING_PLOT_PATH}"
)


# ============================================================
# INFERENCE LATENCY TEST
# ============================================================

print("\n" + "=" * 70)
print("INFERENCE LATENCY TEST")
print("=" * 70)

dummy_input = np.random.rand(
    1,
    IMG_SIZE,
    IMG_SIZE,
    3
).astype("float32")


# Warm-up inference
for _ in range(10):

    best_model.predict(
        dummy_input,
        verbose=0
    )


num_latency_runs = 100

start_time = time.perf_counter()

for _ in range(num_latency_runs):

    best_model.predict(
        dummy_input,
        verbose=0
    )

end_time = time.perf_counter()


total_latency = end_time - start_time

avg_latency_ms = (
    total_latency /
    num_latency_runs
) * 1000

potential_fps = (
    1000 /
    avg_latency_ms
)


print(
    f"Average inference latency : "
    f"{avg_latency_ms:.2f} ms"
)

print(
    f"Potential inference FPS   : "
    f"{potential_fps:.2f}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("BASELINE CNN V2 COMPLETE")
print("=" * 70)

print(
    f"Model saved to:"
    f"\n  {MODEL_PATH}"
)

print(
    f"\nTraining images:"
    f" {train_gen.samples}"
)

print(
    f"Validation images:"
    f" {val_gen.samples}"
)

print(
    f"\nBest validation accuracy:"
    f" {max(history.history['val_accuracy']):.4f}"
)

print(
    f"\nAverage inference latency:"
    f" {avg_latency_ms:.2f} ms"
)

print(
    f"\nPotential FPS:"
    f" {potential_fps:.2f}"
)

print("\nFiles generated:")

print(
    f"  {MODEL_PATH}"
)

print(
    f"  {HISTORY_PATH}"
)

print(
    f"  {CONFUSION_MATRIX_PATH}"
)

print(
    f"  {TRAINING_PLOT_PATH}"
)

print("=" * 70)