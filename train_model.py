import os
import time
import tensorflow as tf
import matplotlib.pyplot as plt

from data_loader import get_datasets
from model import build_model


# ============================================================
# CONFIGURATION
# ============================================================

EPOCHS = 20

MODEL_DIR = "models"
BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_mobilenetv2_stage1.keras"
)

FINAL_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "mobilenetv2_stage1_final.keras"
)

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# LOAD DATASETS
# ============================================================

print("=" * 60)
print("LOADING DATASETS")
print("=" * 60)

train_ds, val_ds, test_ds = get_datasets()

print("\nDatasets loaded successfully.")


# ============================================================
# BUILD MODEL
# ============================================================

print("\n" + "=" * 60)
print("BUILDING MOBILENETV2 MODEL")
print("=" * 60)

model = build_model()

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)


model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    BEST_MODEL_PATH,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1
)


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=2,
    min_lr=1e-6,
    verbose=1
)


# ============================================================
# TRAIN MODEL
# ============================================================

print("\n" + "=" * 60)
print("STARTING MOBILENETV2 TRAINING")
print("=" * 60)

start_time = time.time()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[
        early_stopping,
        model_checkpoint,
        reduce_lr
    ]
)

training_time = time.time() - start_time


# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(FINAL_MODEL_PATH)

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(f"\nTraining time: {training_time / 60:.2f} minutes")

print(f"\nBest model saved at:")
print(BEST_MODEL_PATH)

print(f"\nFinal model saved at:")
print(FINAL_MODEL_PATH)


# ============================================================
# TRAINING RESULTS
# ============================================================

train_accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]

train_loss = history.history["loss"]
val_loss = history.history["val_loss"]

best_epoch = (
    max(
        range(len(val_accuracy)),
        key=lambda i: val_accuracy[i]
    ) + 1
)

best_val_accuracy = max(val_accuracy)

print("\n" + "=" * 60)
print("TRAINING SUMMARY")
print("=" * 60)

print(f"\nBest epoch          : {best_epoch}")
print(f"Best validation acc : {best_val_accuracy:.4f}")


# ============================================================
# PLOT ACCURACY
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    train_accuracy,
    label="Training Accuracy"
)

plt.plot(
    val_accuracy,
    label="Validation Accuracy"
)

plt.title("MobileNetV2 Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "models/mobilenetv2_accuracy.png",
    dpi=150
)

plt.show()


# ============================================================
# PLOT LOSS
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    train_loss,
    label="Training Loss"
)

plt.plot(
    val_loss,
    label="Validation Loss"
)

plt.title("MobileNetV2 Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "models/mobilenetv2_loss.png",
    dpi=150
)

plt.show()