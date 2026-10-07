"""
Evaluate the trained MobileNetV2 sign language model on the test set.

Fixes compared with the original:
  - predictions and true labels are collected in ONE pass (no shuffle mismatch)
  - output folder is created automatically
  - plots are saved without blocking (set SHOW_PLOTS = True to display them)
  - normalised confusion matrix, most-confused pairs, optional training curves
"""
import os
import json
import numpy as np
import tensorflow as tf

import matplotlib
SHOW_PLOTS = False
if not SHOW_PLOTS:
    matplotlib.use("Agg")           # works without a display (WSL, servers)
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score, f1_score,
)

from config import MODEL_PATH
from data_loader import get_datasets, load_class_names

OUT_DIR = os.path.dirname(os.path.abspath(MODEL_PATH))
os.makedirs(OUT_DIR, exist_ok=True)
REPORT_PATH = os.path.join(OUT_DIR, "mobilenetv2_classification_report.txt")
CM_PATH = os.path.join(OUT_DIR, "mobilenetv2_confusion_matrix.png")
CM_NORM_PATH = os.path.join(OUT_DIR, "mobilenetv2_confusion_matrix_normalized.png")
CURVES_PATH = os.path.join(OUT_DIR, "mobilenetv2_training_curves.png")
HISTORY_PATH = os.path.join(OUT_DIR, "training_history.json")   # written by train_model.py

print("=" * 60)
print("MOBILENETV2 MODEL EVALUATION")
print("=" * 60)

# ---------------- LOAD DATA AND MODEL ----------------
_, _, test_ds = get_datasets()
CLASS_NAMES = list(load_class_names())
NUM_CLASSES = len(CLASS_NAMES)
print(f"Classes: {NUM_CLASSES}")

model = tf.keras.models.load_model(MODEL_PATH)
print(f"Model input : {model.input_shape}")
print(f"Model output: {model.output_shape}")

if model.output_shape[-1] != NUM_CLASSES:
    raise ValueError(
        f"Model has {model.output_shape[-1]} outputs but the dataset has {NUM_CLASSES} "
        "classes. Retrain the model with train_model.py."
    )

# ---------------- OFFICIAL KERAS EVALUATION ----------------
test_loss, test_acc = model.evaluate(test_ds, verbose=1)
print(f"\nTest loss    : {test_loss:.4f}")
print(f"Test accuracy: {test_acc * 100:.2f}%")

# ---------------- PREDICTIONS (single pass, order always matches) ----------------
true_labels, pred_labels = [], []
for images, labels in test_ds:
    probs = model(images, training=False).numpy()
    labels = labels.numpy()
    if labels.ndim > 1:                      # one-hot -> index
        labels = np.argmax(labels, axis=1)
    pred_labels.extend(np.argmax(probs, axis=1))
    true_labels.extend(labels)

true_labels = np.array(true_labels)
pred_labels = np.array(pred_labels)
print(f"\nTest samples: {len(true_labels)}")

# ---------------- METRICS ----------------
accuracy = accuracy_score(true_labels, pred_labels)
precision = precision_score(true_labels, pred_labels, average="weighted", zero_division=0)
recall = recall_score(true_labels, pred_labels, average="weighted", zero_division=0)
f1 = f1_score(true_labels, pred_labels, average="weighted", zero_division=0)

if abs(accuracy - test_acc) > 0.005:
    print(f"\n[WARNING] sklearn accuracy ({accuracy:.4f}) differs from Keras "
          f"accuracy ({test_acc:.4f}). Check the data pipeline for shuffling or "
          "different preprocessing.")

print("\n" + "=" * 60)
print("PERFORMANCE METRICS (weighted)")
print("=" * 60)
print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall   : {recall * 100:.2f}%")
print(f"F1 score : {f1 * 100:.2f}%")

report = classification_report(
    true_labels, pred_labels,
    labels=list(range(NUM_CLASSES)),
    target_names=CLASS_NAMES,
    zero_division=0,
)
print("\n" + report)

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("MobileNetV2 Classification Report\n" + "=" * 60 + "\n\n")
    f.write(f"Test loss: {test_loss:.4f}\n")
    f.write(f"Accuracy : {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall   : {recall:.4f}\n")
    f.write(f"F1 score : {f1:.4f}\n\n")
    f.write(report)
print(f"Report saved to: {REPORT_PATH}")

# ---------------- CONFUSION MATRICES ----------------
labels_idx = list(range(NUM_CLASSES))
cm = confusion_matrix(true_labels, pred_labels, labels=labels_idx)


def save_confusion_matrix(matrix, path, title, values_format):
    fig, ax = plt.subplots(figsize=(16, 14))
    ConfusionMatrixDisplay(matrix, display_labels=CLASS_NAMES).plot(
        ax=ax, xticks_rotation=90, values_format=values_format, colorbar=True)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)
    print(f"Saved: {path}")


save_confusion_matrix(cm, CM_PATH, "MobileNetV2 - confusion matrix (counts)", "d")

row_sums = cm.sum(axis=1, keepdims=True)
cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
save_confusion_matrix(cm_norm, CM_NORM_PATH,
                      "MobileNetV2 - confusion matrix (share of each true class)", ".2f")

# ---------------- MOST CONFUSED PAIRS ----------------
off_diag = cm.copy()
np.fill_diagonal(off_diag, 0)
pairs = [(off_diag[i, j], i, j) for i in range(NUM_CLASSES)
         for j in range(NUM_CLASSES) if off_diag[i, j] > 0]
pairs.sort(reverse=True)

print("\nMost confused pairs (true -> predicted):")
if not pairs:
    print("  none, no mistakes on the test set")
for count, i, j in pairs[:10]:
    print(f"  {CLASS_NAMES[i]:>8} -> {CLASS_NAMES[j]:<8} {count} times")

# ---------------- OPTIONAL: TRAINING CURVES ----------------
# In train_model.py, after training, add:
#     json.dump(history.history, open("models/training_history.json", "w"))
# (if you fine-tune, merge both histories' lists before saving)
if os.path.exists(HISTORY_PATH):
    with open(HISTORY_PATH) as f:
        hist = json.load(f)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key, name in zip(axes, ("accuracy", "loss"), ("Accuracy", "Loss")):
        if key in hist:
            ax.plot(hist[key], label="train")
        if f"val_{key}" in hist:
            ax.plot(hist[f"val_{key}"], label="validation")
        ax.set_title(name)
        ax.set_xlabel("Epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(CURVES_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {CURVES_PATH}")
else:
    print(f"\n[Info] No training history at {HISTORY_PATH}; skipping training curves.")

print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)