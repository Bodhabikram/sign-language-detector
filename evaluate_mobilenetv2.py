import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from data_loader import get_datasets


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/mobilenetv2_stage1_final.keras"

NUM_CLASSES = 28

# Use the SAME class order as preprocessing
CLASS_NAMES = [
    "A", "B", "C", "D", "E", "F", "G",
    "H", "I", "J", "K", "L", "M", "N",
    "O", "P", "Q", "R", "S", "T", "U",
    "V", "W", "X", "Y", "Z",
    "nothing",
    "space"
]


# ============================================================
# CHECK CLASS COUNT
# ============================================================

print("=" * 60)
print("MOBILENETV2 MODEL EVALUATION")
print("=" * 60)

print(f"\nNumber of classes: {len(CLASS_NAMES)}")

if len(CLASS_NAMES) != NUM_CLASSES:
    raise ValueError(
        f"Expected {NUM_CLASSES} classes, "
        f"but found {len(CLASS_NAMES)}."
    )


# ============================================================
# LOAD DATASETS
# ============================================================

print("\nLoading datasets...")

train_ds, val_ds, test_ds = get_datasets()

print("Datasets loaded successfully.")


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("\nLoading trained MobileNetV2 model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")

print("\nModel input shape:")
print(model.input_shape)

print("\nModel output shape:")
print(model.output_shape)


# ============================================================
# EVALUATE ON TEST DATA
# ============================================================

print("\n" + "=" * 60)
print("EVALUATING ON TEST DATA")
print("=" * 60)

test_loss, test_accuracy = model.evaluate(
    test_ds,
    verbose=1
)

print("\n" + "=" * 60)
print("TEST RESULTS")
print("=" * 60)

print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy:.4f}")
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

predictions = model.predict(
    test_ds,
    verbose=1
)

predicted_labels = np.argmax(
    predictions,
    axis=1
)


# ============================================================
# GET TRUE LABELS
# ============================================================

true_labels = []

for images, labels in test_ds:

    # Labels are one-hot encoded
    labels = labels.numpy()

    batch_labels = np.argmax(
        labels,
        axis=1
    )

    true_labels.extend(
        batch_labels
    )

true_labels = np.array(true_labels)


# ============================================================
# CHECK PREDICTION COUNT
# ============================================================

print("\nNumber of test samples:")
print(len(true_labels))

print("\nNumber of predictions:")
print(len(predicted_labels))


if len(true_labels) != len(predicted_labels):

    raise ValueError(
        "Number of labels and predictions do not match."
    )


# ============================================================
# CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(
    true_labels,
    predicted_labels
)

precision = precision_score(
    true_labels,
    predicted_labels,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    true_labels,
    predicted_labels,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    true_labels,
    predicted_labels,
    average="weighted",
    zero_division=0
)


# ============================================================
# PRINT METRICS
# ============================================================

print("\n" + "=" * 60)
print("PERFORMANCE METRICS")
print("=" * 60)

print(f"\nAccuracy  : {accuracy:.4f}")
print(f"Accuracy  : {accuracy * 100:.2f}%")

print(f"\nPrecision : {precision:.4f}")
print(f"Precision : {precision * 100:.2f}%")

print(f"\nRecall    : {recall:.4f}")
print(f"Recall    : {recall * 100:.2f}%")

print(f"\nF1 Score  : {f1:.4f}")
print(f"F1 Score  : {f1 * 100:.2f}%")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

report = classification_report(
    true_labels,
    predicted_labels,
    labels=list(range(NUM_CLASSES)),
    target_names=CLASS_NAMES,
    zero_division=0
)

print(report)


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

REPORT_PATH = (
    "models/mobilenetv2_classification_report.txt"
)

with open(REPORT_PATH, "w") as file:

    file.write(
        "MobileNetV2 Classification Report\n"
    )

    file.write("=" * 60 + "\n\n")

    file.write(
        f"Test Loss: {test_loss:.4f}\n"
    )

    file.write(
        f"Test Accuracy: {accuracy:.4f}\n"
    )

    file.write(
        f"Precision: {precision:.4f}\n"
    )

    file.write(
        f"Recall: {recall:.4f}\n"
    )

    file.write(
        f"F1 Score: {f1:.4f}\n\n"
    )

    file.write(report)


print(
    f"\nClassification report saved to:\n"
    f"{REPORT_PATH}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\nGenerating confusion matrix...")

cm = confusion_matrix(
    true_labels,
    predicted_labels,
    labels=list(range(NUM_CLASSES))
)

fig, ax = plt.subplots(
    figsize=(16, 14)
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES
)

display.plot(
    ax=ax,
    xticks_rotation=90,
    values_format="d"
)

plt.title(
    "MobileNetV2 - Sign Language Confusion Matrix"
)

plt.tight_layout()


CONFUSION_PATH = (
    "models/mobilenetv2_confusion_matrix.png"
)

plt.savefig(
    CONFUSION_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    f"\nConfusion matrix saved to:\n"
    f"{CONFUSION_PATH}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)

print(f"\nTest Accuracy : {accuracy * 100:.2f}%")
print(f"Precision     : {precision * 100:.2f}%")
print(f"Recall        : {recall * 100:.2f}%")
print(f"F1 Score      : {f1 * 100:.2f}%")

print("\nGenerated files:")
print("- models/mobilenetv2_classification_report.txt")
print("- models/mobilenetv2_confusion_matrix.png")

print("\nDone!")