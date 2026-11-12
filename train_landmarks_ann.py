"""Train the Landmarks + ANN classifier from the manifest-aligned landmark dataset."""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from config import MODELS_DIR
from landmark_features import FEATURE_DIM, augment_landmarks, normalize_landmarks

DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "dataset", "landmarks_manifest"
)
LABELS_PATH = os.path.join(MODELS_DIR, "landmarks_labels.json")
MODEL_PATH = os.path.join(MODELS_DIR, "landmarks_ann.keras")
BEST_PATH = os.path.join(MODELS_DIR, "landmarks_ann_best.keras")
HISTORY_PATH = os.path.join(MODELS_DIR, "landmarks_training_history.json")
REPORT_PATH = os.path.join(MODELS_DIR, "landmarks_classification_report.txt")
CM_PATH = os.path.join(MODELS_DIR, "landmarks_confusion_matrix.png")
CM_NORM_PATH = os.path.join(MODELS_DIR, "landmarks_confusion_matrix_normalized.png")


def load_split(path, label_to_index):
    with open(path, "r", encoding="utf-8") as split_file:
        records = json.load(split_file)
    features = np.stack([normalize_landmarks(record["landmarks"]) for record in records])
    labels = np.array(
        [label_to_index[record["label"]] for record in records], dtype=np.int32
    )
    return features, labels


def make_augmented(features, labels, copies, seed=42):
    rng = np.random.default_rng(seed)
    feature_parts = [features]
    label_parts = [labels]
    for _ in range(copies):
        feature_parts.append(
            np.stack([augment_landmarks(row, rng) for row in features])
        )
        label_parts.append(labels)
    return np.concatenate(feature_parts), np.concatenate(label_parts)


def build_model(num_classes):
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(FEATURE_DIM,), name="landmarks"),
            tf.keras.layers.Dense(256, activation="relu", name="ann_dense_1"),
            tf.keras.layers.BatchNormalization(name="ann_bn_1"),
            tf.keras.layers.Dropout(0.35, name="ann_dropout_1"),
            tf.keras.layers.Dense(128, activation="relu", name="ann_dense_2"),
            tf.keras.layers.Dropout(0.2, name="ann_dropout_2"),
            tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions"),
        ],
        name="Landmarks_ANN",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_confusion_matrixes(matrix, class_names):
    display = ConfusionMatrixDisplay(matrix, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(16, 14))
    display.plot(ax=ax, xticks_rotation=90, values_format="d", colorbar=True)
    ax.set_title("Landmarks + ANN - confusion matrix (counts)")
    fig.tight_layout()
    fig.savefig(CM_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)

    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix, dtype=float),
        where=row_sums != 0,
    )
    fig, ax = plt.subplots(figsize=(16, 14))
    ConfusionMatrixDisplay(normalized, display_labels=class_names).plot(
        ax=ax, xticks_rotation=90, values_format=".2f", colorbar=True
    )
    ax.set_title("Landmarks + ANN - confusion matrix (share of each true class)")
    fig.tight_layout()
    fig.savefig(CM_NORM_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--augment-copies", type=int, default=3)
    args = parser.parse_args()

    if not os.path.isfile(LABELS_PATH):
        raise FileNotFoundError(
            f"Landmark labels not found at {LABELS_PATH}. Run prepare_landmarks_dataset.py first."
        )

    with open(LABELS_PATH, "r", encoding="utf-8") as labels_file:
        label_mapping = json.load(labels_file)
    class_names = [label_mapping[str(index)] for index in range(len(label_mapping))]
    label_to_index = {name: index for index, name in enumerate(class_names)}

    train_features, train_labels = load_split(
        os.path.join(args.data_dir, "train.json"), label_to_index
    )
    val_features, val_labels = load_split(
        os.path.join(args.data_dir, "val.json"), label_to_index
    )
    print(f"Train samples: {len(train_features)} | Validation samples: {len(val_features)}")
    print(f"Classes: {len(class_names)}")

    if args.augment_copies > 0:
        train_features, train_labels = make_augmented(
            train_features, train_labels, args.augment_copies
        )
        print(f"Augmented training samples: {len(train_features)}")

    num_classes = len(class_names)
    train_y = tf.keras.utils.to_categorical(train_labels, num_classes)
    val_y = tf.keras.utils.to_categorical(val_labels, num_classes)

    tf.keras.utils.set_random_seed(42)
    model = build_model(num_classes)
    model.summary()

    os.makedirs(MODELS_DIR, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            BEST_PATH, monitor="val_accuracy", mode="max", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=12,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5, verbose=1
        ),
    ]

    history = model.fit(
        train_features,
        train_y,
        validation_data=(val_features, val_y),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    model.save(MODEL_PATH)
    with open(HISTORY_PATH, "w", encoding="utf-8") as history_file:
        json.dump(history.history, history_file, indent=2)

    probabilities = model.predict(val_features, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    accuracy = accuracy_score(val_labels, predictions)
    precision = precision_score(val_labels, predictions, average="weighted", zero_division=0)
    recall = recall_score(val_labels, predictions, average="weighted", zero_division=0)
    f1 = f1_score(val_labels, predictions, average="weighted", zero_division=0)

    report = classification_report(
        val_labels,
        predictions,
        labels=list(range(num_classes)),
        target_names=class_names,
        zero_division=0,
    )
    print("\n" + "=" * 60)
    print("LANDMARKS + ANN VALIDATION")
    print("=" * 60)
    print(f"Accuracy : {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall   : {recall * 100:.2f}%")
    print(f"F1 score : {f1 * 100:.2f}%")
    print("\n" + report)

    with open(REPORT_PATH, "w", encoding="utf-8") as report_file:
        report_file.write(
            f"Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\n"
            f"Recall: {recall:.4f}\nF1: {f1:.4f}\n\n{report}"
        )

    matrix = confusion_matrix(val_labels, predictions, labels=list(range(num_classes)))
    save_confusion_matrixes(matrix, class_names)

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
