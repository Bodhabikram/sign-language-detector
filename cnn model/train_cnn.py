"""Train and evaluate the three-layer CNN on the processed ASL dataset."""

import argparse
import json
import os
import sys
import time

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

from model import build_cnn_model


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from config import DATASET_MANIFEST_PATH, PROCESSED_DIR

MANIFEST_PATH = DATASET_MANIFEST_PATH
OUTPUT_DIR = os.path.join(PROJECT_DIR, "models", "cnn")
SEED = 42


def load_manifest():
    if not os.path.isfile(MANIFEST_PATH):
        raise FileNotFoundError(
            f"Dataset manifest not found: {MANIFEST_PATH}. Run preprocess_data.py first."
        )
    with open(MANIFEST_PATH, "r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)

    class_names = manifest.get("classes")
    if (
        not isinstance(class_names, list)
        or len(class_names) < 2
        or any(not isinstance(name, str) or not name for name in class_names)
        or len(set(class_names)) != len(class_names)
    ):
        raise ValueError("Manifest must contain at least two unique class names.")

    image_size = manifest.get("image_size")
    if not (
        isinstance(image_size, list)
        and len(image_size) == 2
        and all(isinstance(value, int) and value > 0 for value in image_size)
    ):
        raise ValueError(f"Invalid image_size in manifest: {image_size}")
    return manifest, class_names, tuple(image_size)


def augment_image(image):
    height = tf.shape(image)[0]
    width = tf.shape(image)[1]
    padding = tf.cast(tf.round(tf.cast(height, tf.float32) * 0.06), tf.int32)
    padded = tf.pad(image, [[padding, padding], [padding, padding], [0, 0]], mode="REFLECT")
    offset_y = tf.random.uniform([], 0, 2 * padding + 1, dtype=tf.int32)
    offset_x = tf.random.uniform([], 0, 2 * padding + 1, dtype=tf.int32)
    image = tf.image.crop_to_bounding_box(padded, offset_y, offset_x, height, width)
    image = tf.image.random_brightness(image, 12.0)
    image = tf.image.random_contrast(image, 0.9, 1.1)
    return tf.clip_by_value(image, 0.0, 255.0)


def load_datasets(batch_size, image_size):
    manifest, class_names, _ = load_manifest()
    processed_root = os.path.abspath(PROCESSED_DIR)
    class_to_label = {name: index for index, name in enumerate(class_names)}
    datasets = []

    for split_name in ("train", "val"):
        records = manifest.get("splits", {}).get(split_name)
        if not isinstance(records, list) or not records:
            raise ValueError(f"The '{split_name}' split is missing or empty in the manifest.")

        image_paths = []
        labels = []
        for record in records:
            class_name = record.get("class_name")
            if class_name not in class_to_label:
                raise ValueError(f"Unknown class in '{split_name}' record: {class_name}")

            relative_path = record.get("image")
            if not isinstance(relative_path, str):
                raise ValueError(f"Missing image path in '{split_name}' record: {record}")
            absolute_path = os.path.abspath(os.path.join(processed_root, relative_path))
            if os.path.commonpath((processed_root, absolute_path)) != processed_root:
                raise ValueError(f"Image path escapes the processed dataset: {relative_path}")
            if not os.path.isfile(absolute_path):
                raise FileNotFoundError(f"Dataset image not found: {absolute_path}")

            image_paths.append(absolute_path)
            labels.append(class_to_label[class_name])

        is_train = split_name == "train"
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        if is_train:
            dataset = dataset.shuffle(
                buffer_size=len(image_paths),
                seed=SEED,
                reshuffle_each_iteration=True,
            )

        def decode_image(path, label, augment=is_train):
            image = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
            image = tf.image.resize(image, image_size)
            image = tf.cast(image, tf.float32)
            if augment:
                image = augment_image(image)
            return image, label

        dataset = dataset.map(decode_image, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        datasets.append(dataset)

    return tuple(datasets), class_names


def make_callbacks(output_dir=OUTPUT_DIR, model_ext=".keras"):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, f"cnn_best{model_ext}"),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=3,
            min_lr=1e-5,
            verbose=1,
        ),
    ]


def evaluate_and_save(best_model, val_ds, class_names, output_dir=OUTPUT_DIR):
    true_labels, pred_labels = [], []
    for images, labels in val_ds:
        probabilities = best_model(images, training=False).numpy()
        pred_labels.extend(np.argmax(probabilities, axis=1))
        true_labels.extend(labels.numpy())

    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average="weighted", zero_division=0)
    recall = recall_score(true_labels, pred_labels, average="weighted", zero_division=0)
    f1 = f1_score(true_labels, pred_labels, average="weighted", zero_division=0)

    report = classification_report(
        true_labels,
        pred_labels,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
    )
    print("\n" + "=" * 60)
    print("3-LAYER CNN + ANN VALIDATION")
    print("=" * 60)
    print(f"Accuracy : {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall   : {recall * 100:.2f}%")
    print(f"F1 score : {f1 * 100:.2f}%")
    print("\n" + report)

    with open(os.path.join(output_dir, "cnn_classification_report.txt"), "w", encoding="utf-8") as report_file:
        report_file.write(
            f"Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\n"
            f"Recall: {recall:.4f}\nF1: {f1:.4f}\n\n{report}"
        )

    matrix = confusion_matrix(true_labels, pred_labels, labels=list(range(len(class_names))))
    for name, data, title, fmt in (
        ("cnn_confusion_matrix.png", matrix, "3-layer CNN - confusion matrix (counts)", "d"),
        (
            "cnn_confusion_matrix_normalized.png",
            np.divide(
                matrix,
                matrix.sum(axis=1, keepdims=True),
                out=np.zeros_like(matrix, dtype=float),
                where=matrix.sum(axis=1, keepdims=True) != 0,
            ),
            "3-layer CNN - confusion matrix (share of each true class)",
            ".2f",
        ),
    ):
        fig, ax = plt.subplots(figsize=(16, 14))
        ConfusionMatrixDisplay(data, display_labels=class_names).plot(
            ax=ax, values_format=fmt, colorbar=True
        )
        ax.set_title(title)
        plt.setp(ax.get_xticklabels(), rotation=90)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, name), dpi=200, bbox_inches="tight")
        plt.close(fig)

    return accuracy, precision, recall, f1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--model-ext", choices=(".keras", ".h5"), default=".keras")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive integers.")

    manifest, class_names, image_size = load_manifest()
    os.makedirs(args.output_dir, exist_ok=True)
    tf.keras.utils.set_random_seed(SEED)
    (train_ds, val_ds), class_names = load_datasets(args.batch_size, image_size)
    print(f"Image size: {image_size} | Classes: {len(class_names)}")
    model = build_cnn_model(num_classes=len(class_names), input_shape=(*image_size, 3))
    model.summary()

    labels = {str(index): name for index, name in enumerate(class_names)}
    with open(os.path.join(args.output_dir, "labels.json"), "w", encoding="utf-8") as label_file:
        json.dump(labels, label_file, indent=2)

    start_time = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=make_callbacks(args.output_dir, args.model_ext),
    )

    best_model_path = os.path.join(args.output_dir, f"cnn_best{args.model_ext}")
    best_model = tf.keras.models.load_model(best_model_path)
    best_model.save(os.path.join(args.output_dir, f"cnn_final{args.model_ext}"))
    with open(os.path.join(args.output_dir, "training_history.json"), "w", encoding="utf-8") as history_file:
        json.dump(history.history, history_file, indent=2)

    val_loss, val_accuracy = best_model.evaluate(val_ds, verbose=1)
    evaluate_and_save(best_model, val_ds, class_names, args.output_dir)

    print(f"\nTraining finished in {(time.time() - start_time) / 60:.1f} minutes.")
    print(f"Best CNN model and training artifacts saved to: {args.output_dir}")
    print(f"Validation loss: {val_loss:.4f}; validation accuracy: {val_accuracy:.4f}")


if __name__ == "__main__":
    main()
