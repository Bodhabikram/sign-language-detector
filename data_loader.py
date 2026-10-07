"""Load the JSON-manifest dataset produced by preprocess_data.py."""

import json
import os

import tensorflow as tf

from config import (
    BATCH_SIZE,
    CHANNELS,
    DATASET_MANIFEST_PATH,
    IMG_SIZE,
    PROCESSED_DIR,
)


def load_dataset_manifest():
    """Read and validate the processed dataset manifest."""
    if not os.path.isfile(DATASET_MANIFEST_PATH):
        raise FileNotFoundError(
            f"Dataset manifest not found at '{DATASET_MANIFEST_PATH}'. "
            "Run preprocess_data.py first."
        )

    with open(DATASET_MANIFEST_PATH, "r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)

    classes = manifest.get("classes")
    if not isinstance(classes, list) or len(classes) < 2:
        raise ValueError("The dataset manifest must contain at least two classes.")
    if manifest.get("num_classes") != len(classes):
        raise ValueError("Manifest num_classes does not match its classes list.")

    image_size = manifest.get("image_size")
    if image_size != [IMG_SIZE, IMG_SIZE]:
        raise ValueError(
            f"Manifest image_size {image_size} does not match configured size "
            f"[{IMG_SIZE}, {IMG_SIZE}]."
        )

    return manifest


def load_class_names():
    """Return class names in the label-index order used during training."""
    return load_dataset_manifest()["classes"]


def get_datasets(batch_size=BATCH_SIZE):
    """Build batched TensorFlow datasets from the train/val/test JSON records."""
    manifest = load_dataset_manifest()
    classes = manifest["classes"]
    num_classes = len(classes)
    processed_root = os.path.abspath(PROCESSED_DIR)
    datasets = []

    for split_name in ("train", "val", "test"):
        records = manifest.get("splits", {}).get(split_name)
        if not isinstance(records, list) or not records:
            raise ValueError(f"The '{split_name}' split is missing or empty in the manifest.")

        image_paths = []
        labels = []
        for record in records:
            label = record.get("label")
            class_name = record.get("class_name")
            image_path = record.get("image")
            if not isinstance(label, int) or not 0 <= label < num_classes:
                raise ValueError(f"Invalid label in '{split_name}' record: {record}")
            if classes[label] != class_name:
                raise ValueError(f"Class/label mismatch in '{split_name}' record: {record}")
            if not isinstance(image_path, str):
                raise ValueError(f"Missing image path in '{split_name}' record: {record}")

            absolute_path = os.path.abspath(os.path.join(processed_root, image_path))
            if os.path.commonpath((processed_root, absolute_path)) != processed_root:
                raise ValueError(f"Image path escapes the processed dataset: {image_path}")
            image_paths.append(absolute_path)
            labels.append(label)

        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        if split_name == "train":
            dataset = dataset.shuffle(
                buffer_size=len(image_paths),
                seed=42,
                reshuffle_each_iteration=True,
            )

        def decode_image(path, label):
            image_bytes = tf.io.read_file(path)
            image = tf.io.decode_jpeg(image_bytes, channels=CHANNELS)
            image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
            image = tf.cast(image, tf.float32)
            label = tf.one_hot(label, depth=num_classes)
            return image, label

        dataset = dataset.map(decode_image, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        datasets.append(dataset)

    return tuple(datasets)
