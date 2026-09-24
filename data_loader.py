import os
import pandas as pd
import tensorflow as tf

# CONFIGURATION

PROCESSED_DATASET = "dataset/processed"

TRAIN_CSV = "dataset/train.csv"
VAL_CSV = "dataset/val.csv"
TEST_CSV = "dataset/test.csv"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
NUM_CLASSES = 28

AUTOTUNE = tf.data.AUTOTUNE


# LOAD CSV

def load_csv(csv_path):
    """
    Reads the CSV file and returns:
    - full image paths
    - integer class labels
    """

    df = pd.read_csv(csv_path)

    full_paths = [
        os.path.join(PROCESSED_DATASET, rel_path)
        for rel_path in df["filepath"]
    ]

    labels = df["class_index"].tolist()

    return full_paths, labels


# IMAGE LOADING + DECODING

def load_and_decode_image(filepath, label):
    """
    Loads an image, converts it to RGB, resizes it,
    and keeps pixel values in the 0-255 range.

    MobileNetV2 preprocessing is handled inside model.py.
    """

    image = tf.io.read_file(filepath)

    image = tf.image.decode_jpeg(
        image,
        channels=3
    )

    image = tf.image.resize(
        image,
        IMG_SIZE
    )

    # Keep pixel values in 0-255 range.
    image = tf.cast(
        image,
        tf.float32
    )

    # Convert integer label to one-hot vector.
    label = tf.one_hot(
        label,
        depth=NUM_CLASSES
    )

    return image, label


# AUGMENTATION

_rotation_layer = tf.keras.layers.RandomRotation(
    factor=0.03
)


def augment_image(image, label):

    # Horizontal flip DISABLED

    image = tf.image.random_brightness(
        image,
        max_delta=38.25
    )

    # Random contrast

    image = tf.image.random_contrast(
        image,
        lower=0.85,
        upper=1.15
    )

    # Small rotation

    image = _rotation_layer(
        image,
        training=True
    )

    # Keep pixels in valid 0-255 range

    image = tf.clip_by_value(
        image,
        0.0,
        255.0
    )

    return image, label


# BUILD DATASET

def build_dataset(
    csv_path,
    batch_size=BATCH_SIZE,
    augment=False,
    shuffle=False
):

    filepaths, labels = load_csv(csv_path)

    dataset = tf.data.Dataset.from_tensor_slices(
        (filepaths, labels)
    )

    # Shuffle training data only.
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(filepaths),
            seed=42
        )

    # Load images.
    dataset = dataset.map(
        load_and_decode_image,
        num_parallel_calls=AUTOTUNE
    )

    # Apply augmentation to training data only.
    if augment:
        dataset = dataset.map(
            augment_image,
            num_parallel_calls=AUTOTUNE
        )

    # Create batches.
    dataset = dataset.batch(
        batch_size
    )

    # Improve pipeline performance.
    dataset = dataset.prefetch(
        AUTOTUNE
    )

    return dataset

# GET TRAIN / VALIDATION / TEST DATASETS

def get_datasets():

    train_ds = build_dataset(
        TRAIN_CSV,
        augment=True,
        shuffle=True
    )

    val_ds = build_dataset(
        VAL_CSV,
        augment=False,
        shuffle=False
    )

    test_ds = build_dataset(
        TEST_CSV,
        augment=False,
        shuffle=False
    )

    return train_ds, val_ds, test_ds

# SANITY CHECK

if __name__ == "__main__":

    train_ds, val_ds, test_ds = get_datasets()

    print("=" * 60)
    print("DATA LOADER SANITY CHECK")
    print("=" * 60)

    # Training batch

    for images, labels in train_ds.take(1):

        print(
            f"\nTrain batch image shape : {images.shape}"
        )

        print(
            f"Train batch label shape : {labels.shape}"
        )

        print(
            f"Train pixel range       : "
            f"[{tf.reduce_min(images):.3f}, "
            f"{tf.reduce_max(images):.3f}]"
        )

    # Validation batch

    for images, labels in val_ds.take(1):

        print(
            f"\nVal batch image shape   : {images.shape}"
        )

        print(
            f"Val batch label shape   : {labels.shape}"
        )

        print(
            f"Val pixel range         : "
            f"[{tf.reduce_min(images):.3f}, "
            f"{tf.reduce_max(images):.3f}]"
        )

    # Test batch

    for images, labels in test_ds.take(1):

        print(
            f"\nTest batch image shape  : {images.shape}"
        )

        print(
            f"Test batch label shape  : {labels.shape}"
        )

        print(
            f"Test pixel range        : "
            f"[{tf.reduce_min(images):.3f}, "
            f"{tf.reduce_max(images):.3f}]"
        )

    print("\nData loader working correctly.")