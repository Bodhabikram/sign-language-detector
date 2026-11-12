"""Three-convolutional-layer CNN for ASL image classification."""

import tensorflow as tf
from tensorflow.keras import layers, models


def build_cnn_model(
    num_classes: int = 29,
    input_shape: tuple[int, int, int] = (224, 224, 3),
    learning_rate: float = 1e-3,
    internal_size: int = 96,
    use_batch_norm: bool = False,
) -> tf.keras.Model:
    """Build and compile a CNN with three convolutional feature extractors."""
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2.")

    inputs = layers.Input(shape=input_shape, name="image")
    x = layers.Resizing(internal_size, internal_size, name="resize")(inputs)
    x = layers.Rescaling(1.0 / 255.0, name="rescale")(x)

    for index, filters in enumerate((32, 64, 128), start=1):
        x = layers.Conv2D(
            filters,
            kernel_size=3,
            padding="same",
            use_bias=not use_batch_norm,
            name=f"feature_conv_{index}",
        )(x)
        if use_batch_norm:
            x = layers.BatchNormalization(name=f"feature_bn_{index}")(x)
        x = layers.Activation("relu", name=f"feature_relu_{index}")(x)
        x = layers.MaxPooling2D(pool_size=2, name=f"feature_pool_{index}")(x)

    x = layers.GlobalAveragePooling2D(name="global_average_pool")(x)
    x = layers.Dropout(0.35, name="classifier_dropout")(x)
    x = layers.Dense(128, activation="relu", name="classifier_dense")(x)
    x = layers.Dropout(0.25, name="classifier_dropout_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="ASL_ThreeLayer_CNN")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
