import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2


# ============================================================
# CONFIGURATION
# ============================================================

IMG_SIZE = (128, 128)
NUM_CLASSES = 28


# ============================================================
# BUILD MODEL
# ============================================================

def build_model():

    # Load pretrained MobileNetV2
    base_model = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )

    # Freeze MobileNetV2 for Stage 1 training
    base_model.trainable = False


    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

    inputs = layers.Input(
        shape=(*IMG_SIZE, 3),
        name="input_image"
    )


    # --------------------------------------------------------
    # MOBILE NET V2 PREPROCESSING
    # --------------------------------------------------------

    x = tf.keras.applications.mobilenet_v2.preprocess_input(
        inputs
    )


    # --------------------------------------------------------
    # MOBILE NET V2 FEATURE EXTRACTION
    # --------------------------------------------------------

    x = base_model(
        x,
        training=False
    )


    # --------------------------------------------------------
    # CLASSIFICATION HEAD
    # --------------------------------------------------------

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.30)(x)

    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="predictions"
    )(x)


    # --------------------------------------------------------
    # CREATE MODEL
    # --------------------------------------------------------

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="SignLanguage_MobileNetV2"
    )

    return model


# ============================================================
# CREATE MODEL
# ============================================================

model = build_model()


# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# MODEL SUMMARY
# ============================================================

model.summary()


# ============================================================
# TEST MODEL
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("MOBILENETV2 MODEL TEST")
    print("=" * 60)

    print(f"\nInput shape  : {model.input_shape}")
    print(f"Output shape : {model.output_shape}")
    print(f"Classes      : {NUM_CLASSES}")

    print("\nModel created successfully.")