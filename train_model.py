"""Train the sign classifier from the preprocessed JSON-manifest dataset."""

import os
import time

import tensorflow as tf

from config import EPOCHS, FINE_TUNE_EPOCHS, MODEL_PATH, MODELS_DIR
from data_loader import get_datasets, load_class_names
from model import build_mobilenetv2_model, unfreeze_for_fine_tuning

BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_mobilenetv2_stage1.keras")


def make_callbacks():
    return [
        tf.keras.callbacks.ModelCheckpoint(
            BEST_MODEL_PATH,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    classes = load_class_names()
    train_ds, val_ds, test_ds = get_datasets()

    print(f"Training with {len(classes)} classes: {classes}")
    model = build_mobilenetv2_model(num_classes=len(classes), freeze_base=True)
    model.summary()

    start_time = time.time()
    print("\n--- Stage 1: train the classification head ---")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=make_callbacks(),
        shuffle=False,
    )

    print("\n--- Stage 2: fine-tune the MobileNetV2 backbone ---")
    model = unfreeze_for_fine_tuning(model, unfreeze_last_n=35, fine_tune_lr=1e-5)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FINE_TUNE_EPOCHS,
        callbacks=make_callbacks(),
        shuffle=False,
    )

    model.save(MODEL_PATH)
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
    print(f"\nTraining finished in {(time.time() - start_time) / 60:.1f} minutes.")
    print(f"Final model saved to: {MODEL_PATH}")
    print(f"Test loss: {test_loss:.4f}; test accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
