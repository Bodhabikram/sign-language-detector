"""Train the sign classifier from the preprocessed JSON-manifest dataset."""

import argparse
import os
import json
import time

import tensorflow as tf

from config import BATCH_SIZE, EPOCHS, FINE_TUNE_EPOCHS, MODEL_PATH, MODELS_DIR
from data_loader import get_datasets, load_class_names
from model import build_mobilenetv2_model, unfreeze_for_fine_tuning

BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_mobilenetv2_stage1.keras")
HISTORY_PATH = os.path.join(MODELS_DIR, "training_history.json")


def make_callbacks(best_model_path=BEST_MODEL_PATH):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            best_model_path,
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Stage-1 epochs")
    parser.add_argument(
        "--fine-tune-epochs",
        type=int,
        default=FINE_TUNE_EPOCHS,
        help="Stage-2 fine-tuning epochs",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable training augmentation (not recommended)",
    )
    parser.add_argument(
        "--output",
        default=MODEL_PATH,
        help="Final model path (.keras or .h5)",
    )
    parser.add_argument(
        "--best-output",
        default=BEST_MODEL_PATH,
        help="Checkpoint path for the best validation accuracy",
    )
    parser.add_argument(
        "--history-output",
        default=HISTORY_PATH,
        help="Path for the training history JSON",
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.fine_tune_epochs < 0:
        parser.error("--epochs must be >= 1 and --fine-tune-epochs must be >= 0")
    if args.batch_size < 1:
        parser.error("--batch-size must be a positive integer")

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.best_output)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.history_output)), exist_ok=True)
    classes = load_class_names()
    train_ds, val_ds = get_datasets(
        batch_size=args.batch_size, augment_train=not args.no_augment
    )

    print(f"Training with {len(classes)} classes: {classes}")
    model = build_mobilenetv2_model(num_classes=len(classes), freeze_base=True)
    model.summary()

    start_time = time.time()
    print("\n--- Stage 1: train the classification head ---")
    stage1_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=make_callbacks(args.best_output),
        shuffle=False,
    )

    print("\n--- Stage 2: fine-tune the MobileNetV2 backbone ---")
    model = unfreeze_for_fine_tuning(model, unfreeze_last_n=35, fine_tune_lr=1e-5)
    stage2_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.fine_tune_epochs,
        callbacks=make_callbacks(args.best_output),
        shuffle=False,
    )

    combined_history = {}
    history_keys = set(stage1_history.history) | set(stage2_history.history)
    for key in history_keys:
        combined_history[key] = (
            list(stage1_history.history.get(key, []))
            + list(stage2_history.history.get(key, []))
        )
    combined_history["stage1_epochs"] = len(stage1_history.epoch)
    with open(args.history_output, "w", encoding="utf-8") as history_file:
        json.dump(combined_history, history_file, indent=2)

    model.save(args.output)
    val_loss, val_accuracy = model.evaluate(val_ds, verbose=1)
    print(f"\nTraining finished in {(time.time() - start_time) / 60:.1f} minutes.")
    print(f"Final model saved to: {args.output}")
    print(f"Training history saved to: {args.history_output}")
    print(f"Validation loss: {val_loss:.4f}; validation accuracy: {val_accuracy:.4f}")


if __name__ == "__main__":
    main()
