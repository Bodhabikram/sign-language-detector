"""Compare MobileNetV2, 3-layer CNN, and Landmarks+ANN on the same raw validation images."""

import argparse
import json
import os
import time

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from config import DATASET_MANIFEST_PATH, LABELS_PATH, MODEL_PATH, MODELS_DIR, RAW_ASL_DIR
from extract_landmarks import detect_landmarks
from landmark_features import NO_HAND_LABEL, normalize_landmarks
from preprocess_data import get_hand_crop, preprocess_single_image

CNN_DIR = os.path.join(MODELS_DIR, "cnn")
CNN_MODEL_PATH = os.path.join(CNN_DIR, "cnn_final.keras")
CNN_LABELS_PATH = os.path.join(CNN_DIR, "labels.json")
LANDMARK_MODEL_PATH = os.path.join(MODELS_DIR, "landmarks_ann.keras")
LANDMARK_LABELS_PATH = os.path.join(MODELS_DIR, "landmarks_labels.json")
REPORT_PATH = os.path.join(MODELS_DIR, "model_comparison_report.txt")
RESULTS_PATH = os.path.join(MODELS_DIR, "model_comparison.json")


def raw_path_for_record(record):
    base_name = os.path.splitext(os.path.basename(record["image"]))[0]
    folder = os.path.join(RAW_ASL_DIR, record["class_name"])
    for extension in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        candidate = os.path.join(folder, base_name + extension)
        if os.path.isfile(candidate):
            return candidate
    return None


def load_label_list(path):
    with open(path, "r", encoding="utf-8") as labels_file:
        mapping = json.load(labels_file)
    return [mapping[str(index)] for index in range(len(mapping))]


def preprocess_for_model(image_bgr):
    processed = preprocess_single_image(image_bgr)
    rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB).astype(np.float32)
    return np.expand_dims(rgb, axis=0)


def summarize(name, true_names, pred_names, classes, extra=None):
    class_to_index = {class_name: index for index, class_name in enumerate(classes)}
    y_true = np.array([class_to_index[name] for name in true_names])
    y_pred = np.array([class_to_index[name] for name in pred_names])
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "samples": int(len(y_true)),
    }
    if extra:
        metrics.update(extra)
    metrics["report"] = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(classes))),
        target_names=classes,
        zero_division=0,
    )
    print(
        f"  {name:<20} acc={metrics['accuracy'] * 100:5.2f}%  "
        f"f1_weighted={metrics['f1_weighted'] * 100:5.2f}%"
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit for a quicker comparison",
    )
    parser.add_argument(
        "--models",
        default="mobilenet,cnn,landmarks",
        help="Comma-separated subset of: mobilenet,cnn,landmarks",
    )
    parser.add_argument(
        "--crop-mode",
        choices=("live", "full"),
        default="live",
        help="Image-model input: live-style hand crop or full frame",
    )
    parser.add_argument("--mobilenet-model", default=None)
    parser.add_argument("--mobilenet-labels", default=None)
    parser.add_argument("--cnn-model", default=None)
    parser.add_argument("--cnn-labels", default=None)
    args = parser.parse_args()

    mobilenet_model_path = args.mobilenet_model or MODEL_PATH
    mobilenet_labels_path = args.mobilenet_labels or LABELS_PATH
    cnn_model_path = args.cnn_model or CNN_MODEL_PATH
    cnn_labels_path = args.cnn_labels or CNN_LABELS_PATH
    selected = {item.strip().lower() for item in args.models.split(",")}
    with open(DATASET_MANIFEST_PATH, "r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    classes = list(manifest["classes"])
    records = list(manifest["splits"][args.split])

    if args.max_samples is not None and args.max_samples < len(records):
        rng = np.random.default_rng(42)
        indices = rng.choice(len(records), size=args.max_samples, replace=False)
        records = [records[index] for index in sorted(indices)]

    models = {}
    if "mobilenet" in selected and os.path.isfile(mobilenet_model_path):
        models["MobileNetV2"] = (
            tf.keras.models.load_model(mobilenet_model_path, compile=False),
            load_label_list(mobilenet_labels_path),
        )
    if "cnn" in selected and os.path.isfile(cnn_model_path):
        models["3-layer CNN"] = (
            tf.keras.models.load_model(cnn_model_path, compile=False),
            load_label_list(cnn_labels_path),
        )
    if "landmarks" in selected and os.path.isfile(LANDMARK_MODEL_PATH):
        models["Landmarks + ANN"] = (
            tf.keras.models.load_model(LANDMARK_MODEL_PATH, compile=False),
            load_label_list(LANDMARK_LABELS_PATH),
        )

    if not models:
        raise FileNotFoundError(
            "No trained models found. Train at least one model before comparing."
        )

    print(f"Comparing on {len(records)} '{args.split}' raw images")
    print(f"Models: {', '.join(models)}")
    print(f"Image-model crop mode: {args.crop_mode}\n")

    hands = mp.solutions.hands.Hands(
        static_image_mode=True, max_num_hands=2, min_detection_confidence=0.6
    )
    true_names = {name: [] for name in models}
    pred_names = {name: [] for name in models}
    landmark_detected = []
    start_time = time.time()

    try:
        for index, record in enumerate(records, 1):
            raw_path = raw_path_for_record(record)
            if raw_path is None:
                continue
            image = cv2.imread(raw_path)
            if image is None:
                continue
            ground_truth = record["class_name"]

            image_batch = None
            image_models = [name for name in models if name != "Landmarks + ANN"]
            if image_models:
                if args.crop_mode == "live":
                    crop_results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    if crop_results.multi_hand_landmarks:
                        hand_crop = get_hand_crop(
                            image, crop_results.multi_hand_landmarks[0]
                        )
                        if hand_crop is not None and hand_crop.size:
                            image_batch = preprocess_for_model(hand_crop)
                else:
                    image_batch = preprocess_for_model(image)

            for name, (model, labels) in models.items():
                if name == "Landmarks + ANN":
                    landmarks = detect_landmarks(hands, image)
                    if landmarks is None:
                        predicted = NO_HAND_LABEL
                        landmark_detected.append(0)
                    else:
                        landmark_detected.append(1)
                        features = np.expand_dims(normalize_landmarks(landmarks), axis=0)
                        probabilities = model(features, training=False).numpy()[0]
                        predicted = labels[int(np.argmax(probabilities))]
                elif image_batch is None:
                    predicted = NO_HAND_LABEL
                else:
                    probabilities = model(image_batch, training=False).numpy()[0]
                    predicted = labels[int(np.argmax(probabilities))]
                true_names[name].append(ground_truth)
                pred_names[name].append(predicted)

            if index % 250 == 0:
                print(f"  processed {index}/{len(records)} images")
    finally:
        hands.close()

    print("\n" + "=" * 64)
    print(
        f"MODEL COMPARISON ({args.split} split, crop={args.crop_mode}, "
        f"{len(next(iter(true_names.values())))} samples)"
    )
    print("=" * 64)
    results = {}
    for name in models:
        extra = None
        if name == "Landmarks + ANN":
            detected = np.array(landmark_detected)
            detected_true = [
                truth for truth, flag in zip(true_names[name], detected) if flag
            ]
            detected_pred = [
                prediction for prediction, flag in zip(pred_names[name], detected) if flag
            ]
            extra = {"hand_detection_rate": float(detected.mean())}
            if detected_true:
                extra["accuracy_on_detected"] = float(
                    accuracy_score(detected_true, detected_pred)
                )
        results[name] = summarize(name, true_names[name], pred_names[name], classes, extra)

    with open(RESULTS_PATH, "w", encoding="utf-8") as results_file:
        json.dump(results, results_file, indent=2, default=float)
    with open(REPORT_PATH, "w", encoding="utf-8") as report_file:
        for name, metrics in results.items():
            report_file.write(
                f"\n{'=' * 64}\n{name}\n{'=' * 64}\n"
                f"Accuracy: {metrics['accuracy']:.4f}\n"
                f"Precision (weighted): {metrics['precision_weighted']:.4f}\n"
                f"Recall (weighted): {metrics['recall_weighted']:.4f}\n"
                f"F1 (weighted): {metrics['f1_weighted']:.4f}\n"
                f"F1 (macro): {metrics['f1_macro']:.4f}\n"
            )
            if "hand_detection_rate" in metrics:
                report_file.write(
                    f"Hand detection rate: {metrics['hand_detection_rate']:.4f}\n"
                )
            if "accuracy_on_detected" in metrics:
                report_file.write(
                    f"Accuracy on detected hands: {metrics['accuracy_on_detected']:.4f}\n"
                )
            report_file.write(f"\n{metrics['report']}\n")

    print(f"\nResults saved to: {RESULTS_PATH}")
    print(f"Full report saved to: {REPORT_PATH}")
    print(f"Comparison finished in {(time.time() - start_time) / 60:.1f} minutes.")


if __name__ == "__main__":
    main()
