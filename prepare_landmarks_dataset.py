"""Extract MediaPipe landmarks for the exact raw images used by the image models."""

import argparse
import json
import os
import time

import cv2
import mediapipe as mp
import numpy as np

from config import DATASET_MANIFEST_PATH, MODELS_DIR, RAW_ASL_DIR
from extract_landmarks import detect_landmarks
from landmark_features import NO_HAND_LABEL

DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(RAW_ASL_DIR), "landmarks_manifest")
LABELS_PATH = os.path.join(MODELS_DIR, "landmarks_labels.json")
REPORT_PATH = os.path.join(MODELS_DIR, "landmark_dataset_report.json")


def raw_path_for_record(record):
    base_name = os.path.splitext(os.path.basename(record["image"]))[0]
    folder = os.path.join(RAW_ASL_DIR, record["class_name"])
    for extension in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        candidate = os.path.join(folder, base_name + extension)
        if os.path.isfile(candidate):
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DATASET_MANIFEST_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Optional cap per class per split (for a quick run)",
    )
    args = parser.parse_args()

    with open(args.manifest, "r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)

    classes = list(manifest["classes"])
    hand_classes = [name for name in classes if name != NO_HAND_LABEL]
    label_to_index = {name: index for index, name in enumerate(hand_classes)}

    hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
    )

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    report = {"classes": hand_classes, "splits": {}}
    start_time = time.time()

    try:
        for split_name in ("train", "val"):
            records = list(manifest["splits"][split_name])
            per_class_counts = {}
            kept = []
            missed = 0
            skipped_no_hand_class = 0

            for record in records:
                class_name = record["class_name"]
                if class_name == NO_HAND_LABEL:
                    skipped_no_hand_class += 1
                    continue
                if args.max_per_class is not None:
                    used = per_class_counts.get(class_name, 0)
                    if used >= args.max_per_class:
                        continue

                raw_path = raw_path_for_record(record)
                if raw_path is None:
                    missed += 1
                    continue
                image = cv2.imread(raw_path)
                if image is None:
                    missed += 1
                    continue

                landmarks = detect_landmarks(hands, image)
                if landmarks is None:
                    missed += 1
                    continue

                per_class_counts[class_name] = per_class_counts.get(class_name, 0) + 1
                kept.append(
                    {
                        "label": class_name,
                        "label_index": label_to_index[class_name],
                        "landmarks": np.round(landmarks, 5).tolist(),
                        "source": os.path.relpath(raw_path, RAW_ASL_DIR).replace(os.sep, "/"),
                    }
                )

            output_path = os.path.join(args.output, f"{split_name}.json")
            with open(output_path, "w", encoding="utf-8") as output_file:
                json.dump(kept, output_file)
            report["splits"][split_name] = {
                "records": len(records),
                "kept": len(kept),
                "missed_no_landmarks": missed,
                "skipped_no_hand_class": skipped_no_hand_class,
                "per_class": per_class_counts,
            }
            print(
                f"[{split_name}] kept {len(kept)} / {len(records)} "
                f"(missed {missed}, skipped {skipped_no_hand_class} '{NO_HAND_LABEL}' images)"
            )
    finally:
        hands.close()

    labels = {str(index): name for index, name in enumerate(hand_classes)}
    with open(LABELS_PATH, "w", encoding="utf-8") as labels_file:
        json.dump(labels, labels_file, indent=2)
    with open(REPORT_PATH, "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)

    print(f"\nSaved landmark datasets to: {args.output}")
    print(f"Saved labels to: {LABELS_PATH}")
    print(f"Saved report to: {REPORT_PATH}")
    print(f"Done in {(time.time() - start_time) / 60:.1f} minutes.")


if __name__ == "__main__":
    main()
