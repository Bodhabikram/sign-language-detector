"""
Convert dataset/raw_asl images into MediaPipe hand-landmark JSON files.

    dataset/raw_asl/<class>/*.jpg   ->   dataset/landmarks/train.json
                                         dataset/landmarks/val.json
                                         dataset/landmarks/test.json
                                         models/landmark_extraction_report.json

Run:  python extract_landmarks.py
      python extract_landmarks.py --max-per-class 1000     (quick first run)

IMPORTANT: this reads the RAW images. No CLAHE, denoising, skin masking or
resizing, because those change the picture MediaPipe has to find the hand in.

Coordinates are stored aspect-correct: x*width, y*height, z*width, all divided
by the longer image side. The live scripts must use the same convention.
"""
import os
import json
import glob
import random
import argparse
import cv2
import numpy as np
import mediapipe as mp

from config import RAW_ASL_DIR, MODELS_DIR

# ---------------- SETTINGS ----------------
LANDMARKS_DIR = os.path.join(os.path.dirname(os.path.abspath(RAW_ASL_DIR)), "landmarks")
REPORT_PATH = os.path.join(MODELS_DIR, "landmark_extraction_report.json")

# Tightly cropped hands often fail detection, so retry with extra border around the image
PAD_RATIOS = (0.0, 0.25, 0.5)
MIN_DETECTION_CONFIDENCE = 0.5
LOW_RATE_WARNING = 0.70          # warn if fewer than 70% of a class's images give a hand
TRAIN_RATIO, VAL_RATIO = 0.70, 0.15
SEED = 42


def list_images(folder):
    files = []
    for ext in ("jpg", "jpeg", "png"):
        files += glob.glob(os.path.join(folder, f"*.{ext}"))
        files += glob.glob(os.path.join(folder, f"*.{ext.upper()}"))
    return sorted(set(files))


def detect_landmarks(hands, img_bgr):
    """Returns a (21, 3) array of aspect-correct landmarks, or None if no hand is found."""
    h0, w0 = img_bgr.shape[:2]
    for ratio in PAD_RATIOS:
        pad = int(round(max(h0, w0) * ratio))
        img = cv2.copyMakeBorder(img_bgr, pad, pad, pad, pad, cv2.BORDER_REPLICATE) if pad else img_bgr
        h, w = img.shape[:2]
        result = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            lm = np.array([[p.x * w, p.y * h, p.z * w] for p in hand.landmark], dtype=np.float32)
            return lm / max(h, w)
    return None


def split_per_class(records_by_class):
    rng = random.Random(SEED)
    train, val, test = [], [], []
    for items in records_by_class.values():
        rng.shuffle(items)
        n = len(items)
        n_train, n_val = int(n * TRAIN_RATIO), int(n * VAL_RATIO)
        train += items[:n_train]
        val += items[n_train:n_train + n_val]
        test += items[n_train + n_val:]
    for part in (train, val, test):
        rng.shuffle(part)
    return train, val, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=RAW_ASL_DIR)
    parser.add_argument("--output", default=LANDMARKS_DIR)
    parser.add_argument("--max-per-class", type=int, default=None)
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        raise FileNotFoundError(f"Raw dataset folder not found: {args.input}")

    classes = sorted(d for d in os.listdir(args.input)
                     if os.path.isdir(os.path.join(args.input, d)) and not d.startswith("."))
    if len(classes) < 2:
        raise ValueError(f"Need class subfolders in '{args.input}'. Found: {classes}")

    rng = random.Random(SEED)
    hands = mp.solutions.hands.Hands(
        static_image_mode=True, max_num_hands=1,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE)

    records_by_class, report = {}, {}
    print(f"[Extract] {len(classes)} classes from {args.input}\n")

    for cls in classes:
        files = list_images(os.path.join(args.input, cls))
        if args.max_per_class:
            rng.shuffle(files)
            files = files[:args.max_per_class]

        records = []
        for i, path in enumerate(files, 1):
            img = cv2.imread(path)
            if img is None:
                continue
            lm = detect_landmarks(hands, img)
            if lm is not None:
                records.append({
                    "label": cls,
                    "landmarks": np.round(lm, 5).tolist(),
                    "source": os.path.relpath(path, args.input).replace(os.sep, "/"),
                })
            if i % 1000 == 0:
                print(f"    {cls}: {i}/{len(files)}")

        rate = len(records) / len(files) if files else 0.0
        flag = "   <-- low detection rate" if rate < LOW_RATE_WARNING else ""
        print(f"  {cls:>8}: {len(records):>6} / {len(files):<6} ({rate * 100:5.1f}%){flag}")
        report[cls] = {"images": len(files), "detected": len(records), "rate": round(rate, 4)}
        if records:
            records_by_class[cls] = records

    hands.close()

    dropped = [c for c in classes if c not in records_by_class]
    if dropped:
        print(f"\n[Warning] No hand detected at all for: {dropped}. These classes are left out.")
        print("          (A class like 'nothing' has no hand in its images, so this is expected.)")
    if not records_by_class:
        raise RuntimeError("No landmarks extracted. Check the dataset folder and MediaPipe install.")

    train, val, test = split_per_class(records_by_class)
    os.makedirs(args.output, exist_ok=True)
    for name, part in (("train", train), ("val", val), ("test", test)):
        with open(os.path.join(args.output, f"{name}.json"), "w") as f:
            json.dump(part, f)
        print(f"[Saved] {name}.json: {len(part)} samples")

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump({"classes_kept": sorted(records_by_class), "classes_dropped": dropped,
                   "per_class": report}, f, indent=2)
    print(f"[Saved] detection report -> {REPORT_PATH}")
    print(f"\nDone. Output folder: {args.output}")


if __name__ == "__main__":
    main()