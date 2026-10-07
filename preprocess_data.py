import os
import cv2
import json
import glob
import random
import shutil
import argparse
import numpy as np

from config import MODELS_DIR, LABELS_PATH, IMG_SIZE, BOX_PADDING

# Default directory configurations targeting dataset/raw_asl
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RAW_DIR = os.path.join(BASE_DIR, "dataset", "raw_asl")
DEFAULT_PROCESSED_DIR = os.path.join(BASE_DIR, "dataset", "processed")
CLASS_WEIGHTS_PATH = os.path.join(MODELS_DIR, "class_weights.json")


# =====================================================================
# 1. NOISE & QUALITY CONTROL
# =====================================================================
def check_image_quality(img_bgr, min_blur_var=60.0, min_brightness=25.0, max_brightness=240.0):
    
    if img_bgr is None or img_bgr.size == 0:
        return False, "Corrupted or unreadable image"

    h, w = img_bgr.shape[:2]
    if h < 30 or w < 30:
        return False, f"Resolution too low ({w}x{h})"

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Blur detection via Laplacian variance
    blur_variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_variance < min_blur_var:
        return False, f"Too blurry (variance {blur_variance:.1f} < {min_blur_var})"

    # Brightness / exposure validation
    avg_brightness = float(np.mean(gray))
    if avg_brightness < min_brightness:
        return False, f"Too dark / underexposed ({avg_brightness:.1f} < {min_brightness})"
    if avg_brightness > max_brightness:
        return False, f"Too bright / overexposed ({avg_brightness:.1f} > {max_brightness})"

    return True, "Valid"


# =====================================================================
# 2. HAND REGION ISOLATION & SKIN MASKING
# =====================================================================
def isolate_hand_region(img_bgr, tracker=None):
   
    if tracker is not None:
        try:
            landmarks = tracker.find_hand(img_bgr)
            if landmarks:
                cropped, _ = tracker.extract_hand_roi(
                    img_bgr, landmarks, target_size=IMG_SIZE, padding=BOX_PADDING
                )
                if cropped is not None:
                    return cropped
        except Exception:
            pass
    return img_bgr


def suppress_background(img_bgr, bg_intensity=0.3):
   
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    # Human skin color boundaries in YCrCb space
    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

    # Clean mask noise with morphological closing/dilation and gentle blur
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Soft attenuation: keep hand sharp while dimming non-skin background
    mask_norm = (mask.astype(np.float32) / 255.0)[:, :, None]
    dimmed_bg = img_bgr * bg_intensity
    isolated = (img_bgr * mask_norm + dimmed_bg * (1.0 - mask_norm)).astype(np.uint8)
    return isolated


# =====================================================================
# 3. IMAGE-LEVEL PROCESSING (CLAHE & DENOISING)
# =====================================================================
def enhance_image_levels(img_bgr, target_size=(IMG_SIZE, IMG_SIZE)):

    # 1. Edge-preserving smoothing
    denoised = cv2.bilateralFilter(img_bgr, d=5, sigmaColor=50, sigmaSpace=50)

    # 2. CLAHE on Luminance (LAB color space)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    enhanced_bgr = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    # 3. Resize to target dimensions
    resized = cv2.resize(enhanced_bgr, target_size, interpolation=cv2.INTER_AREA)
    return resized


def preprocess_single_image(img_bgr, tracker=None, isolate_hand=True, enhance_levels=True):
    """Combines hand region isolation, background suppression, and image-level enhancements."""
    processed = img_bgr
    if isolate_hand:
        processed = isolate_hand_region(processed, tracker=tracker)
        processed = suppress_background(processed)
    if enhance_levels:
        processed = enhance_image_levels(processed)
    return processed


# =====================================================================
# 4. DATA AUGMENTATION (GEOMETRIC & PHOTOMETRIC)
# =====================================================================
def augment_image(img_bgr):
   
    h, w = img_bgr.shape[:2]

    # 1. Random Rotation & Zoom
    angle = random.uniform(-10.0, 10.0)
    scale = random.uniform(0.92, 1.08)
    center = (w / 2, h / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)

    # 2. Random Translation
    tx = random.uniform(-0.06, 0.06) * w
    ty = random.uniform(-0.06, 0.06) * h
    matrix[0, 2] += tx
    matrix[1, 2] += ty

    augmented = cv2.warpAffine(
        img_bgr, matrix, (w, h),
        borderMode=cv2.BORDER_REFLECT_101
    )

    # 3. Random Brightness & Contrast Jitter
    alpha = random.uniform(0.85, 1.15)  # Contrast
    beta = random.uniform(-15.0, 15.0)  # Brightness
    augmented = np.clip(alpha * augmented.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    return augmented


# =====================================================================
# 5. LABELLING, QUALITY SCREENING & STRATIFIED SPLIT
# =====================================================================
def scan_and_label_dataset(raw_dir=DEFAULT_RAW_DIR):
   
    if not os.path.exists(raw_dir):
        raise FileNotFoundError(f"Raw data directory '{raw_dir}' not found. Please verify the folder exists.")

    classes = sorted([
        d for d in os.listdir(raw_dir)
        if os.path.isdir(os.path.join(raw_dir, d)) and not d.startswith('.')
    ])
    if len(classes) < 2:
        raise ValueError(
            f"Need at least 2 class subdirectories in '{raw_dir}'. Found {len(classes)}: {classes}"
        )

    class_to_idx = {c: i for i, c in enumerate(classes)}

    valid_samples = []  # list of (file_path, class_name, class_idx)
    discarded_stats = {"total_scanned": 0, "accepted": 0, "rejected": 0}

    print(f"\n[Quality Control] Screening raw dataset in '{raw_dir}'...")
    for cls in classes:
        cls_folder = os.path.join(raw_dir, cls)
        image_files = glob.glob(os.path.join(cls_folder, "*.[jJ][pP][gG]")) + \
                      glob.glob(os.path.join(cls_folder, "*.[pP][nN][gG]")) + \
                      glob.glob(os.path.join(cls_folder, "*.[jJ][pP][eE][gG]"))
        
        cls_accepted = 0
        for p in image_files:
            discarded_stats["total_scanned"] += 1
            img = cv2.imread(p)
            is_valid, reason = check_image_quality(img)
            if is_valid:
                valid_samples.append((p, cls, class_to_idx[cls]))
                cls_accepted += 1
            else:
                discarded_stats["rejected"] += 1

        discarded_stats["accepted"] += cls_accepted
        print(f"  - Class '{cls}': {cls_accepted} accepted / {len(image_files)} total")

    print(f"[Quality Control] Total Accepted: {discarded_stats['accepted']} | Rejected: {discarded_stats['rejected']}")
    return valid_samples, classes, class_to_idx


def stratified_split(samples, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    
    rng = random.Random(seed)
    by_class = {}
    for sample in samples:
        by_class.setdefault(sample[1], []).append(sample)

    train_set, val_set, test_set = [], [], []

    for cls, items in by_class.items():
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_set.extend(items[:n_train])
        val_set.extend(items[n_train:n_train + n_val])
        test_set.extend(items[n_train + n_val:])

    rng.shuffle(train_set)
    rng.shuffle(val_set)
    rng.shuffle(test_set)

    print(f"\n[Splits] Stratified Split:")
    print(f"  - Train: {len(train_set)} samples ({train_ratio*100:.0f}%)")
    print(f"  - Validation: {len(val_set)} samples ({val_ratio*100:.0f}%)")
    print(f"  - Test: {len(test_set)} samples ({test_ratio*100:.0f}%)")

    return train_set, val_set, test_set


# =====================================================================
# 6. CLASS BALANCING
# =====================================================================
def compute_and_save_class_weights(train_samples, num_classes, save_path=None):

    if save_path is None:
        save_path = CLASS_WEIGHTS_PATH
    
    labels = [s[2] for s in train_samples]
    counts = np.bincount(labels, minlength=num_classes)
    total = len(labels)

    class_weights = {}
    print("\n[Class Balancing] Sample frequencies and weights:")
    for i in range(num_classes):
        cnt = counts[i]
        weight = float(total / (num_classes * cnt)) if cnt > 0 else 1.0
        class_weights[i] = round(weight, 4)
        print(f"  - Class index {i}: count={cnt}, weight={weight:.4f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(class_weights, f, indent=4)
    print(f"[Class Balancing] Weights saved to: {save_path}")
    return class_weights


# =====================================================================
# 7. EXPORT PREPROCESSED DATASET TO DISK
# =====================================================================
def build_processed_dataset(raw_dir=DEFAULT_RAW_DIR, output_dir=DEFAULT_PROCESSED_DIR,
                            augment_train=True, target_per_class=None, tracker=None):
    
    print(f"Starting ASL dataset preprocessing from:\n -> Source: {raw_dir}\n -> Destination: {output_dir}\n")
    raw_abs = os.path.normcase(os.path.abspath(raw_dir))
    output_abs = os.path.normcase(os.path.abspath(output_dir))
    common_path = os.path.commonpath((raw_abs, output_abs))
    if common_path in (raw_abs, output_abs):
        raise ValueError("The output directory must be separate from the raw dataset directory.")

    samples, classes, class_to_idx = scan_and_label_dataset(raw_dir)
    num_classes = len(classes)

    if not samples:
        print(f"[Warning] No valid images found in '{raw_dir}'. Please ensure your gesture images are located inside '{raw_dir}/<class_name>/'.")
        return

    train_set, val_set, test_set = stratified_split(samples, 0.70, 0.15, 0.15)
    compute_and_save_class_weights(train_set, num_classes)

    partitions = {
        "train": train_set,
        "val": val_set,
        "test": test_set
    }

    print(f"\n[Processing] Writing preprocessed dataset to: {output_dir}")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    manifest_splits = {split_name: [] for split_name in partitions}

    for split_name, split_samples in partitions.items():
        is_train = (split_name == "train")
        print(f"  Processing '{split_name}' split ({len(split_samples)} samples)...")

        # Track count per class for training oversampling/balancing
        class_counts = {}

        for file_path, cls, class_idx in split_samples:
            target_class_dir = os.path.join(output_dir, split_name, cls)
            os.makedirs(target_class_dir, exist_ok=True)

            img = cv2.imread(file_path)
            if img is None:
                raise OSError(f"Could not read image after quality screening: {file_path}")
            processed_img = preprocess_single_image(img, tracker=tracker)

            base_name = os.path.splitext(os.path.basename(file_path))[0]
            out_path = os.path.join(target_class_dir, f"{base_name}.jpg")
            if not cv2.imwrite(out_path, processed_img):
                raise OSError(f"Could not write processed image: {out_path}")

            manifest_splits[split_name].append({
                "image": os.path.relpath(out_path, output_dir).replace(os.sep, "/"),
                "class_name": cls,
                "label": class_idx,
            })

            class_counts[cls] = class_counts.get(cls, 0) + 1

        # If training split and augmentation is enabled, synthesize augmented samples
        if is_train and augment_train and class_counts:
            print("  Applying balanced data augmentation to training partition...")
            max_class_count = max(class_counts.values()) if target_per_class is None else target_per_class

            for cls in classes:
                cls_dir = os.path.join(output_dir, "train", cls)
                existing_files = glob.glob(os.path.join(cls_dir, "*.jpg"))
                current_cnt = len(existing_files)

                # Augment up to max_class_count to achieve balanced class counts
                needed = max_class_count - current_cnt
                if needed > 0 and existing_files:
                    for i in range(needed):
                        src_path = random.choice(existing_files)
                        src_img = cv2.imread(src_path)
                        if src_img is None:
                            raise OSError(f"Could not read processed image for augmentation: {src_path}")
                        aug_img = augment_image(src_img)

                        aug_name = f"aug_{i}_{os.path.basename(src_path)}"
                        aug_path = os.path.join(cls_dir, aug_name)
                        if not cv2.imwrite(aug_path, aug_img):
                            raise OSError(f"Could not write augmented image: {aug_path}")
                        manifest_splits[split_name].append({
                            "image": os.path.relpath(aug_path, output_dir).replace(os.sep, "/"),
                            "class_name": cls,
                            "label": class_to_idx[cls],
                        })

    os.makedirs(MODELS_DIR, exist_ok=True)
    idx_to_class = {i: name for i, name in enumerate(classes)}
    with open(LABELS_PATH, "w", encoding="utf-8") as labels_file:
        json.dump(idx_to_class, labels_file, indent=4)

    split_counts = {
        split_name: {
            cls: sum(record["class_name"] == cls for record in records)
            for cls in classes
        }
        for split_name, records in manifest_splits.items()
    }
    manifest = {
        "version": 1,
        "image_size": [IMG_SIZE, IMG_SIZE],
        "classes": classes,
        "num_classes": num_classes,
        "split_seed": 42,
        "split_ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
        "counts": split_counts,
        "splits": manifest_splits,
    }
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)

    print(f"[Labels] Saved {len(classes)} classes to: {LABELS_PATH}")
    print(f"[Manifest] Saved training records to: {manifest_path}")
    print(f"\nPreprocessed dataset generation finished successfully at:\n{output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess ASL sign language images from dataset/raw_asl")
    parser.add_argument("--input", type=str, default=DEFAULT_RAW_DIR, help="Path to raw dataset folder (default: dataset/raw_asl)")
    parser.add_argument("--output", type=str, default=DEFAULT_PROCESSED_DIR, help="Path to output processed folder (default: dataset/processed)")
    parser.add_argument("--no-augment", action="store_true", help="Disable synthetic augmentation on the train set")
    args = parser.parse_args()

    build_processed_dataset(
        raw_dir=args.input,
        output_dir=args.output,
        augment_train=not args.no_augment
    )


if __name__ == "__main__":
    main()