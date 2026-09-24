import os
import csv
import random
from PIL import Image

# ============================================================
# CONFIGURATION
# ============================================================

RAW_DATASET = "dataset/raw_asl"
PROCESSED_DATASET = "dataset/processed"

IMG_SIZE = (128, 128)

# Supported image formats
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Output CSV files
LABELS_CSV = "dataset/labels.csv"
TRAIN_CSV = "dataset/train.csv"
VAL_CSV = "dataset/val.csv"
TEST_CSV = "dataset/test.csv"

# Split ratios (must sum to 1.0)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(PROCESSED_DATASET, exist_ok=True)
random.seed(RANDOM_SEED)


# ============================================================
# FIND CLASS FOLDERS
# ============================================================

class_names = sorted([
    folder for folder in os.listdir(RAW_DATASET)
    if os.path.isdir(os.path.join(RAW_DATASET, folder))
])

# class_name -> integer index (needed by most training frameworks)
class_to_index = {name: idx for idx, name in enumerate(class_names)}

print("=" * 60)
print("SIGN LANGUAGE DATASET PREPROCESSING + LABELING")
print("=" * 60)

print(f"\nInput dataset : {RAW_DATASET}")
print(f"Output dataset: {PROCESSED_DATASET}")
print(f"Image size    : {IMG_SIZE}")
print(f"Classes found : {len(class_names)}")

print("\nClasses:")
print(class_names)

if len(class_names) != 28:
    print("\nWARNING:")
    print(f"Expected 28 classes, but found {len(class_names)} classes.")


# ============================================================
# PROCESS EACH CLASS + COLLECT LABEL RECORDS
# ============================================================

total_processed = 0
total_failed = 0

# Each record: (relative_path, class_name, class_index)
all_records = []

print("\n" + "=" * 60)
print("PROCESSING IMAGES")
print("=" * 60)

for class_name in class_names:

    input_class_path = os.path.join(RAW_DATASET, class_name)
    output_class_path = os.path.join(PROCESSED_DATASET, class_name)

    os.makedirs(output_class_path, exist_ok=True)

    processed_count = 0
    failed_count = 0

    image_files = [
        file for file in os.listdir(input_class_path)
        if file.lower().endswith(IMAGE_EXTENSIONS)
    ]

    print(f"\n[{class_name}] Found {len(image_files)} images")

    for filename in image_files:

        input_path = os.path.join(input_class_path, filename)
        output_filename = os.path.splitext(filename)[0] + ".jpg"
        output_path = os.path.join(output_class_path, output_filename)

        try:
            # ------------------------------------------------
            # Open, convert, resize, save
            # ------------------------------------------------
            image = Image.open(input_path)
            image = image.convert("RGB")
            image = image.resize(IMG_SIZE, Image.Resampling.LANCZOS)
            image.save(output_path, "JPEG", quality=95)

            processed_count += 1

            # ------------------------------------------------
            # Record label entry (relative path, for portability)
            # ------------------------------------------------
            relative_path = os.path.join(class_name, output_filename)
            all_records.append((relative_path, class_name, class_to_index[class_name]))

        except Exception as e:
            failed_count += 1
            print(f"  Failed: {filename} -> {e}")

    total_processed += processed_count
    total_failed += failed_count

    print(f"  Processed: {processed_count} | Failed: {failed_count}")


# ============================================================
# WRITE FULL LABELS CSV
# ============================================================

print("\n" + "=" * 60)
print("WRITING LABELS CSV")
print("=" * 60)

with open(LABELS_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filepath", "label", "class_index"])
    writer.writerows(all_records)

print(f"\nWrote {len(all_records)} entries to {LABELS_CSV}")


# ============================================================
# CLASS-BALANCED TRAIN / VAL / TEST SPLIT
# ============================================================
# Splitting per class (not globally at random) keeps every class
# proportionally represented in train, val, and test.

print("\n" + "=" * 60)
print("SPLITTING DATASET (train / val / test)")
print("=" * 60)

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6, \
    "Split ratios must sum to 1.0"

# Group records by class
records_by_class = {name: [] for name in class_names}
for record in all_records:
    records_by_class[record[1]].append(record)

train_records, val_records, test_records = [], [], []

for class_name, records in records_by_class.items():
    shuffled = records[:]
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    # remainder goes to test, so all images are accounted for
    n_test = n - n_train - n_val

    train_records.extend(shuffled[:n_train])
    val_records.extend(shuffled[n_train:n_train + n_val])
    test_records.extend(shuffled[n_train + n_val:])

    print(f"[{class_name}] train={n_train} val={n_val} test={n_test}")


def write_split_csv(path, records):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "label", "class_index"])
        writer.writerows(records)


write_split_csv(TRAIN_CSV, train_records)
write_split_csv(VAL_CSV, val_records)
write_split_csv(TEST_CSV, test_records)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PREPROCESSING + LABELING COMPLETE")
print("=" * 60)

print(f"\nClasses processed : {len(class_names)}")
print(f"Images processed  : {total_processed}")
print(f"Images failed     : {total_failed}")

print(f"\nTrain samples     : {len(train_records)}")
print(f"Val samples       : {len(val_records)}")
print(f"Test samples      : {len(test_records)}")

print(f"\nProcessed dataset saved at : {PROCESSED_DATASET}")
print(f"Full labels CSV            : {LABELS_CSV}")
print(f"Train CSV                  : {TRAIN_CSV}")
print(f"Val CSV                    : {VAL_CSV}")
print(f"Test CSV                   : {TEST_CSV}")

print("\nDataset structure:")
print("dataset/")
print("├── raw_asl/")
print("│   ├── A/")
print("│   ├── B/")
print("│   └── ...")
print("│")
print("├── processed/")
print("│   ├── A/")
print("│   ├── B/")
print("│   └── ...")
print("│")
print("├── labels.csv     <- full dataset, all images labeled")
print("├── train.csv       <- 70% per class")
print("├── val.csv         <- 15% per class")
print("└── test.csv        <- 15% per class")

print("\nDone!")