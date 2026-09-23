import os
from PIL import Image

# ============================================================
# CONFIGURATION
# ============================================================

RAW_DATASET = "dataset/raw_asl"
PROCESSED_DATASET = "dataset/processed"

IMG_SIZE = (128, 128)

# Supported image formats
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(PROCESSED_DATASET, exist_ok=True)


# ============================================================
# FIND CLASS FOLDERS
# ============================================================

class_names = sorted([
    folder for folder in os.listdir(RAW_DATASET)
    if os.path.isdir(os.path.join(RAW_DATASET, folder))
])

print("=" * 60)
print("SIGN LANGUAGE DATASET PREPROCESSING")
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
# PROCESS EACH CLASS
# ============================================================

total_processed = 0
total_failed = 0

print("\n" + "=" * 60)
print("PROCESSING IMAGES")
print("=" * 60)

for class_name in class_names:

    input_class_path = os.path.join(
        RAW_DATASET,
        class_name
    )

    output_class_path = os.path.join(
        PROCESSED_DATASET,
        class_name
    )

    os.makedirs(output_class_path, exist_ok=True)

    processed_count = 0
    failed_count = 0

    # Get image files
    image_files = [
        file for file in os.listdir(input_class_path)
        if file.lower().endswith(IMAGE_EXTENSIONS)
    ]

    print(f"\n[{class_name}] Found {len(image_files)} images")

    for filename in image_files:

        input_path = os.path.join(
            input_class_path,
            filename
        )

        # Save everything as JPG
        output_filename = os.path.splitext(filename)[0] + ".jpg"

        output_path = os.path.join(
            output_class_path,
            output_filename
        )

        try:

            # ------------------------------------------------
            # Open image
            # ------------------------------------------------

            image = Image.open(input_path)

            # ------------------------------------------------
            # Convert to RGB
            # ------------------------------------------------

            image = image.convert("RGB")

            # ------------------------------------------------
            # Resize to 128x128
            # ------------------------------------------------

            image = image.resize(
                IMG_SIZE,
                Image.Resampling.LANCZOS
            )

            # ------------------------------------------------
            # Save processed image
            # ------------------------------------------------

            image.save(
                output_path,
                "JPEG",
                quality=95
            )

            processed_count += 1

        except Exception as e:

            failed_count += 1

            print(
                f"  Failed: {filename} -> {e}"
            )

    total_processed += processed_count
    total_failed += failed_count

    print(
        f"  Processed: {processed_count} | "
        f"Failed: {failed_count}"
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE")
print("=" * 60)

print(f"\nClasses processed : {len(class_names)}")
print(f"Images processed  : {total_processed}")
print(f"Images failed     : {total_failed}")

print(
    f"\nProcessed dataset saved at:"
    f"\n{PROCESSED_DATASET}"
)

print("\nDataset structure:")
print("dataset/")
print("├── raw_asl/")
print("│   ├── A/")
print("│   ├── B/")
print("│   ├── C/")
print("│   └── ...")
print("│")
print("└── processed/")
print("    ├── A/")
print("    ├── B/")
print("    ├── C/")
print("    └── ...")

print("\nDone!")