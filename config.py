import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
RAW_ASL_DIR = os.path.join(DATA_DIR, "raw_asl")
PROCESSED_DIR = os.environ.get("ASL_PROCESSED_DIR", os.path.join(DATA_DIR, "processed"))
DATASET_MANIFEST_PATH = os.path.join(PROCESSED_DIR, "manifest.json")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.environ.get(
    "ASL_MODEL_PATH", os.path.join(MODELS_DIR, "mobilenetv2_stage1_final.keras")
)
LABELS_PATH = os.path.join(MODELS_DIR, "labels.json")
PLOT_PATH = os.path.join(MODELS_DIR, "training_history.png")

# Model & Preprocessing Parameters
IMG_SIZE = 160          # Smaller than 224 so CPU training stays feasible (~3x faster)
CHANNELS = 3
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-4    # Ideal for transfer learning fine-tuning
FINE_TUNE_EPOCHS = 8    # Unfreezing top base layers for high accuracy

# Real-Time Detection Parameters
CAMERA_INDEX = 0        # Default webcam
CONFIDENCE_THRESHOLD = 0.75  # Minimum probability to accept prediction
BOX_PADDING = 25        # Pixel padding around detected hand bounding box
PREDICTION_HISTORY_LEN = 10  # Smoothing window (majority vote over frames)