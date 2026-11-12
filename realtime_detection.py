import json
import os

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import deque
from config import CAMERA_INDEX, LABELS_PATH, MODEL_PATH
from preprocess_data import preprocess_single_image

# CONFIGURATION

if not os.path.isfile(LABELS_PATH):
    raise FileNotFoundError("Class labels are missing. Run preprocessing before inference.")
with open(LABELS_PATH, "r", encoding="utf-8") as labels_file:
    label_mapping = json.load(labels_file)
CLASS_NAMES = [label_mapping[str(index)] for index in range(len(label_mapping))]

# Minimum confidence to display a prediction as "recognized"
CONFIDENCE_THRESHOLD = 0.75

# Padding added around MediaPipe's hand bounding box before cropping
BOX_PADDING_RATIO = 0.25

# Number of recent predictions to smooth over
SMOOTHING_WINDOW = 8

# Show a separate window with the exact image being fed to the model.
SHOW_DEBUG_WINDOW = True


# LOAD TRAINED MODEL

print("Loading trained model...")
model = tf.keras.models.load_model(MODEL_PATH)
if model.output_shape[-1] != len(CLASS_NAMES):
    raise ValueError(
        f"Model has {model.output_shape[-1]} outputs, but labels.json has "
        f"{len(CLASS_NAMES)} classes. Retrain the model after preprocessing."
    )
print("Model loaded.")


# SETUP MEDIAPIPE HANDS

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)


def get_hand_bounding_box(landmarks, frame_width, frame_height):
    """Calculate a padded, square pixel bounding box around hand landmarks."""
    x_coords = [lm.x * frame_width for lm in landmarks.landmark]
    y_coords = [lm.y * frame_height for lm in landmarks.landmark]

    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    box_w = x_max - x_min
    box_h = y_max - y_min
    pad_x = box_w * BOX_PADDING_RATIO
    pad_y = box_h * BOX_PADDING_RATIO

    x_min -= pad_x
    x_max += pad_x
    y_min -= pad_y
    y_max += pad_y

    side = max(x_max - x_min, y_max - y_min)
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2

    x_min = max(0, int(center_x - side / 2))
    y_min = max(0, int(center_y - side / 2))
    x_max = min(frame_width, int(center_x + side / 2))
    y_max = min(frame_height, int(center_y + side / 2))
    return x_min, y_min, x_max, y_max


def preprocess_camera_frame(hand_crop_bgr):
    """Preprocess a cropped hand region with the training image transforms."""
    processed_bgr = preprocess_single_image(
        hand_crop_bgr,
        tracker=None,
        isolate_hand=True,
        enhance_levels=True,
    )
    rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
    array = rgb.astype(np.float32)  # keep 0-255; the model preprocesses internally
    return np.expand_dims(array, axis=0)


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    recent_predictions = deque(maxlen=SMOOTHING_WINDOW)
    print("\nStarting real-time detection. Press 'q' to quit.\n")

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Failed to read frame from webcam.")
                break

            # Keep inference pixels clean and unmirrored; overlays are display-only.
            model_frame = frame.copy()
            frame_height, frame_width = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            display_label = "No hand detected"
            display_color = (0, 0, 255)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                )
                x_min, y_min, x_max, y_max = get_hand_bounding_box(
                    hand_landmarks,
                    frame_width,
                    frame_height,
                )

                if x_max > x_min and y_max > y_min:
                    hand_crop = model_frame[y_min:y_max, x_min:x_max]
                    model_input = preprocess_camera_frame(hand_crop)

                    if SHOW_DEBUG_WINDOW:
                        debug_view = model_input[0].astype(np.uint8)
                        debug_view_bgr = cv2.cvtColor(debug_view, cv2.COLOR_RGB2BGR)
                        debug_view_bgr = cv2.resize(
                            debug_view_bgr,
                            (256, 256),
                            interpolation=cv2.INTER_NEAREST,
                        )
                        cv2.imshow(
                            "Model Input (what the model actually sees)",
                            debug_view_bgr,
                        )

                    predictions = model.predict(model_input, verbose=0)[0]
                    predicted_index = int(np.argmax(predictions))
                    confidence = float(predictions[predicted_index])
                    recent_predictions.append((predicted_index, confidence))

                    vote_counts = np.bincount(
                        [index for index, _ in recent_predictions],
                        minlength=len(CLASS_NAMES),
                    )
                    smoothed_index = int(np.argmax(vote_counts))
                    smoothed_confidence = float(np.mean([
                        score
                        for index, score in recent_predictions
                        if index == smoothed_index
                    ]))

                    cv2.rectangle(
                        frame,
                        (x_min, y_min),
                        (x_max, y_max),
                        (0, 255, 0),
                        2,
                    )
                    confidence_note = (
                        "" if smoothed_confidence >= CONFIDENCE_THRESHOLD else " ?"
                    )
                    display_label = (
                        f"{CLASS_NAMES[smoothed_index]} "
                        f"({smoothed_confidence * 100:.1f}%)"
                        f"{confidence_note}"
                    )
                    display_color = (
                        (0, 255, 0)
                        if smoothed_confidence >= CONFIDENCE_THRESHOLD
                        else (0, 165, 255)
                    )
                else:
                    recent_predictions.clear()
            else:
                recent_predictions.clear()

            cv2.putText(
                frame,
                display_label,
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                display_color,
                3,
            )
            cv2.imshow("Sign Language Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()

    print("\nStopped.")


if __name__ == "__main__":
    main()
