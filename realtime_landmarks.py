"""Real-time ASL recognition with MediaPipe landmarks and the ANN classifier."""

import json
import os
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

from config import CAMERA_INDEX, CONFIDENCE_THRESHOLD, MODELS_DIR
from landmark_features import (
    NO_HAND_LABEL,
    landmarks_from_mediapipe,
    normalize_landmarks,
)

MODEL_PATH = os.path.join(MODELS_DIR, "landmarks_ann.keras")
LABELS_PATH = os.path.join(MODELS_DIR, "landmarks_labels.json")
SMOOTHING_WINDOW = 8
BOX_PADDING_RATIO = 0.25


def get_hand_box(hand_landmarks, frame_width, frame_height):
    x_values = [point.x * frame_width for point in hand_landmarks.landmark]
    y_values = [point.y * frame_height for point in hand_landmarks.landmark]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    padding_x = (x_max - x_min) * BOX_PADDING_RATIO
    padding_y = (y_max - y_min) * BOX_PADDING_RATIO
    return (
        max(0, int(x_min - padding_x)),
        max(0, int(y_min - padding_y)),
        min(frame_width, int(x_max + padding_x)),
        min(frame_height, int(y_max + padding_y)),
    )


def main():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Landmark model not found: {MODEL_PATH}. Run prepare_landmarks_dataset.py "
            "and train_landmarks_ann.py first."
        )
    if not os.path.isfile(LABELS_PATH):
        raise FileNotFoundError(f"Landmark labels not found: {LABELS_PATH}")

    with open(LABELS_PATH, "r", encoding="utf-8") as labels_file:
        mapping = json.load(labels_file)
    class_names = [mapping[str(index)] for index in range(len(mapping))]

    print("Loading landmark ANN model...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("Model loaded. Press 'q' to quit.")

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    capture = cv2.VideoCapture(CAMERA_INDEX)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open webcam index {CAMERA_INDEX}.")

    recent_predictions = deque(maxlen=SMOOTHING_WINDOW)
    try:
        while True:
            success, frame = capture.read()
            if not success or frame is None:
                break
            frame_height, frame_width = frame.shape[:2]
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            label = NO_HAND_LABEL
            confidence = 0.0
            color = (0, 0, 255)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )
                landmarks = landmarks_from_mediapipe(
                    hand_landmarks, frame_width, frame_height
                )
                features = np.expand_dims(normalize_landmarks(landmarks), axis=0)
                probabilities = model(features, training=False).numpy()[0]
                predicted_index = int(np.argmax(probabilities))
                confidence = float(probabilities[predicted_index])
                recent_predictions.append((predicted_index, confidence))

                votes = np.bincount(
                    [index for index, _ in recent_predictions],
                    minlength=len(class_names),
                )
                smoothed_index = int(np.argmax(votes))
                confidence = float(
                    np.mean(
                        [
                            score
                            for index, score in recent_predictions
                            if index == smoothed_index
                        ]
                    )
                )
                label = class_names[smoothed_index]
                color = (
                    (40, 200, 100)
                    if confidence >= CONFIDENCE_THRESHOLD
                    else (0, 165, 255)
                )
                x_min, y_min, x_max, y_max = get_hand_box(
                    hand_landmarks, frame_width, frame_height
                )
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            else:
                recent_predictions.clear()

            suffix = "" if confidence >= CONFIDENCE_THRESHOLD else " ?"
            display = "No hand detected" if label == NO_HAND_LABEL and not results.multi_hand_landmarks else (
                f"{label} ({confidence * 100:.1f}%){suffix}"
            )
            cv2.putText(
                frame,
                display,
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                color,
                3,
                cv2.LINE_AA,
            )
            cv2.imshow("ASL Landmarks + ANN", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
        hands.close()


if __name__ == "__main__":
    main()
