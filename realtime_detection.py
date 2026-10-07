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
    max_num_hands=2,          # double-hand fingerspelling
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)


# HELPER: GET PADDED BOUNDING BOX FROM LANDMARKS

def get_hand_bounding_box(landmarks, frame_width, frame_height):
    
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

   
    box_w = x_max - x_min
    box_h = y_max - y_min
    side = max(box_w, box_h)

    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2

    x_min = center_x - side / 2
    x_max = center_x + side / 2
    y_min = center_y - side / 2
    y_max = center_y + side / 2

    
    x_min = max(0, int(x_min))
    y_min = max(0, int(y_min))
    x_max = min(frame_width, int(x_max))
    y_max = min(frame_height, int(y_max))

    return x_min, y_min, x_max, y_max


# HELPER: PREPROCESS CROPPED HAND FOR THE MODEL

def preprocess_camera_frame(frame_bgr):
    """Apply the same background and image-level preprocessing used for training."""
    processed_bgr = preprocess_single_image(
        frame_bgr,
        tracker=None,
        isolate_hand=True,
        enhance_levels=True,
    )
    rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
    array = rgb.astype(np.float32)  # keep 0-255; the model preprocesses internally
    array = np.expand_dims(array, axis=0)  # add batch dimension
    return array


# MAIN LOOP

def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    # Rolling window of recent predicted class indices, for smoothing
    recent_predictions = deque(maxlen=SMOOTHING_WINDOW)

    print("\nStarting real-time detection. Press 'q' to quit.\n")

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to read frame from webcam.")
            break

        # Keep the camera image unmirrored to match the orientation in the training set.
        model_frame = frame.copy()
        frame_height, frame_width = frame.shape[:2]

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        display_label = "No hand detected"
        display_color = (0, 0, 255)  # red

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            # Draw landmarks on frame for visual feedback
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            x_min, y_min, x_max, y_max = get_hand_bounding_box(
                hand_landmarks, frame_width, frame_height
            )

            # Skip degenerate boxes (can happen at frame edges)
            if x_max > x_min and y_max > y_min:
                # Training uses the preprocessed full image (not a landmark crop).
                # Use the untouched frame: drawing overlays would alter model input.
                model_input = preprocess_camera_frame(model_frame)

                if SHOW_DEBUG_WINDOW:
                    # model_input is RGB, 0-255 float, shape (1,224,224,3)
                    debug_view = model_input[0].astype(np.uint8)
                    debug_view_bgr = cv2.cvtColor(debug_view, cv2.COLOR_RGB2BGR)
                    # Upscale just for visibility, doesn't affect the model
                    debug_view_bgr = cv2.resize(debug_view_bgr, (256, 256), interpolation=cv2.INTER_NEAREST)
                    cv2.imshow("Model Input (what the model actually sees)", debug_view_bgr)

                predictions = model.predict(model_input, verbose=0)[0]

                predicted_index = int(np.argmax(predictions))
                confidence = float(predictions[predicted_index])

                recent_predictions.append((predicted_index, confidence))

                # Smooth the class and its confidence over the same recent predictions.
                vote_counts = np.bincount(
                    [index for index, _ in recent_predictions],
                    minlength=len(CLASS_NAMES),
                )
                smoothed_index = int(np.argmax(vote_counts))
                smoothed_confidence = float(np.mean([
                    score for index, score in recent_predictions
                    if index == smoothed_index
                ]))

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                if smoothed_confidence >= CONFIDENCE_THRESHOLD:
                    display_label = f"{CLASS_NAMES[smoothed_index]} ({smoothed_confidence*100:.1f}%)"
                    display_color = (0, 255, 0)  # green
                else:
                    display_label = f"Uncertain ({smoothed_confidence*100:.1f}%)"
                    display_color = (0, 165, 255)  # orange
            else:
                recent_predictions.clear()
        else:
            recent_predictions.clear()

        # Draw prediction text on frame
        cv2.putText(
            frame, display_label, (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, display_color, 3
        )

        cv2.imshow("Sign Language Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("\nStopped.")


if __name__ == "__main__":
    main()