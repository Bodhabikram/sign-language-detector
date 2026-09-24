import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import deque

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/mobilenetv2_stage1_final.keras"

IMG_SIZE = (128, 128)

# Same order used during preprocessing/training - do not change
CLASS_NAMES = [
    "A", "B", "C", "D", "E", "F", "G",
    "H", "I", "J", "K", "L", "M", "N",
    "O", "P", "Q", "R", "S", "T", "U",
    "V", "W", "X", "Y", "Z",
    "nothing",
    "space"
]

# Minimum confidence to display a prediction as "recognized"
CONFIDENCE_THRESHOLD = 0.75

# Padding added around MediaPipe's hand bounding box before cropping,
# as a fraction of the box size. Prevents fingertips being cut off.
BOX_PADDING_RATIO = 0.25

# Number of recent predictions to smooth over (reduces frame-to-frame
# jitter/flicker in the displayed label)
SMOOTHING_WINDOW = 8

# Show a separate window with the exact image being fed to the model.
# Useful for comparing live crops against training images visually.
SHOW_DEBUG_WINDOW = True


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("Loading trained model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.")


# ============================================================
# SETUP MEDIAPIPE HANDS
# ============================================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,          # single-hand fingerspelling; raise if you need two
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)


# ============================================================
# HELPER: GET PADDED BOUNDING BOX FROM LANDMARKS
# ============================================================

def get_hand_bounding_box(landmarks, frame_width, frame_height):
    """
    Converts MediaPipe's normalized (0-1) landmark coordinates into a
    pixel-space bounding box, pads it, and then forces it SQUARE
    (centered on the hand) before returning.

    Why square: cv2.resize() to a square IMG_SIZE will stretch/distort
    a non-square crop, warping hand proportions the model never saw
    during training (Kaggle training images are effectively square
    hand crops). Squaring the box here, before resize, avoids that
    distortion entirely - no letterboxing needed.
    """
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

    # --------------------------------------------------------
    # Force square: expand the shorter side to match the longer
    # side, keeping the box centered on the hand.
    # --------------------------------------------------------
    box_w = x_max - x_min
    box_h = y_max - y_min
    side = max(box_w, box_h)

    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2

    x_min = center_x - side / 2
    x_max = center_x + side / 2
    y_min = center_y - side / 2
    y_max = center_y + side / 2

    # Clamp to frame bounds (may slightly break squareness at edges,
    # which is unavoidable - hand too close to frame edge)
    x_min = max(0, int(x_min))
    y_min = max(0, int(y_min))
    x_max = min(frame_width, int(x_max))
    y_max = min(frame_height, int(y_max))

    return x_min, y_min, x_max, y_max


# ============================================================
# HELPER: PREPROCESS CROPPED HAND FOR THE MODEL
# ============================================================

def preprocess_hand_crop(hand_crop_bgr):
    """
    Takes a BGR crop from OpenCV, converts to RGB, resizes to the
    model's expected input size, and keeps pixels in 0-255 range -
    matching data_loader.py, since preprocess_input is baked into
    the model graph itself (model.py).
    """
    rgb = cv2.cvtColor(hand_crop_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, IMG_SIZE)
    array = resized.astype(np.float32)  # keep 0-255, do NOT divide here
    array = np.expand_dims(array, axis=0)  # add batch dimension
    return array


# ============================================================
# MAIN LOOP
# ============================================================

def main():
    cap = cv2.VideoCapture(0)

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

        frame = cv2.flip(frame, 1)  # mirror for natural selfie-view
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
                hand_crop = frame[y_min:y_max, x_min:x_max]

                model_input = preprocess_hand_crop(hand_crop)

                if SHOW_DEBUG_WINDOW:
                    # model_input is RGB, 0-255 float, shape (1,128,128,3)
                    debug_view = model_input[0].astype(np.uint8)
                    debug_view_bgr = cv2.cvtColor(debug_view, cv2.COLOR_RGB2BGR)
                    # Upscale just for visibility, doesn't affect the model
                    debug_view_bgr = cv2.resize(debug_view_bgr, (256, 256), interpolation=cv2.INTER_NEAREST)
                    cv2.imshow("Model Input (what the model actually sees)", debug_view_bgr)

                predictions = model.predict(model_input, verbose=0)[0]

                predicted_index = int(np.argmax(predictions))
                confidence = float(predictions[predicted_index])

                recent_predictions.append(predicted_index)

                # Smooth: use the most common prediction in the recent window
                smoothed_index = max(
                    set(recent_predictions),
                    key=recent_predictions.count
                )

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                if confidence >= CONFIDENCE_THRESHOLD:
                    display_label = f"{CLASS_NAMES[smoothed_index]} ({confidence*100:.1f}%)"
                    display_color = (0, 255, 0)  # green
                else:
                    display_label = f"Uncertain ({confidence*100:.1f}%)"
                    display_color = (0, 165, 255)  # orange
            else:
                recent_predictions.clear()
        else:
            recent_predictions.clear()

        # --------------------------------------------------------
        # Draw prediction text on frame
        # --------------------------------------------------------
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