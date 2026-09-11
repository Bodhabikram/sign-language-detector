import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os

MODEL_PATH = "models/best_sign_language_model.keras"
DATASET_DIR = "dataset/processed" # Must match where we trained from
IMG_SIZE = 128
CONFIDENCE_THRESHOLD = 0.75

model = tf.keras.models.load_model(MODEL_PATH)
class_names = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
print("Loaded classes:", class_names)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    display_text = "No hand detected"
    box_color = (128, 128, 128)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
            y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            padding = 30
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)

            hand_crop = frame[y_min:y_max, x_min:x_max]
            if hand_crop.size == 0: continue

            # CRITICAL: Preprocess EXACTLY like training data
            hand_img = cv2.resize(hand_crop, (IMG_SIZE, IMG_SIZE))
            hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
            hand_img = hand_img / 255.0
            hand_img = np.expand_dims(hand_img, axis=0)

            prediction = model.predict(hand_img, verbose=0)[0]
            confidence = float(np.max(prediction))
            predicted_idx = np.argmax(prediction)
            predicted_label = class_names[predicted_idx]

            if confidence >= CONFIDENCE_THRESHOLD:
                display_text = f"{predicted_label} ({confidence:.0%})"
                box_color = (0, 255, 0)
            else:
                display_text = "Unrecognized"
                box_color = (0, 0, 255)

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 2)
            cv2.putText(frame, display_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)

    cv2.putText(frame, f"System: {display_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
    cv2.putText(frame, "Press 'Q' to exit", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow("Real-Time Sign Language Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
