import cv2
import mediapipe as mp
import os

# ==================== CONFIG ====================
RAW_DIR = "dataset/raw_asl"
PROCESSED_DIR = "dataset/processed"
IMG_SIZE = 128
# ================================================

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

if not os.path.exists(PROCESSED_DIR):
    os.makedirs(PROCESSED_DIR)

for class_name in os.listdir(RAW_DIR):
    class_path = os.path.join(RAW_DIR, class_name)
    
    if not os.path.isdir(class_path):
        continue
        
    save_path = os.path.join(PROCESSED_DIR, class_name)
    os.makedirs(save_path, exist_ok=True)
    
    processed_count = 0
    total_images = len(os.listdir(class_path))
    
    print(f"Processing class '{class_name}' ({total_images} images)...")

    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)
        
        if img is None:
            continue
            
        h, w, _ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            x_coords = [int(lm.x * w) for lm in results.multi_hand_landmarks[0].landmark]
            y_coords = [int(lm.y * h) for lm in results.multi_hand_landmarks[0].landmark]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            padding = 30
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)

            hand_crop = img[y_min:y_max, x_min:x_max]
            hand_resized = cv2.resize(hand_crop, (IMG_SIZE, IMG_SIZE))
            
            cv2.imwrite(os.path.join(save_path, img_name), hand_resized)
            processed_count += 1

    print(f" -> Successfully cropped and saved {processed_count}/{total_images} images.")

hands.close()
print("\n✅ Data preprocessing complete! Check the 'dataset/processed' folder.")