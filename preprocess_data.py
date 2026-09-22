# preprocessing code for asl project
# basically converts all the hand images into landmark points using mediapipe
# then saves it as train/test data for the model

import os
import cv2
import mediapipe as mp
import numpy as np
import json
import random

DATA_DIR = 'dataset/raw_asl'
SAVE_DIR = "dataset/processed"

# make the save folder if it doesnt exist yet
if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

classes = os.listdir(DATA_DIR)
classes.sort()  # sorting so labels stay consistent everytime we run this
print("classes found:", classes)
print("total classes:", len(classes))

data = []
labels = []

skipped = 0
count_done = 0  # just to keep track how many images we actually processed

# main loop, goes through every class folder one by one
for label in range(len(classes)):

    folder_path = os.path.join(DATA_DIR, classes[label])
    images = os.listdir(folder_path)

    print("----")
    print("class:", classes[label], " label num:", label)
    print("images in this folder:", len(images))

    for img_name in images:

        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)

        # sometimes an image doesnt load properly so we skip it
        if img is None:
            skipped = skipped + 1
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # mediapipe needs RGB
        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:

            hand = result.multi_hand_landmarks[0]  # only using first hand

            landmark_list = []
            for point in hand.landmark:
                landmark_list.append(point.x)
                landmark_list.append(point.y)
                landmark_list.append(point.z)

            # normalizing - subtracting wrist point (first landmark) from
            # all the other points so the position on screen doesnt matter
            wrist_x = landmark_list[0]
            wrist_y = landmark_list[1]
            wrist_z = landmark_list[2]

            for i in range(0, len(landmark_list), 3):
                landmark_list[i] = landmark_list[i] - wrist_x
                landmark_list[i+1] = landmark_list[i+1] - wrist_y
                landmark_list[i+2] = landmark_list[i+2] - wrist_z

            data.append(landmark_list)
            labels.append(label)
            count_done += 1

        else:
            # no hand detected in the image, skip it
            skipped = skipped + 1

print("----")
print("done processing all folders")
print("total used:", len(data))
print("total skipped:", skipped)
# print(len(data) == count_done)   # just checking, should be true lol

X = np.array(data)
y = np.array(labels)

# ----- shuffle -----
# doing it manually with indexes instead of using sklearn train_test_split
# cuz i didnt wanna install another library for just this

indexes = []
for i in range(len(X)):
    indexes.append(i)

random.shuffle(indexes)

X_shuffled = []
y_shuffled = []

for i in indexes:
    X_shuffled.append(X[i])
    y_shuffled.append(y[i])

X = np.array(X_shuffled)
y = np.array(y_shuffled)

# ----- train test split (80/20) -----
split = int(len(X) * 0.8)

X_train = X[:split]
y_train = y[:split]
X_test = X[split:]
y_test = y[split:]

print("train size:", len(X_train))
print("test size:", len(X_test))

# saving all the numpy files
np.save(os.path.join(SAVE_DIR, "X_train.npy"), X_train)
np.save(os.path.join(SAVE_DIR, "y_train.npy"), y_train)
np.save(os.path.join(SAVE_DIR, "X_test.npy"), X_test)
np.save(os.path.join(SAVE_DIR, "y_test.npy"), y_test)

# saving label mapping as a picture (table image) instead of json
import matplotlib.pyplot as plt

label_dict = {}
count = 0
for class_name in classes:
    label_dict[count] = class_name
    count = count + 1

table_data = []
for key in label_dict:
    table_data.append([str(key), label_dict[key]])

fig, ax = plt.subplots(figsize=(4, len(table_data) * 0.35))
ax.axis("off")

table = ax.table(
    cellText=table_data,
    colLabels=["Label #", "Class Name"],
    cellLoc="center",
    loc="center"
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.2)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "labels.png"), dpi=150, bbox_inches="tight")
plt.close()

print("all done, files saved in processed_data folder")