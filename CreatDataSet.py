import os                                        #this lets us access the file system
import cv2                                       #this is the opencv library, it lets us access the camera and do image processing
import pandas as pd                              #
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Configure MediaPipe in IMAGE mode (for static photos, not live video)
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5
)

DATA_DIR = "dataset"
MAX_IMAGES_PER_LETTER = 400  # Limits to 400 images per letter so it finishes in ~2 minutes

all_rows = []
labels = []

print("Starting landmark extraction... Please wait.")

with vision.HandLandmarker.create_from_options(options) as landmarker:
    # Loop through each folder (A, B, C, ... Z)
    for letter in sorted(os.listdir(DATA_DIR)):
        letter_folder = os.path.join(DATA_DIR, letter)
        
        # Skip if it's not a folder
        if not os.path.isdir(letter_folder):
            continue

        print(f"Processing letter: {letter}")
        image_files = os.listdir(letter_folder)[:MAX_IMAGES_PER_LETTER]

        for img_name in image_files:
            img_path = os.path.join(letter_folder, img_name)
            frame = cv2.imread(img_path)

            if frame is None:
                continue

            # Convert BGR to SRGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Detect hand landmarks in the static photo
            result = landmarker.detect(mp_image)

            if result.hand_landmarks:
                hand_landmarks = result.hand_landmarks[0]

                # Collect all X and Y values first so we can normalize them
                x_coords = [lm.x for lm in hand_landmarks]
                y_coords = [lm.y for lm in hand_landmarks]

                min_x = min(x_coords)
                min_y = min(y_coords)

                # Store 42 normalized numbers (21 X's and 21 Y's) for this image
                row = []
                for lm in hand_landmarks:
                    row.append(lm.x - min_x)
                    row.append(lm.y - min_y)

                all_rows.append(row)
                labels.append(letter)

# 2. Save everything into a clean CSV spreadsheet using Pandas
df = pd.DataFrame(all_rows)
df["label"] = labels
df.to_csv("asl_data.csv", index=False)

print(f"Done! Saved {len(df)} hand samples to asl_data.csv.")
