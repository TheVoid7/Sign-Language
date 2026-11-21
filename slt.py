import pickle
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Load your trained Random Forest model from model.p
with open("model.p", "rb") as f:
    model_dict = pickle.load(f)
model = model_dict["model"]

# 2. Set up Google MediaPipe HandLandmarker in VIDEO mode
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5
)

cap = cv2.VideoCapture(0)
frame_timestamp_ms = 0

with vision.HandLandmarker.create_from_options(options) as landmarker:
    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame from camera.")
            break

        # Mirror the frame
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape

        # Convert BGR to SRGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        frame_timestamp_ms += 33
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]

            # Collect X and Y coordinates to draw dots AND normalize for the model
            x_coords = [lm.x for lm in hand_landmarks]
            y_coords = [lm.y for lm in hand_landmarks]

            min_x = min(x_coords)
            min_y = min(y_coords)

            row = []
            for lm in hand_landmarks:
                # Draw the 21 green dots on your hand
                px = int(lm.x * width)
                py = int(lm.y * height)
                cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)

                # Normalize coordinates the exact same way CreatDataSet.py did
                row.append(lm.x - min_x)
                row.append(lm.y - min_y)

            # 3. Ask your trained model to predict the letter!
            prediction = model.predict([np.asarray(row)])
            predicted_letter = prediction[0]

            # 4. Draw a bounding box and the predicted letter above your hand
            x1 = int(min_x * width) - 20
            y1 = int(min_y * height) - 20
            x2 = int(max(x_coords) * width) + 20
            y2 = int(max(y_coords) * height) + 20

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                predicted_letter,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 0),
                3,
                cv2.LINE_AA
            )

        cv2.imshow("ASL Translator (Terminal Prototype) - Press Q to Quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
