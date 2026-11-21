import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

#this code will load google's hand landmarker model.
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")


options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5
)

cap = cv2.VideoCapture(0)                             #cap mean capture, we made a variable called cap and assigned the default camera in windows which is 0.
frame_timestamp_ms = 0

# Start the hand landmarker and keep it open while the camera runs
with vision.HandLandmarker.create_from_options(options) as landmarker:
    while True:
        success, frame = cap.read()
        if not success:
            print("Could not read from camera.")
            break

        # Mirror the frame
        frame = cv2.flip(frame, 1)

        # Convert OpenCV colors (BGR) to MediaPipe colors (RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Advance timer by ~33ms (30 fps) and ask MediaPipe to find hands
        frame_timestamp_ms += 33
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        # If a hand is detected, loop through its 21 landmarks and draw green dots
        if result.hand_landmarks:
            height, width, _ = frame.shape
            for hand_landmarks in result.hand_landmarks:
                for landmark in hand_landmarks:
                    # Convert decimal coordinates (0.0 to 1.0) into screen pixels
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    cv2.circle(frame, (x, y), 6, (0, 255, 0), -1)

        cv2.imshow("MediaPipe Hand Tracking - Press Q to Quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
