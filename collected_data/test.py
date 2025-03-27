import cv2
import joblib
import time
import mediapipe as mp
import numpy as np

camera_url = "http://localhost:8080/video"
model = joblib.load("intention_model.joblib")

# Simulate EEG from visual cues
def simulate_eeg_from_pose(pose_landmarks):
    if not pose_landmarks:
        return np.zeros(7)

    return np.array([
        0.4,  # alpha
        0.6,  # beta
        0.3,  # theta
        0.2,  # gamma
        0.6 / (0.4 + 0.3 + 1e-6),  # engagement
        0.6 - 0.4,  # asymmetry
        0.05  # blink
    ])

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

cap = cv2.VideoCapture(camera_url)
while True:
    success, frame = cap.read()
    if not success:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(rgb)

    landmarks = []
    if pose_results.pose_landmarks:
        landmarks = pose_results.pose_landmarks.landmark

    eeg_features = simulate_eeg_from_pose(landmarks)
    prediction = model.predict([eeg_features])[0]

    # Display prediction
    cv2.putText(frame, f"Predicted Intention: {prediction}", (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Live Intention Predictor", frame)
    if cv2.waitKey(1) == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
