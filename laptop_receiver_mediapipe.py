import requests
import base64
import cv2
import numpy as np
import mediapipe as mp

URL = 'http://localhost:5000/frame'

pose = mp.solutions.pose.Pose()
hands = mp.solutions.hands.Hands()

while True:
    try:
        res = requests.get(URL, timeout=1)
        data = res.json()
        frame_b64 = data['frame']
        if frame_b64 is None:
            continue

        img_bytes = base64.b64decode(frame_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Process with MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_result = pose.process(frame_rgb)
        hands_result = hands.process(frame_rgb)

        # Draw results
        if pose_result.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, pose_result.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS
            )
        if hands_result.multi_hand_landmarks:
            for hand_landmarks in hands_result.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS
                )

        cv2.imshow("Live Feed from Phone", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"[ERROR] {e}")

cv2.destroyAllWindows()
