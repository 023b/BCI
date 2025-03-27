# camera_phone.py - Real-Time Phone Camera with Imutils Stream Fix

import cv2
import threading
import mediapipe as mp
from imutils.video import VideoStream
import time

class PhoneCamera:
    def __init__(self, stream_url="http://localhost:8080/video"):
        self.stream_url = stream_url
        self.capture = VideoStream(src=self.stream_url).start()
        self.frame = None
        self.processed_frame = None
        self.running = False

        self.mp_face = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.draw = mp.solutions.drawing_utils

    def start(self):
        self.running = True
        threading.Thread(target=self._capture_loop, daemon=True).start()
        threading.Thread(target=self._process_loop, daemon=True).start()

    def _capture_loop(self):
        while self.running:
            frame = self.capture.read()
            if frame is not None:
                self.frame = frame

    def _process_loop(self):
        while self.running:
            if self.frame is not None:
                frame_copy = self.frame.copy()
                rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
                face = self.mp_face.process(rgb)
                hands = self.mp_hands.process(rgb)
                pose = self.mp_pose.process(rgb)

                if face.multi_face_landmarks:
                    for landmarks in face.multi_face_landmarks:
                        self.draw.draw_landmarks(
                            frame_copy, landmarks, mp.solutions.face_mesh.FACEMESH_TESSELATION)

                if hands.multi_hand_landmarks:
                    for hand in hands.multi_hand_landmarks:
                        self.draw.draw_landmarks(
                            frame_copy, hand, mp.solutions.hands.HAND_CONNECTIONS)

                if pose.pose_landmarks:
                    self.draw.draw_landmarks(
                        frame_copy, pose.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

                self.processed_frame = frame_copy

    def get_raw_frame(self):
        return self.frame

    def get_processed_frame(self):
        return self.processed_frame if self.processed_frame is not None else self.frame

    def stop(self):
        self.running = False
        self.capture.stop()

if __name__ == '__main__':
    cam = PhoneCamera()
    cam.start()
    try:
        while True:
            frame = cam.get_raw_frame()
            if frame is not None:
                cv2.imshow("Live Raw", cam.get_raw_frame())
                cv2.imshow("With Tracking", cam.get_processed_frame())
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cam.stop()
        cv2.destroyAllWindows()