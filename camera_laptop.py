# camera_laptop.py - Real-Time Laptop Camera with Tracking and Raw Feed

import cv2
import threading
import mediapipe as mp

class LaptopCamera:
    def __init__(self, camera_index=0):
        self.capture = cv2.VideoCapture(camera_index)
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
            ret, frame = self.capture.read()
            if ret:
                self.frame = frame

    def _process_loop(self):
        while self.running:
            if self.frame is not None:
                frame_copy = self.frame.copy()
                rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
                face = self.mp_face.process(rgb)
                pose = self.mp_pose.process(rgb)

                if face.multi_face_landmarks:
                    for landmarks in face.multi_face_landmarks:
                        self.draw.draw_landmarks(
                            frame_copy, landmarks, mp.solutions.face_mesh.FACEMESH_TESSELATION)

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
        self.capture.release()

if __name__ == '__main__':
    cam = LaptopCamera()
    cam.start()
    try:
        while True:
            raw = cam.get_raw_frame()
            processed = cam.get_processed_frame()
            if raw is not None:
                cv2.imshow("Laptop Raw", raw)
            if processed is not None:
                cv2.imshow("Laptop Tracked", processed)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cam.stop()
        cv2.destroyAllWindows()