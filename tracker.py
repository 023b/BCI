# tracker.py - Extract movement/emotion data from landmarks

import cv2
import mediapipe as mp

class Tracker:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
        self.pose = mp.solutions.pose.Pose(static_image_mode=False)
        self.hands = mp.solutions.hands.Hands(static_image_mode=False)
        self.draw = mp.solutions.drawing_utils

    def analyze(self, frame):
        result = {}
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face = self.face_mesh.process(rgb)
        pose = self.pose.process(rgb)
        hands = self.hands.process(rgb)

        if face.multi_face_landmarks:
            result['face_landmarks'] = face.multi_face_landmarks
        if pose.pose_landmarks:
            result['pose_landmarks'] = pose.pose_landmarks
        if hands.multi_hand_landmarks:
            result['hand_landmarks'] = hands.multi_hand_landmarks

        return result

    def draw(self, frame, results):
        if 'face_landmarks' in results:
            for face in results['face_landmarks']:
                self.draw.draw_landmarks(
                    frame, face, mp.solutions.face_mesh.FACEMESH_TESSELATION)
        if 'pose_landmarks' in results:
            self.draw.draw_landmarks(
                frame, results['pose_landmarks'], mp.solutions.pose.POSE_CONNECTIONS)
        if 'hand_landmarks' in results:
            for hand in results['hand_landmarks']:
                self.draw.draw_landmarks(
                    frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
        return frame