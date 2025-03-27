# fusion_engine.py - Combine EEG and visual insights into structured state

import numpy as np

class FusionEngine:
    def __init__(self):
        self.eeg_weight = 0.5  # start with equal reliance
        self.cam_weight = 0.5

    def set_weights(self, eeg_ratio):
        self.eeg_weight = eeg_ratio
        self.cam_weight = 1.0 - eeg_ratio

    def fuse(self, eeg_data, cam_data):
        """
        eeg_data: dict with keys like attention, alpha, beta
        cam_data: dict with emotion, gaze, posture, etc
        """
        fused_state = {}

        # Example: combine attention and facial engagement
        attention = eeg_data.get("attention", 0.5)
        emotion = cam_data.get("emotion_level", 0.5)

        fused_state['engagement'] = round(
            self.eeg_weight * attention + self.cam_weight * emotion, 3)

        # You can add more fusion logic here
        fused_state['intent'] = self.infer_intent(eeg_data, cam_data)
        fused_state['confidence'] = np.clip(
            0.5 * attention + 0.5 * emotion, 0.0, 1.0)

        return fused_state

    def infer_intent(self, eeg, cam):
        # Dummy logic — extend this for your needs
        if eeg['attention'] > 0.7 and cam.get('gaze_centered', False):
            return "focused_action"
        elif eeg['alpha'] > eeg['beta']:
            return "relaxed"
        elif cam.get("emotion", "") == "sad":
            return "withdraw"
        else:
            return "idle"
