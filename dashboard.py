# dashboard.py - Full Custom PyQt5 BCI Control Interface (Upgraded w/ Data + Tracker)

import sys
import cv2
import psutil
import time
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QSlider, QTextEdit, QGridLayout, QGroupBox)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer, Qt
from camera_laptop import LaptopCamera
from camera_phone import PhoneCamera
from eeg_simulator import EEGSimulator
from fusion_engine import FusionEngine
from ollama_sender import OllamaSender
from tracker import Tracker

class BCIDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🧠 Real-Time BCI Control Panel")
        self.setGeometry(100, 100, 1600, 900)

        self.phone = PhoneCamera()
        self.laptop = LaptopCamera()
        self.eeg = EEGSimulator()
        self.fusion = FusionEngine()
        self.ollama = OllamaSender()
        self.tracker = Tracker()

        self._setup_ui()
        self.phone.start()
        self.laptop.start()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_dashboard)
        self.timer.start(30)

    def _setup_ui(self):
        main = QWidget()
        layout = QGridLayout()

        # Video Feeds
        self.label_phone = QLabel("Phone Camera")
        self.label_laptop = QLabel("Laptop Camera")

        self.label_phone.setFixedSize(640, 360)
        self.label_laptop.setFixedSize(640, 360)

        # EEG + Prediction + Controls
        self.eeg_label = QLabel("EEG Stats: ")
        self.intent_label = QLabel("Intent: -")
        self.engagement_label = QLabel("Engagement: -")
        self.confidence_label = QLabel("Confidence: -")

        font = QFont("Courier", 10)
        for l in [self.eeg_label, self.intent_label, self.engagement_label, self.confidence_label]:
            l.setFont(font)

        # Control Sliders
        self.ratio_slider = QSlider(Qt.Horizontal)
        self.ratio_slider.setRange(0, 100)
        self.ratio_slider.setValue(50)

        # LLM Reply Box
        self.reply_box = QTextEdit()
        self.reply_box.setPlaceholderText("LLM Replies...")
        self.reply_box.setReadOnly(True)

        # Buttons
        self.llm_button = QPushButton("Send to LLM")
        self.llm_button.clicked.connect(self.send_to_llm)

        # Group Boxes
        feed_box = QGroupBox("Camera Feeds")
        stat_box = QGroupBox("Mental State & EEG")
        control_box = QGroupBox("Controls & LLM")

        # Feed layout
        feed_layout = QHBoxLayout()
        feed_layout.addWidget(self.label_laptop)
        feed_layout.addWidget(self.label_phone)
        feed_box.setLayout(feed_layout)

        # Stats layout
        stat_layout = QVBoxLayout()
        stat_layout.addWidget(self.eeg_label)
        stat_layout.addWidget(self.intent_label)
        stat_layout.addWidget(self.engagement_label)
        stat_layout.addWidget(self.confidence_label)
        stat_box.setLayout(stat_layout)

        # Control layout
        ctrl_layout = QVBoxLayout()
        ctrl_layout.addWidget(QLabel("EEG vs Visual Ratio"))
        ctrl_layout.addWidget(self.ratio_slider)
        ctrl_layout.addWidget(self.llm_button)
        ctrl_layout.addWidget(self.reply_box)
        control_box.setLayout(ctrl_layout)

        # Final Layout
        layout.addWidget(feed_box, 0, 0, 1, 2)
        layout.addWidget(stat_box, 1, 0)
        layout.addWidget(control_box, 1, 1)

        main.setLayout(layout)
        self.setCentralWidget(main)

    def update_dashboard(self):
        pf = self.phone.get_processed_frame()
        lf = self.laptop.get_processed_frame()

        if pf is not None:
            face_data = self.tracker.analyze(pf)
            pf = self.tracker.draw(pf, face_data)
            self.label_phone.setPixmap(self.convert(pf))

        if lf is not None:
            pose_data = self.tracker.analyze(lf)
            lf = self.tracker.draw(lf, pose_data)
            self.label_laptop.setPixmap(self.convert(lf))

        eeg_df = self.eeg.generate()
        eeg_data = eeg_df.iloc[-1].to_dict()

        cam_data = {
            "emotion_level": 0.7 if face_data else 0.4,
            "gaze_centered": True if face_data else False,
            "emotion": "neutral"
        }

        ratio = self.ratio_slider.value() / 100.0
        self.fusion.set_weights(ratio)
        fused = self.fusion.fuse(eeg_data, cam_data)

        self.eeg_label.setText(f"EEG: a={eeg_data['alpha']:.2f}, b={eeg_data['beta']:.2f}, mu={eeg_data['mu']:.2f}")
        self.intent_label.setText(f"Intent: {fused['intent']}")
        self.engagement_label.setText(f"Engagement: {fused['engagement']:.2f}")
        self.confidence_label.setText(f"Confidence: {fused['confidence']:.2f}")
        self.fused_state = fused

    def send_to_llm(self):
        if hasattr(self, 'fused_state'):
            reply = self.ollama.send(self.fused_state)
            self.reply_box.append("LLM: " + reply)

    def convert(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        return QPixmap.fromImage(QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888))

    def closeEvent(self, event):
        self.phone.stop()
        self.laptop.stop()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = BCIDashboard()
    win.show()
    sys.exit(app.exec_())
