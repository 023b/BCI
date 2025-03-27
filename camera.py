# BCI Project - Main Code
# Step 1: Capture and process phone camera feed over USB

import cv2
import numpy as np
import threading
import time

class PhoneCamera:
    def __init__(self, stream_url):
        self.stream_url = stream_url
        self.frame = None
        self.capture = cv2.VideoCapture(self.stream_url)
        self.running = False

    def start(self):
        self.running = True
        thread = threading.Thread(target=self._update, daemon=True)
        thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.capture.read()
            if ret:
                self.frame = frame
            else:
                print("[Warning] Failed to read from camera stream")
            time.sleep(0.01)

    def get_frame(self):
        return self.frame

    def stop(self):
        self.running = False
        self.capture.release()


if __name__ == '__main__':
    # Step 1: Connect to phone camera via USB-forwarded stream
    stream_url = "http://localhost:8080/video"
    phone_cam = PhoneCamera(stream_url)
    phone_cam.start()

    print("[INFO] Starting phone camera stream processing...")

    try:
        while True:
            frame = phone_cam.get_frame()
            if frame is not None:
                cv2.imshow("Phone Camera", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("[INFO] Stopped by user")
    finally:
        phone_cam.stop()
        cv2.destroyAllWindows()
