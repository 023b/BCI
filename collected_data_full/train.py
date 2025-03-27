import cv2
import numpy as np
import joblib
import mediapipe as mp

# Load trained model
model = joblib.load("bci_multilabel_model.joblib")

# MediaPipe setup (optional for future pose/gesture logic)
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False, refine_landmarks=True, max_num_faces=1)

# EEG simulation function
def simulate_eeg():
    return {
        "alpha": np.random.uniform(0.3, 0.7),
        "beta": np.random.uniform(0.3, 0.7),
        "theta": np.random.uniform(0.2, 0.6),
        "gamma": np.random.uniform(0.1, 0.5)
    }

def compute_features(eeg):
    alpha = eeg["alpha"]
    beta = eeg["beta"]
    theta = eeg["theta"]
    gamma = eeg["gamma"]
    engagement_index = beta / (alpha + theta + 1e-6)
    asymmetry = beta - alpha
    blink_artifact = np.random.uniform(0.03, 0.08)
    return [alpha, beta, theta, gamma, engagement_index, asymmetry, blink_artifact]

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not access webcam.")
    exit()

print("✅ Live predictor running. Press [Q] to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame capture failed.")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    _ = face_mesh.process(rgb)  # Optional: we could expand this later

    eeg = simulate_eeg()
    features = compute_features(eeg)
    prediction = model.predict([features])[0]

    label_text = f"🧠 Prediction: {prediction}"
    cv2.putText(frame, label_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 100), 2)
    cv2.imshow("BCI Live Predictor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Live session ended.")
