import cv2
import time
import json
import os
from datetime import datetime
from flask import Flask, render_template_string, request, Response
import mediapipe as mp
import threading

# === CONFIG ===
SAVE_DIR = "collected_data_full"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

camera_url = "http://localhost:8080/video"  # Replace with your phone's stream

app = Flask(__name__)

# MediaPipe setup with safer parameters
mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
face_mesh = mp_face.FaceMesh(static_image_mode=False, refine_landmarks=True, max_num_faces=1)
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)

# === Label Sets and Simulations ===
LABELS = {
    "intention": {
        "focused": "Actively concentrating on task.",
        "distracted": "Attention drifting externally or internally.",
        "trying_to_remember": "Memory recall effort, gaze aversion, stillness.",
        "planning": "Deep internal focus, slow movement, gaze fixed.",
        "internally_debating": "Micro-expression switching, uncertainty.",
        "decision_making": "Hesitant motion, pause-and-go behavior.",
        "attempting_to_engage": "Gaze returning to camera, head tilt forward.",
        "mind_wandering": "Gaze unfocused, passive expressions.",
        "mentally_multitasking": "Shifting gaze, lip biting, scanning."
    },
    "emotion": {
        "frustrated": "Tense jaw, frown, restless.",
        "angry": "Sharp eye movement, brow tension.",
        "calm": "Minimal movement, neutral expression.",
        "confused": "Tilted head, furrowed brow, pauses.",
        "interested": "Leaning in, eyes wide open.",
        "bored": "Slouching, frequent blinks, eye roll.",
        "surprised": "Eyebrow lift, blink spike."
    },
    "gesture": {
        "leaning_forward": "Lean your upper body toward the camera.",
        "leaning_back": "Lean your upper body away from the camera.",
        "looking_away": "Turn your eyes or face to the side.",
        "looking_up": "Glance upwards, not at the camera.",
        "looking_down": "Lower your gaze or tilt your head down.",
        "resting_chin": "Place your chin on your hand.",
        "biting_lip": "Bite or press your lips together.",
        "head_shake": "Gently shake your head left and right.",
        "head_nod": "Nod your head slowly.",
        "fidgeting": "Move your hands or fingers while idle.",
        "hands_on_face": "Touch your face with one or both hands.",
        "arms_crossed": "Cross your arms across your chest.",
        "sighing": "Exhale noticeably or drop your shoulders."
    }
}

EEG_SIM = {
    "focused": {"alpha": 0.3, "beta": 0.7, "theta": 0.2, "gamma": 0.3},
    "distracted": {"alpha": 0.7, "beta": 0.3, "theta": 0.4, "gamma": 0.2},
    "trying_to_remember": {"alpha": 0.4, "beta": 0.4, "theta": 0.7, "gamma": 0.3},
    "planning": {"alpha": 0.3, "beta": 0.6, "theta": 0.6, "gamma": 0.3},
    "internally_debating": {"alpha": 0.5, "beta": 0.5, "theta": 0.7, "gamma": 0.4},
    "decision_making": {"alpha": 0.4, "beta": 0.6, "theta": 0.6, "gamma": 0.6},
    "attempting_to_engage": {"alpha": 0.5, "beta": 0.6, "theta": 0.4, "gamma": 0.3},
    "mind_wandering": {"alpha": 0.7, "beta": 0.2, "theta": 0.3, "gamma": 0.2},
    "mentally_multitasking": {"alpha": 0.4, "beta": 0.5, "theta": 0.4, "gamma": 0.7},
    "confused": {"alpha": 0.5, "beta": 0.4, "theta": 0.6, "gamma": 0.3},
    "interested": {"alpha": 0.4, "beta": 0.7, "theta": 0.3, "gamma": 0.2},
    "bored": {"alpha": 0.8, "beta": 0.2, "theta": 0.4, "gamma": 0.1},
    "calm": {"alpha": 0.7, "beta": 0.3, "theta": 0.3, "gamma": 0.1},
    "angry": {"alpha": 0.2, "beta": 0.6, "theta": 0.3, "gamma": 0.6},
    "frustrated": {"alpha": 0.3, "beta": 0.8, "theta": 0.3, "gamma": 0.2},
    "surprised": {"alpha": 0.5, "beta": 0.6, "theta": 0.4, "gamma": 0.7}
}

latest_frame = None

def camera_loop():
    global latest_frame
    cap = cv2.VideoCapture(camera_url)
    while True:
        success, frame = cap.read()
        if success and frame is not None and frame.shape[0] > 0:
            latest_frame = frame

threading.Thread(target=camera_loop, daemon=True).start()

@app.route('/')
def index():
    default_type = "intention"
    default_label = list(LABELS[default_type].keys())[0]
    default_description = LABELS[default_type][default_label]

    return render_template_string('''
    <html>
    <head><title>🧠 Multi-Label BCI Collector</title></head>
    <body style="text-align:center;font-family:sans-serif;">
        <h2>🎥 Record 5s Video Sample</h2>
        <form method="POST" action="/record_video">
            <label>Label Type:</label>
            <select name="label_type" id="label_type" onchange="updateOptions()">
                {% for k in labels %}<option value="{{ k }}">{{ k }}</option>{% endfor %}
            </select><br><br>
            <label>Label:</label>
            <select name="label_value" id="label_value">
                {% for k in labels['intention'] %}<option value="{{ k }}">{{ k }}</option>{% endfor %}
            </select><br>
            <p id="description">{{ default_description }}</p><br>
            <button type="submit">🎬 Record</button>
        </form>
        <div><img src="/video_feed" width="640" height="480"></div>
        <script>
            const labelMap = {{ labels | tojson }};
            function updateOptions() {
                const type = document.getElementById("label_type").value;
                const valueSelect = document.getElementById("label_value");
                valueSelect.innerHTML = "";
                for (let label in labelMap[type]) {
                    const opt = document.createElement("option");
                    opt.value = label;
                    opt.innerText = label;
                    valueSelect.appendChild(opt);
                }
                document.getElementById("description").innerText = labelMap[type][Object.keys(labelMap[type])[0]];
            }
            document.getElementById("label_value").addEventListener("change", function() {
                const type = document.getElementById("label_type").value;
                const label = this.value;
                document.getElementById("description").innerText = labelMap[type][label];
            });
        </script>
    </body>
    </html>
    ''', labels=LABELS, default_description=default_description)

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            if latest_frame is not None:
                _, buffer = cv2.imencode('.jpg', latest_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.03)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/record_video', methods=['POST'])
def record_video():
    global latest_frame
    label_type = request.form["label_type"]
    label_value = request.form["label_value"]
    duration = 5
    fps = 30
    total_frames = duration * fps
    frame_interval = 1 / fps
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i in range(total_frames):
        if latest_frame is None or latest_frame.shape[0] == 0:
            continue

        frame = latest_frame.copy()
        start = time.time()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        entry = {
            "timestamp": timestamp,
            "label_type": label_type,
            "label": label_value,
            "visual": {},
            "simulated_eeg": {},
            "system_diagnostics": {}
        }

        face_result = face_mesh.process(rgb)
        if face_result.multi_face_landmarks:
            entry["visual"]["face_landmarks"] = [
                {"x": lm.x, "y": lm.y, "z": lm.z}
                for lm in face_result.multi_face_landmarks[0].landmark
            ]

        pose_result = pose.process(rgb)
        if pose_result.pose_landmarks:
            entry["visual"]["pose_landmarks"] = [
                {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
                for lm in pose_result.pose_landmarks.landmark
            ]

        eeg = EEG_SIM.get(label_value, {})
        alpha, beta, theta = eeg.get("alpha", 0), eeg.get("beta", 0), eeg.get("theta", 0)
        entry["simulated_eeg"] = {
            "alpha": alpha,
            "beta": beta,
            "theta": theta,
            "gamma": eeg.get("gamma", 0),
            "engagement_index": round(beta / (alpha + theta + 1e-6), 3),
            "frontal_asymmetry": round(beta - alpha, 3),
            "blink_artifact": 0.05
        }

        entry["system_diagnostics"] = {
            "frame_number": i,
            "latency_ms": round((time.time() - start) * 1000, 2),
            "capture_time": datetime.now().isoformat()
        }

        filename = f"{label_type}_{label_value}_{timestamp}_frame_{str(i).zfill(3)}.json"
        path = os.path.join(SAVE_DIR, filename)
        with open(path, 'w') as f:
            json.dump(entry, f, indent=2)

        time.sleep(frame_interval)

    return f"✅ {total_frames} frames recorded for '{label_type}:{label_value}'"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
