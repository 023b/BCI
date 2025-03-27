from simulation.eeg_simulator import EEGSimulator
from inference.intent_predictor import IntentPredictor
from utils.logger import setup_logger
from ui.dashboard import BCIDashboard
import sys
from PyQt5.QtWidgets import QApplication

def main():
    logger = setup_logger("bci_project")
    eeg_simulator = EEGSimulator()
    intent_predictor = IntentPredictor()

    # Simplified simulation loop example
    facial_features = [0.5, 0.7]
    posture_features = [0.2, 0.4]

    eeg_features = eeg_simulator.simulate_from_features(facial_features, posture_features)
    predicted_intent = intent_predictor.predict_intent(eeg_features)

    logger.info(f"Predicted Intent: {predicted_intent}")

    app = QApplication(sys.argv)
    dashboard = BCIDashboard()
    dashboard.intent_label.setText(f"Predicted Intent: {predicted_intent}")
    dashboard.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
