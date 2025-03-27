from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton
import sys

class BCIDashboard(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.eeg_label = QLabel("EEG Data Visualization", self)
        self.intent_label = QLabel("Predicted Intent: None", self)

        self.control_button = QPushButton('Toggle Simulation', self)
        self.control_button.clicked.connect(self.toggle_simulation)

        layout.addWidget(self.eeg_label)
        layout.addWidget(self.intent_label)
        layout.addWidget(self.control_button)

        self.setLayout(layout)
        self.setWindowTitle('Advanced BCI Dashboard')

    def toggle_simulation(self):
        print("Simulation toggled")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    dashboard = BCIDashboard()
    dashboard.show()
    sys.exit(app.exec_())
