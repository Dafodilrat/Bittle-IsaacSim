from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QMessageBox, QSlider, QApplication
)
from PyQt5.QtCore import Qt

import subprocess
import json
import sys
import signal
import os 

class RLParamInputGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RL Parameter Input")
        self.default_weights = [100, 10, 10, 5, 2, 10]
        self.param_defs = [
            ("Correct Posture Bonus", 0, 100, self.default_weights[0]),
            ("Smooth Bonus Weight", 0, 100, self.default_weights[1]),
            ("Incorrect Posture Penalty", 0, 100, self.default_weights[2]),
            ("Jerking Movement Penalty (x10)", 0, 50, self.default_weights[3] * 10),
            ("High Joint Velocity Penalty (x10)", 0, 50, self.default_weights[4] * 10),
            ("Distance to Goal Penalty", 0, 100, self.default_weights[5]),
        ]
        self.sliders = []
        self.labels = []
        self.proc = None  # <-- Store the subprocess handle
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        for label_text, min_val, max_val, default in self.param_defs:
            hbox = QHBoxLayout()
            label = QLabel(f"{label_text}: {default / 10.0 if 'x10' in label_text else default}")
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(min_val)
            slider.setMaximum(max_val)
            slider.setValue(default)
            slider.setTickInterval(1)
            slider.setSingleStep(1)

            def update_label(val, label=label, text=label_text):
                scaled_val = val / 10.0 if "x10" in text else val
                label.setText(f"{text}: {scaled_val}")

            slider.valueChanged.connect(update_label)
            hbox.addWidget(label)
            hbox.addWidget(slider)

            self.labels.append(label)
            self.sliders.append(slider)
            layout.addLayout(hbox)

        train_btn = QPushButton("Start Training")
        train_btn.clicked.connect(self.startTrainer)
        layout.addWidget(train_btn)

        stop_btn = QPushButton("Stop Training")
        stop_btn.clicked.connect(self.stopTrainer)
        layout.addWidget(stop_btn)

        self.setLayout(layout)

    def startTrainer(self):
        try:
            values = [
                self.sliders[0].value(),
                self.sliders[1].value(),
                self.sliders[2].value(),
                self.sliders[3].value() / 10.0,
                self.sliders[4].value() / 10.0,
                self.sliders[5].value(),
            ]

            param_dict = {
                name: val / 10.0 if "x10" in name else val
                for (name, *_), val in zip(self.param_defs, values)
            }

            with open("params.json", "w") as f:
                json.dump(param_dict, f, indent=2)

            isaac_root = "/home/dafodilrat/Documents/bu/RASTIC/isaac-sim-standalone@4.5.0-rc.36+release.19112.f59b3005.gl.linux-x86_64.release"
            setup_script = f"{isaac_root}/python.sh"
            python_bin = f"{isaac_root}/kit/python/bin/python3"
            train_script = f"{isaac_root}/alpha/exts/customView/customView/test.py"

            self.proc = subprocess.Popen(
                [setup_script, train_script],
                preexec_fn=os.setsid  # Launches in a new session
            )

        except Exception as e:
            QMessageBox.critical(self, "Execution Error", f"Unexpected error: {e}")

    def stopTrainer(self):
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                self.proc.wait(timeout=5)
                QMessageBox.information(self, "Training Stopped", "Isaac Sim was terminated.")
            except Exception as e:
                QMessageBox.critical(self, "Stop Error", f"Failed to terminate Isaac Sim: {e}")
        else:
            QMessageBox.information(self, "No Active Process", "There is no running Isaac Sim process.")



if __name__ == "__main__":
    
    app = QApplication(sys.argv)
    gui = RLParamInputGUI()
    gui.show()
    sys.exit(app.exec_())
