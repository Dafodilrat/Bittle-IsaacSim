import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit,
    QPushButton, QVBoxLayout, QHBoxLayout, QMessageBox
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer

from PIL import Image
import numpy as np
import json
import subprocess
import os

class RLParamInputGUI(QWidget):
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RL Parameter Input")
        self.params = {}

        # Define your custom reward function parameters and defaults
        self.param_defs = {
            "correct posture bonus": "100",
            "smooth_bonus_weight": "10",
            "incorrect posture penalty": "10",
            "jerking movement penalty": "0.5",
            "high joint velocity penalty": "0.2",
            "distance to goal penalty": "10"
        }

        self.input_boxes = {}
        self.initUI()

    def receive_frame(self):
        try:
            raw_len = self.sock.recv(4)
            if not raw_len:
                return
            img_len = int.from_bytes(raw_len, 'big')
            img_data = b''
            while len(img_data) < img_len:
                img_data += self.sock.recv(img_len - len(img_data))

            img = Image.open(io.BytesIO(img_data)).convert("RGB")
            arr = np.array(img)
            h, w, ch = arr.shape
            qt_img = QImage(arr.data, w, h, ch * w, QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(qt_img))
        except Exception as e:
            print("Error receiving frame:", e)

    def closeEvent(self, event):
        self.sock.close()
        event.accept()
    
    
    def initUI(self):
        layout = QVBoxLayout()

        for name, default in self.param_defs.items():
            hbox = QHBoxLayout()
            label = QLabel(name)
            input_box = QLineEdit()
            input_box.setText(default)
            hbox.addWidget(label)
            hbox.addWidget(input_box)
            layout.addLayout(hbox)
            self.input_boxes[name] = input_box

        train_btn = QPushButton("Start Training")
        train_btn.clicked.connect(self.collect_params)
        layout.addWidget(train_btn)

        self.setLayout(layout)

    def collect_params(self):
        try:
            self.params = {
                name: float(box.text())
                for name, box in self.input_boxes.items()
            }

            with open("params.json", "w") as f:
                json.dump(self.params, f)

            isaac_python_sh = "/home/dafodilrat/Documents/bu/RASTIC/isaac-sim-standalone@4.5.0-rc.36+release.19112.f59b3005.gl.linux-x86_64.release/python.sh"
            launch_script = "/home/dafodilrat/Documents/bu/RASTIC/alpha/launch.py"

            subprocess.Popen([
                isaac_python_sh,
                launch_script,
                "--/renderer/activeGpu=1",
                "--/app/window/enableMenuBar=False",
                "--/app/window/enableBrowser=False ",
                "--/app/window/enableLayout=False ",
                "--/app/window/enableExtensions=False",
                "--/app/window/showStatusBar=False",
                "--no-window=False"
            ])

        except ValueError:
            QMessageBox.critical(self, "Input Error", "Please enter valid numeric values for all parameters.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = RLParamInputGUI()
    gui.show()
    sys.exit(app.exec_())