from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
        QMessageBox, QSlider, QApplication, QFrame, QCheckBox, QTabWidget, QSpinBox
)

from PyQt5.QtCore import Qt
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtk
import subprocess
import json
import sys
import signal
import os
import random


# Define custom interactor style to allow only yaw rotation (side-to-side)
class DragRotateInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self):
        super().__init__()
        self.last_pos = None
        self.left_button_down = False

        self.AddObserver("LeftButtonPressEvent", self.on_left_button_down)
        self.AddObserver("LeftButtonReleaseEvent", self.on_left_button_up)
        self.AddObserver("MouseMoveEvent", self.on_mouse_move)

    def on_left_button_down(self, obj, event):
        self.left_button_down = True
        self.last_pos = self.GetInteractor().GetEventPosition()

    def on_left_button_up(self, obj, event):
        self.left_button_down = False
        self.last_pos = None

    def on_mouse_move(self, obj, event):
        if not self.left_button_down:
            return

        interactor = self.GetInteractor()
        x, y = interactor.GetEventPosition()
        if self.last_pos is None:
            self.last_pos = (x, y)
            return

        last_x, last_y = self.last_pos
        delta_x = x - last_x
        delta_y = y - last_y
        self.last_pos = (x, y)

        renderer = self.GetDefaultRenderer()
        if renderer:
            camera = renderer.GetActiveCamera()
            camera.Azimuth(-delta_x * 0.3)       # ← yaw (unchanged)
            camera.Elevation(-delta_y * 0.3)     # ← pitch (inverted)
            renderer.ResetCameraClippingRange()
            interactor.Render()


class RLParamInputGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RL Multi-Agent Parameter GUI")

        self.isaac_root = "/home/dafodilrat/Documents/bu/RASTIC/isaac-sim-standalone@4.5.0-rc.36+release.19112.f59b3005.gl.linux-x86_64.release"
        self.default_weights = [100, 10, 10, 5, 2, 10]
        self.param_defs = [
            ("Correct Posture Bonus", 0, 100, self.default_weights[0]),
            ("Smooth Bonus Weight", 0, 100, self.default_weights[1]),
            ("Incorrect Posture Penalty", 0, 100, self.default_weights[2]),
            ("Jerking Movement Penalty (x10)", 0, 50, self.default_weights[3] * 10),
            ("High Joint Velocity Penalty (x10)", 0, 50, self.default_weights[4] * 10),
            ("Distance to Goal Penalty", 0, 100, self.default_weights[5]),
        ]
        self.joint_labels = {
            "left_back_shoulder_joint":   (20, 50, 270),
            "left_back_knee_joint":   (13, 20, 235),
            "left_front_shoulder_joint":   (-1, 55, 165),
            "left_front_knee_joint":   (-5, 20, 130),
            "right_back_shoulder_joint":   (130, 50, 250),
            "right_back_knee_joint":   (130, 10, 235),
            "right_front_shoulder_joint":   (110, 50, 150),
            "right_front_knee_joint":   (110, 15, 115),
        }

        self.bittle_tabs = []
        self.initUI()

    def initUI(self):
        main_layout = QHBoxLayout()
        self.control_layout = QVBoxLayout()

        self.agent_spinner = QSpinBox()
        self.agent_spinner.setMinimum(1)
        self.agent_spinner.setMaximum(10)
        self.agent_spinner.setValue(2)
        self.agent_spinner.valueChanged.connect(self.generateTabs)
        self.control_layout.addWidget(QLabel("Number of Bittles"))
        self.control_layout.addWidget(self.agent_spinner)

        self.tabs = QTabWidget()
        self.control_layout.addWidget(self.tabs)

        self.initButtons()
        main_layout.addLayout(self.control_layout)

        self.init_vtk(main_layout)  # ← Add shared 3D viewer

        self.setLayout(main_layout)
        self.generateTabs()

    def init_vtk(self, parent_layout):
        vtk_frame = QFrame()
        vtk_layout = QVBoxLayout()
        self.vtk_widget = QVTKRenderWindowInteractor(vtk_frame)
        vtk_layout.addWidget(self.vtk_widget)
        vtk_frame.setLayout(vtk_layout)
        parent_layout.addWidget(vtk_frame)

        self.renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()

        stl_path = f"{self.isaac_root}/alpha/Bittle_URDF/urdf/bittle.stl"
        reader = vtk.vtkSTLReader()
        reader.SetFileName(stl_path)
        reader.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(reader.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.7, 0.7, 0.9)
        self.renderer.AddActor(actor)

        self.renderer.SetBackground(0.1, 0.1, 0.1)
        self.renderer.ResetCamera()

        for name, pos in self.joint_labels.items():
            text_src = vtk.vtkVectorText()
            text_src.SetText(name.replace("_", " "))
            text_mapper = vtk.vtkPolyDataMapper()
            text_mapper.SetInputConnection(text_src.GetOutputPort())

            text_actor = vtk.vtkFollower()
            text_actor.SetMapper(text_mapper)
            text_actor.SetScale(10, 10, 10)
            text_actor.SetPosition(*pos)
            color = [random.uniform(0.3, 1.0) for _ in range(3)]
            text_actor.GetProperty().SetColor(*color)
            text_actor.GetProperty().SetOpacity(1.0)
            text_actor.SetCamera(self.renderer.GetActiveCamera())
            self.renderer.AddActor(text_actor)

            sphere_src = vtk.vtkSphereSource()
            sphere_src.SetRadius(6)
            sphere_src.SetThetaResolution(12)
            sphere_src.SetPhiResolution(12)

            sphere_mapper = vtk.vtkPolyDataMapper()
            sphere_mapper.SetInputConnection(sphere_src.GetOutputPort())

            sphere_actor = vtk.vtkActor()
            sphere_actor.SetMapper(sphere_mapper)
            sphere_actor.SetPosition(pos[0], pos[1] - 40, pos[2])
            sphere_actor.GetProperty().SetColor(*color)
            self.renderer.AddActor(sphere_actor)

        style = DragRotateInteractorStyle()
        style.SetDefaultRenderer(self.renderer)
        self.interactor.SetInteractorStyle(style)
        self.interactor.Initialize()

    def generateTabs(self):
        self.tabs.clear()
        self.bittle_tabs = []

        for i in range(self.agent_spinner.value()):
            joint_checkboxes = []
            sliders = []
            
            tab = QWidget()
            vbox = QVBoxLayout()

            header = QLabel("Training Parameters")          
            header.setStyleSheet("font-weight: bold; font-size: 14pt;")
            header.setContentsMargins(0, 3, 0, 3)  # Left, Top, Right, Bottom
            vbox.addWidget(header)

            for label_text, min_val, max_val, default in self.param_defs:
                hbox = QHBoxLayout()
                label = QLabel(f"{label_text}: {default / 10.0 if 'x10' in label_text else default}")
                slider = QSlider(Qt.Horizontal)
                slider.setMinimum(min_val)
                slider.setMaximum(max_val)
                slider.setValue(default)
                slider.setTickInterval(1)
                slider.setSingleStep(1)
                slider.valueChanged.connect(lambda val, l=label, t=label_text: l.setText(f"{t}: {val / 10.0 if 'x10' in t else val}"))
                hbox.addWidget(label)
                hbox.addWidget(slider)
                vbox.addLayout(hbox)
                sliders.append(slider)

            

            header = QLabel("Lock Joints")          
            header.setStyleSheet("font-weight: bold; font-size: 14pt;")
            header.setContentsMargins(0, 3, 0, 3)  # Left, Top, Right, Bottom

            for joint in self.joint_labels.keys():
                cb = QCheckBox(joint)
                joint_checkboxes.append(cb)
                vbox.addWidget(cb)

            tab.setLayout(vbox)
            self.tabs.addTab(tab, f"Bittle {i+1}")
            self.bittle_tabs.append((sliders, joint_checkboxes))

    def get_config(self):
        all_weights = []
        all_joint_states = []

        for sliders, joint_checkboxes in self.bittle_tabs:
            values = [s.value() for s in sliders]
            weights = [val / 10.0 if "x10" in label else val
                       for (label, *_), val in zip(self.param_defs, values)]
            joints = {cb.text(): cb.isChecked() for cb in joint_checkboxes}
            all_weights.append(weights)
            all_joint_states.append(joints)

        return {
            "params": all_weights,
            "joint_states": all_joint_states,
            "num_agents": self.agent_spinner.value()
        }

    def initButtons(self):
        train_btn = QPushButton("Start Training")
        train_btn.clicked.connect(self.startTrainer)
        self.control_layout.addWidget(train_btn)

        stop_btn = QPushButton("Stop Training")
        stop_btn.clicked.connect(self.stopTrainer)
        self.control_layout.addWidget(stop_btn)

    def startTrainer(self):
        try:
            config = self.get_config()
            with open("params.json", "w") as f:
                json.dump(config, f, indent=2)

            setup_script = f"{self.isaac_root}/python.sh"
            train_script = f"{self.isaac_root}/alpha/exts/customView/customView/test.py"

            self.proc = subprocess.Popen(
                [setup_script, train_script],
                preexec_fn=os.setsid
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
