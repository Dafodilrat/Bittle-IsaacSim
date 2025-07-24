from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
        QMessageBox, QSlider, QApplication, QFrame, QCheckBox, QTabWidget, QSpinBox
)
from PyQt5.QtOpenGL import QGLWidget
from PyQt5.QtCore import Qt
from OpenGL.GL import glGetString, GL_RENDERER, GL_VENDOR
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

        self.isaac_root = os.environ.get("ISAACSIM_PATH")
        self.default_weights = [5, 4, 2, 1, 0.5, 3, 8, 2]
        self.param_defs = [
            ("Correct Posture Bonus", 0, 10, self.default_weights[0]),
            ("Smooth Bonus Weight", 0, 10, self.default_weights[1]),
            ("Incorrect Posture Penalty", 0, 10, self.default_weights[2]),
            ("Jerking Movement Penalty (x10)", 0, 50, int(self.default_weights[3] * 10)),  # scaled
            ("High Joint Velocity Penalty (x10)", 0, 50, int(self.default_weights[4] * 10)),  # scaled
            ("Z Height Penalty", 0, 10, self.default_weights[5]),
            ("Distance to Goal Penalty", 0, 20, self.default_weights[6]),
            ("Goal Alignment Bonus", 0, 10, self.default_weights[7]),
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
        self.train_btn = None
        self.stop_btn = None
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

        try:
            self.renderer_info = self.detect_renderer()
        except Exception as e:
            self.renderer_info = f"Renderer: [Unavailable]"

        # Add renderer label (green text)
        self.renderer_label = QLabel(self.renderer_info)
        self.renderer_label.setStyleSheet("color: green; font-size: 10pt;")
        self.control_layout.addWidget(self.renderer_label)

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

    def detect_renderer(self):

        class DummyGL(QGLWidget):
            def initializeGL(self_):
                self_.renderer = glGetString(GL_RENDERER).decode()
                self_.vendor = glGetString(GL_VENDOR).decode()
                self_.close()

        dummy = DummyGL()
        dummy.show()  # required to trigger initializeGL
        dummy.raise_()
        dummy.activateWindow()
        QApplication.processEvents()
        return f"Renderer: {dummy.vendor} - {dummy.renderer}"


    def generateTabs(self):
        self.tabs.clear()
        self.bittle_tabs = []

        for i in range(self.agent_spinner.value()):
            joint_checkboxes = []
            sliders = []

            from PyQt5.QtWidgets import QComboBox
            algo_combo = QComboBox()
            algo_combo.addItems(["ppo", "dp3d", "td3", "a2c"])

            tab = QWidget()
            vbox = QVBoxLayout()

            header = QLabel("Algorithm Selection")
            header.setStyleSheet("font-weight: bold; font-size: 12pt;")
            vbox.addWidget(header)
            vbox.addWidget(algo_combo)

            header = QLabel("Training Parameters")
            header.setStyleSheet("font-weight: bold; font-size: 12pt;")
            vbox.addWidget(header)

            for label_text, min_val, max_val, default in self.param_defs:
                hbox = QHBoxLayout()
                slider = QSlider(Qt.Horizontal)
                scaled = "x10" in label_text
                label = QLabel(f"{label_text}: {default / 10.0:.1f}" if scaled else f"{label_text}: {default}")
                slider.setMinimum(min_val)
                slider.setMaximum(max_val)
                slider.setValue(default * 10 if scaled else default)
                slider.setTickInterval(1)
                slider.setSingleStep(1)
                slider.valueChanged.connect(
                    lambda val, l=label, t=label_text: l.setText(
                        f"{t}: {val / 10.0:.1f}" if "x10" in t else f"{t}: {val}"
                    )
                )
                hbox.addWidget(label)
                hbox.addWidget(slider)
                vbox.addLayout(hbox)
                sliders.append(slider)

            header = QLabel("Lock Joints")
            header.setStyleSheet("font-weight: bold; font-size: 12pt;")
            vbox.addWidget(header)
            
            for joint in self.joint_labels.keys():
                cb = QCheckBox(joint)
                joint_checkboxes.append(cb)
                vbox.addWidget(cb)

            tab.setLayout(vbox)
            self.tabs.addTab(tab, f"Bittle {i+1}")
            self.bittle_tabs.append((sliders, joint_checkboxes, algo_combo))


    def get_config(self):
        all_weights = []
        all_joint_states = []
        algorithms = []

        for sliders, joint_checkboxes, algo_combo in self.bittle_tabs:
            values = [s.value() for s in sliders]
            weights = [val / 10.0 if "x10" in label else val
                    for (label, *_), val in zip(self.param_defs, values)]
            joints = {cb.text(): cb.isChecked() for cb in joint_checkboxes}
            algo = algo_combo.currentText()

            all_weights.append(weights)
            all_joint_states.append(joints)
            algorithms.append(algo)

        return {
            "params": all_weights,
            "joint_states": all_joint_states,
            "algorithms": algorithms,
            "num_agents": self.agent_spinner.value(),
            "headless": self.headless_checkbox.isChecked(),
            "training_mode": self.training_mode_checkbox.isChecked()
        }


    def initButtons(self):

        self.training_mode_checkbox = QCheckBox("Training Mode (Separate Ground Planes)")
        self.training_mode_checkbox.setChecked(False)
        self.control_layout.addWidget(self.training_mode_checkbox)

        self.headless_checkbox = QCheckBox("Run in Headless Mode")
        self.headless_checkbox.setChecked(False)  # Default: unchecked
        self.control_layout.addWidget(self.headless_checkbox)

        self.train_btn = QPushButton("Start Training")
        self.train_btn.clicked.connect(self.startTrainer)
        self.control_layout.addWidget(self.train_btn)

        self.stop_btn = QPushButton("Stop Training")
        self.stop_btn.clicked.connect(self.stopTrainer)
        self.control_layout.addWidget(self.stop_btn)

    def startTrainer(self):
        try:
            config = self.get_config()
            with open("params.json", "w") as f:
                json.dump(config, f, indent=2)

            setup_script = f"{self.isaac_root}/python.sh"
            train_script = f"{self.isaac_root}/alpha/exts/customView/customView/trainer.py"

            self.proc = subprocess.Popen(
                [setup_script, train_script],
                preexec_fn=os.setsid
            )

            # self.proc = subprocess.Popen(
            #     f"{self.isaac_root}/isaac-sim.sh",
            #     preexec_fn=os.setsid
            # )

            self.train_btn.setEnabled(False)   # 🔒 Disable Start
            self.stop_btn.setEnabled(True)     # 🔓 Enable Stop

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

        self.train_btn.setEnabled(True)   # 🔓 Re-enable Start
        self.stop_btn.setEnabled(False)   # 🔒 Disable Stop


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = RLParamInputGUI()
    gui.show()
    sys.exit(app.exec_())
