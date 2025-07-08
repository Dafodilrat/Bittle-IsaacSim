from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QMessageBox, QSlider, QApplication, QFrame, QCheckBox
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
        self.setWindowTitle("RL Parameter Input + 3D STL Viewer")
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
        self.proc = None
        self.isaac_root = "/home/dafodilrat/Documents/bu/RASTIC/isaac-sim-standalone@4.5.0-rc.36+release.19112.f59b3005.gl.linux-x86_64.release"
        
        self.joint_labels = {
            "Left back shoulder":   (20, 50, 270),
            "Left back knee":   (13, 20, 235),
            "Left front shoulder":   (-1, 55, 165),
            "Left front knee":   (-5, 20, 130),
            "Right back shoulder":   (130, 50, 250),
            "Right back knee":   (130, 10, 235),
            "Right front shoulder":   (110, 50, 150),
            "Right front knee":   (110, 15, 115),
        }

        self.joint_checkboxes = []
        self.initUI()

    def initUI(self):
        main_layout = QHBoxLayout()

        # === Left Panel: Parameter Sliders, Buttons, Checkboxes ===
        self.control_layout = QVBoxLayout()

        self.initSliders()   
        self.initJointLocks()
        self.initButtons()  

        main_layout.addLayout(self.control_layout)  # <--- Move this here

        # === Right Panel: VTK STL Viewer ===
        vtk_frame = QFrame()
        vtk_layout = QVBoxLayout()
        self.vtk_widget = QVTKRenderWindowInteractor(vtk_frame)
        vtk_layout.addWidget(self.vtk_widget)
        vtk_frame.setLayout(vtk_layout)
        main_layout.addWidget(vtk_frame)

        self.setLayout(main_layout)
        self.init_vtk()
    

    def initButtons(self):

        train_btn = QPushButton("Start Training")
        train_btn.clicked.connect(self.startTrainer)
        self.control_layout.addWidget(train_btn)

        stop_btn = QPushButton("Stop Training")
        stop_btn.clicked.connect(self.stopTrainer)
        self.control_layout.addWidget(stop_btn)

    def initSliders(self):
        
        header = QLabel("Training Parameters")          
        header.setStyleSheet("font-weight: bold; font-size: 14pt;")
        header.setContentsMargins(0, 3, 0, 3)  # Left, Top, Right, Bottom

        self.control_layout.addWidget(header)

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
            self.control_layout.addLayout(hbox)


    def initJointLocks(self):
        
        header = QLabel("Select Joints To Lock")          
        header.setStyleSheet("font-weight: bold; font-size: 14pt;")
        header.setContentsMargins(0, 3, 0, 3)  # Left, Top, Right, Bottom

        self.control_layout.addWidget(header)
        
        for joint in self.joint_labels.keys():
            checkbox = QCheckBox(joint)
            checkbox.setChecked(False)
            self.joint_checkboxes.append(checkbox)
            self.control_layout.addWidget(checkbox)

    def init_vtk(self):
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

        # Joint label coordinates (in STL coordinate frame)
        # Joint label coordinates (in STL coordinate frame)
        # (-0.445, -0.519, -0.021),

        for name, pos in self.joint_labels.items():
            # --- Create the text ---
            text_src = vtk.vtkVectorText()
            text_src.SetText(name.replace("_", " "))

            text_mapper = vtk.vtkPolyDataMapper()
            text_mapper.SetInputConnection(text_src.GetOutputPort())

            text_actor = vtk.vtkFollower()
            text_actor.SetMapper(text_mapper)
            text_actor.SetScale(10,10,10)
            text_actor.SetPosition(*pos)

            # --- Assign a random distinct color ---
            color = [random.uniform(0.3, 1.0) for _ in range(3)]
            text_actor.GetProperty().SetColor(*color)
            text_actor.GetProperty().SetOpacity(10)
            text_actor.SetCamera(self.renderer.GetActiveCamera())
            self.renderer.AddActor(text_actor)

            # --- Add a small point marker at the joint ---
            sphere_src = vtk.vtkSphereSource()
            sphere_src.SetRadius(6)  # Adjust size
            sphere_src.SetThetaResolution(12)
            sphere_src.SetPhiResolution(12)

            sphere_mapper = vtk.vtkPolyDataMapper()
            sphere_mapper.SetInputConnection(sphere_src.GetOutputPort())

            sphere_actor = vtk.vtkActor()
            sphere_actor.SetMapper(sphere_mapper)
            sphere_actor.SetPosition(pos[0],pos[1]-40,pos[2])
            sphere_actor.GetProperty().SetColor(*color)  # Match label color

            self.renderer.AddActor(sphere_actor)

            style = DragRotateInteractorStyle()
            style.SetDefaultRenderer(self.renderer)
            self.interactor.SetInteractorStyle(style)

            self.interactor.Initialize()
            # self.interactor.Start()

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

            joint_states = {
                cb.text(): cb.isChecked() for cb in self.joint_checkboxes
            }

            with open("params.json", "w") as f:
                json.dump({
                    "params": param_dict,
                    "joint_states": joint_states
                }, f, indent=2)

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
