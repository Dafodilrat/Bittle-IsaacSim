from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QMessageBox, QSlider, QApplication, QFrame, QCheckBox, QTabWidget, QSpinBox, QFileDialog
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
            camera.Azimuth(-delta_x * 0.3)
            camera.Elevation(-delta_y * 0.3)
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
            ("Jerking Movement Penalty (x10)", 0, 50, int(self.default_weights[3] * 10)),
            ("High Joint Velocity Penalty (x10)", 0, 50, int(self.default_weights[4] * 10)),
            ("Z Height Penalty", 0, 10, self.default_weights[5]),
            ("Distance to Goal Penalty", 0, 20, self.default_weights[6]),
            ("Goal Alignment Bonus", 0, 10, self.default_weights[7]),
        ]

        self.joint_labels = {
            "left_back_shoulder_joint": (20, 50, 270),
            "left_back_knee_joint": (13, 20, 235),
            "left_front_shoulder_joint": (-1, 55, 165),
            "left_front_knee_joint": (-5, 20, 130),
            "right_back_shoulder_joint": (130, 50, 250),
            "right_back_knee_joint": (130, 10, 235),
            "right_front_shoulder_joint": (110, 50, 150),
            "right_front_knee_joint": (110, 15, 115),
        }

        self.bittle_tabs = []
        self.train_btn = None
        self.stop_btn = None
        self.tab_memory = {}  # Memory for preserving tab state
        self.demo_ckpt_slider = None
        self.demo_ckpt_label = None

        self.initUI()

    def init_load_save(self):

        self.load_btn = QPushButton("Load Config")
        self.load_btn.clicked.connect(self.load_config_file)

        self.save_btn = QPushButton("Save Config")
        self.save_btn.clicked.connect(self.save_config_file)

        # Add them somewhere in your layout
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.load_btn)
        btn_layout.addWidget(self.save_btn)

        return btn_layout

    def apply_config_to_ui(self, cfg):
        """
        Apply loaded config dictionary to the UI.
        Matches exactly what get_config() outputs.
        """
        try:
            # Set number of agents
            if "num_agents" in cfg:
                self.agent_spinner.setValue(cfg["num_agents"])
                self.generateTabs()  # rebuild tabs so we can populate them

            # Restore parameters, joint states, and algorithms
            for i, (sliders, checkboxes, algo_combo) in enumerate(self.bittle_tabs):
                if "params" in cfg and i < len(cfg["params"]):
                    for s, val in zip(sliders, cfg["params"][i]):
                        # Scale back up if this param was divided by 10 in get_config()
                        idx = sliders.index(s)
                        label_text = self.param_defs[idx][0]
                        if "x10" in label_text:
                            s.setValue(int(val * 10))
                        else:
                            s.setValue(int(val))

                if "joint_states" in cfg and i < len(cfg["joint_states"]):
                    joint_state = cfg["joint_states"][i]
                    for cb in checkboxes:
                        if cb.text() in joint_state:
                            cb.setChecked(bool(joint_state[cb.text()]))

                if "algorithms" in cfg and i < len(cfg["algorithms"]):
                    algo_combo.setCurrentText(cfg["algorithms"][i])

            # Restore headless mode
            if "headless" in cfg:
                self.headless_checkbox.setChecked(bool(cfg["headless"]))

            # Restore training mode
            if "training_mode" in cfg:
                self.training_mode_checkbox.setChecked(bool(cfg["training_mode"]))

            # Restore checkpoint slider
            if "demo_ckpt_step" in cfg and not self.load_latest_checkbox.isChecked():
                if not self.demo_ckpt_slider:
                    self.toggle_demo_slider()
                if self.demo_ckpt_slider:
                    self.demo_ckpt_slider.setValue(int(cfg["demo_ckpt_step"]))

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply config: {e}")


    def load_config_file(self):
        """Load a JSON config and update the UI."""
        fname, _ = QFileDialog.getOpenFileName(self, "Load Config", "", "JSON Files (*.json)")
        if not fname:
            return
        try:
            with open(fname, "r") as f:
                cfg = json.load(f)
            # Apply cfg to your UI
            self.apply_config_to_ui(cfg)
            QMessageBox.information(self, "Config Loaded", f"Loaded configuration from {fname}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load config: {e}")

    def save_config_file(self):
        """Save the current UI config to a chosen JSON file."""
        fname, _ = QFileDialog.getSaveFileName(self, "Save Config", "", "JSON Files (*.json)")
        if not fname:
            return
        cfg = self.get_config()  # already returns a dict of params
        try:
            with open(fname, "w") as f:
                json.dump(cfg, f, indent=2)
            QMessageBox.information(self, "Config Saved", f"Saved configuration to {fname}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save config: {e}")

    def initUI(self):

        btn_layout = self.init_load_save()

        main_layout = QHBoxLayout()
        self.control_layout = QVBoxLayout()

        self.agent_spinner = QSpinBox()
        self.agent_spinner.setMinimum(1)
        self.agent_spinner.setMaximum(40)
        self.agent_spinner.setValue(2)
        self.agent_spinner.valueChanged.connect(self.generateTabs)
        self.control_layout.addLayout(btn_layout)
        self.control_layout.addWidget(QLabel("Number of Bittles"))
        self.control_layout.addWidget(self.agent_spinner)

        self.tabs = QTabWidget()
        self.control_layout.addWidget(self.tabs)

        self.init_sliders()
        self.initButtons()

        self.training_mode_checkbox.stateChanged.connect(self.toggle_demo_slider)

        main_layout.addLayout(self.control_layout)

        self.init_vtk(main_layout)
        self.setLayout(main_layout)
        self.generateTabs()

        try:
            self.renderer_info = self.detect_renderer()
        except Exception:
            self.renderer_info = f"Renderer: [Unavailable]"
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
    
    def toggle_demo_slider(self):
        is_latest = self.load_latest_checkbox.isChecked()

        if is_latest:
            if self.demo_ckpt_slider:
                self.control_layout.removeWidget(self.demo_ckpt_slider)
                self.demo_ckpt_slider.setParent(None)
                self.demo_ckpt_slider.deleteLater()
                self.demo_ckpt_slider = None

            if self.demo_ckpt_label:
                self.control_layout.removeWidget(self.demo_ckpt_label)
                self.demo_ckpt_label.setParent(None)
                self.demo_ckpt_label.deleteLater()
                self.demo_ckpt_label = None
        else:
            if not self.demo_ckpt_slider:
                self.demo_ckpt_label = QLabel("Checkpoint Step: 0")
                self.demo_ckpt_slider = QSlider(Qt.Horizontal)
                self.demo_ckpt_slider.setMinimum(0)
                self.demo_ckpt_slider.setMaximum(10000)
                self.demo_ckpt_slider.setTickInterval(1000)
                self.demo_ckpt_slider.setSingleStep(1000)
                self.demo_ckpt_slider.setPageStep(1000)
                self.demo_ckpt_slider.setValue(0)
                self.demo_ckpt_slider.setTickPosition(QSlider.TicksBelow)
                self.demo_ckpt_slider.valueChanged.connect(self.snap_demo_ckpt)

                self.control_layout.addWidget(self.demo_ckpt_label)
                self.control_layout.addWidget(self.demo_ckpt_slider)



    def init_sliders(self):
        # === Add Training Mode Checkbox ===
        self.training_mode_checkbox = QCheckBox("Training Mode (Separate Ground Planes)")
        self.training_mode_checkbox.setChecked(False)
        self.control_layout.addWidget(self.training_mode_checkbox)

        # === Add Load Latest Checkbox ===
        self.load_latest_checkbox = QCheckBox("Load Latest Checkpoint")
        self.load_latest_checkbox.setChecked(True)
        self.load_latest_checkbox.stateChanged.connect(self.toggle_demo_slider)
        self.control_layout.addWidget(self.load_latest_checkbox)

        # === Defer creation of slider and label to toggle_demo_slider
        self.demo_ckpt_label = None
        self.demo_ckpt_slider = None

        # === Initialize UI state
        self.toggle_demo_slider()

    def snap_demo_ckpt(self, val):
        snapped_val = round(val / 1000) * 1000
        if snapped_val != val:
            self.demo_ckpt_slider.blockSignals(True)
            self.demo_ckpt_slider.setValue(snapped_val)
            self.demo_ckpt_slider.blockSignals(False)
        self.demo_ckpt_label.setText(f"Checkpoint Step: {snapped_val}")

    def generateTabs(self):
        # Save current tab states to memory
        for idx, (sliders, checkboxes, algo_combo) in enumerate(self.bittle_tabs):
            self.tab_memory[idx] = {
                "slider_vals": [s.value() for s in sliders],
                "checkbox_vals": [cb.isChecked() for cb in checkboxes],
                "algo": algo_combo.currentText()
            }

        self.tabs.clear()
        self.bittle_tabs = []

        for i in range(self.agent_spinner.value()):
            from PyQt5.QtWidgets import QComboBox
            sliders, checkboxes = [], []
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
                slider.setValue(default)
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
                checkboxes.append(cb)
                vbox.addWidget(cb)

            tab.setLayout(vbox)
            self.tabs.addTab(tab, f"Bittle {i+1}")
            self.bittle_tabs.append((sliders, checkboxes, algo_combo))

            # Restore from memory if available
            if i in self.tab_memory:
                state = self.tab_memory[i]
                for s, val in zip(sliders, state["slider_vals"]):
                    s.setValue(val)
                for cb, val in zip(checkboxes, state["checkbox_vals"]):
                    cb.setChecked(val)
                algo_combo.setCurrentText(state["algo"])


    def get_config(self):
        all_weights, all_joint_states, algorithms = [], [], []
        for sliders, joint_checkboxes, algo_combo in self.bittle_tabs:
            values = [s.value() for s in sliders]
            weights = [val / 10.0 if "x10" in label else val for (label, *_), val in zip(self.param_defs, values)]
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
            "training_mode": self.training_mode_checkbox.isChecked(),
            "demo_ckpt_step": self.demo_ckpt_slider.value() if self.demo_ckpt_slider else -1
        }

    def initButtons(self):
        # Remove this line:
        # self.training_mode_checkbox = QCheckBox("Training Mode (Separate Ground Planes)")

        self.headless_checkbox = QCheckBox("Run in Headless Mode")
        self.headless_checkbox.setChecked(False)
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

            # Handle bittle.config rollover
            base_name = "bittle"
            ext = ".config"
            current_file = base_name + ext

            if os.path.exists(current_file):
                # Find next available bittleN.config
                i = 2
                while os.path.exists(f"{base_name}{i}{ext}"):
                    i += 1
                os.rename(current_file, f"{base_name}{i}{ext}")

            # Save current config to bittle.config
            with open(current_file, "w") as f:
                json.dump(config, f, indent=2)

            setup_script = f"{self.isaac_root}/python.sh"
            script_path = (
                f"{self.isaac_root}/alpha/exts/customView/customView/trainer.py"
                if config.get("training_mode", False)
                else f"{self.isaac_root}/alpha/exts/customView/customView/demo.py"
            )
            self.proc = subprocess.Popen([setup_script, script_path], preexec_fn=os.setsid)
            self.train_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Execution Error", f"Unexpected error: {e}")

    def stopTrainer(self):
        if hasattr(self, "proc") and self.proc and self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                self.proc.wait(timeout=5)
                QMessageBox.information(self, "Training Stopped", "Isaac Sim was terminated.")
            except Exception as e:
                QMessageBox.critical(self, "Stop Error", f"Failed to terminate Isaac Sim: {e}")
        else:
            QMessageBox.information(self, "No Active Process", "There is no running Isaac Sim process.")
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def closeEvent(self, event):
        self.tab_memory.clear()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = RLParamInputGUI()
    gui.show()
    sys.exit(app.exec_())
