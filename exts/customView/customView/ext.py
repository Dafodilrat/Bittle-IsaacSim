import omni.ext
import omni.ui as ui
import omni.kit.app
import omni.timeline

from .Bittle import Bittle
from .PPO import train
from .world import Environment

class MinimalViewportExtension(omni.ext.IExt):

    def __init__(self):
        super().__init__()
        self._training_subscription = None
        self._window = None
        self.default_weights = [100, 10, 10, 0.5, 0.2, 10]
        self.trainer = None
        self.world = None

    def on_startup(self, ext_id):

        stage_path = "/home/dafodilrat/Documents/bu/RASTIC/rl_world.usd"
        omni.usd.get_context().open_stage(stage_path)
        # print("Stage path:", omni.usd.get_context().get_stage_path())


        print("[MinimalViewportExtension] Starting up")

        self._window = ui.Window("RL Training Panel", width=500, height=900)

        with self._window.frame:
            with ui.VStack(height=ui.Fraction(1.0), width=ui.Fraction(0.25), spacing=10):
                ui.Label("RL Parameters", height=20)

                self.correct_posture_slider = self.make_slider("Correct Posture Bonus", 0, 200, 1, 100)
                self.smooth_bonus_slider = self.make_slider("Smooth Bonus Weight", 0, 100, 1, 10)
                self.incorrect_posture_slider = self.make_slider("Incorrect Posture Penalty", 0, 100, 1, 10)
                self.jerking_penalty_slider = self.make_slider("Jerking Movement Penalty", 0, 5, 0.1, 0.5)
                self.joint_velocity_slider = self.make_slider("High Joint Velocity Penalty", 0, 5, 0.1, 0.2)
                self.distance_to_goal_slider = self.make_slider("Distance to Goal Penalty", 0, 50, 1, 10)

                ui.Button("Start Training", clicked_fn=self.start_training)
                ui.Button("Stop Training", clicked_fn=self.stop_training)

    def make_slider(self, label_text, min_val, max_val, step, default):
        ui.Label(label_text)
        slider = ui.FloatSlider(min=min_val, max=max_val, step=step)
        slider.model.set_value(default)
        return slider

    def stop_training(self):
        print("[MinimalViewportExtension] Training stopped (placeholder)")
        return

    def start_training(self):

        self.world = Environment()

        print("[UI] Scheduling training start on next frame...")

        omni.timeline.get_timeline_interface().play()

        self._training_subscription = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
            self._delayed_start_once
        )

    def _delayed_start_once(self, event):
        
        print("[DELAYED] Creating Bittle and starting training.")

        if self._training_subscription:
            self._training_subscription.unsubscribe()
            self._training_subscription = None

        bittle = Bittle()

        try:
            params = [
                self.correct_posture_slider.model.get_value_as_float(),
                self.smooth_bonus_slider.model.get_value_as_float(),
                self.incorrect_posture_slider.model.get_value_as_float(),
                self.jerking_penalty_slider.model.get_value_as_float(),
                self.joint_velocity_slider.model.get_value_as_float(),
                self.distance_to_goal_slider.model.get_value_as_float(),
            ]

            self.trainer = train(params, bittle, self.world)
            print("[MinimalViewportExtension] Launching training with:", params)
            self.trainer.start()

        except Exception as e:
            import traceback
            print("Error launching training:", e)
            traceback.print_exc()

    def on_shutdown(self):
        print("[MinimalViewportExtension] Shutting down...")
        if self._window:
            self._window.destroy()
            self._window = None
        self.bittle = None
        self._training_subscription = None