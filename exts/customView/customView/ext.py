import omni.ext
import omni.ui as ui
import omni.kit.app
import omni.timeline
import omni.usd
from isaacsim.core.utils.stage import get_current_stage
from omni.isaac.core.simulation_context import SimulationContext

import traceback
from .PPO import stb3_PPO
from .world import Environment
from time import time

class MinimalViewportExtension(omni.ext.IExt):

    def __init__(self):
        super().__init__()
        self._training_subscription = None
        self._window = None
        self.default_weights = [100, 10, 10, 0.5, 0.2, 10]
        self.trainer = None
        self.world = None
        self.env = None

    def on_startup(self, ext_id):
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

                self._start_button = ui.Button("Start Training", clicked_fn=self.start_training, enabled=True)
                ui.Button("Stop Training", clicked_fn=self.stop_training)

    def make_slider(self, label_text, min_val, max_val, step, default):
        ui.Label(label_text)
        slider = ui.FloatSlider(min=min_val, max=max_val, step=step)
        slider.model.set_value(default)
        return slider

    def stop_training(self):
        print("[MinimalViewportExtension] Training stopped.")
        if self.trainer is not None:
            self.trainer.stop_training()
            self.trainer = None
        if self.env is not None:
            print("[MinimalViewportExtension] Resetting world.")
            self.env.reset()

        self.on_startup("reinit")

    def start_training(self):
        print("[UI] Scheduling training start on next frame...")

        self._train_params = [
            self.correct_posture_slider.model.get_value_as_float(),
            self.smooth_bonus_slider.model.get_value_as_float(),
            self.incorrect_posture_slider.model.get_value_as_float(),
            self.jerking_penalty_slider.model.get_value_as_float(),
            self.joint_velocity_slider.model.get_value_as_float(),
            self.distance_to_goal_slider.model.get_value_as_float(),
        ]
        self._train_count = 1

        # Defer to next frame to ensure stage + physics is ready
        self._training_subscription = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
            self._delayed_start_once
        )

    def _delayed_start_once(self, _):

        if self._training_subscription:
            self._training_subscription.unsubscribe()
            self._training_subscription = None

        try:
            print("[Delayed Start] Initializing environment and launching training...")
            
            # Destroy previous instance
            Environment.destroy()
            
            # Make sure timeline is stopped before creating new environment
            timeline = omni.timeline.get_timeline_interface()
            if timeline.is_playing():
                timeline.stop()
                
            # Wait a moment for timeline to stop
            app = omni.kit.app.get_app()
            for _ in range(10):  # Wait up to 1 second
                app.update()
                if not timeline.is_playing():
                    break
                time.sleep(0.1)
            
            # Create environment (this will handle physics initialization)
            self.env = Environment()
            
            # Add Bittles
            self.env.add_bittles(n=self._train_count)
            
            # Start timeline/simulation AFTER everything is set up
            timeline.play()
            
            # Wait for simulation to actually start
            for _ in range(20):  # Wait up to 2 seconds
                app.update()
                if timeline.is_playing():
                    break
                time.sleep(0.1)
            
            # Now start training
            self.trainer = stb3_PPO(
                params=self._train_params,
                bittle=self.env.bittlles[0],
                env=self.env
            )

            print("[MinimalViewportExtension] Launching training with:", self._train_params)
            self.trainer.start_training()

        except Exception as e:
            print("[EXTENSION ERROR] Exception during training:", e)
            import traceback
            traceback.print_exc()
            # Stop timeline on error
            omni.timeline.get_timeline_interface().stop()
    def on_shutdown(self):
        print("[MinimalViewportExtension] Shutting down...")
        if self._window:
            self._window.destroy()
            self._window = None
        self._training_subscription = None
        if self.trainer is not None:
            self.trainer = None
        if self.env is not None:
            print("[MinimalViewportExtension] Resetting world on shutdown.")
            self.env.reset()
