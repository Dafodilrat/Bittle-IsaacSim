from omni.isaac.kit import SimulationApp

import os
import json
import time
import traceback
import omni.kit.app
import pynvml
import torch as th

class SensorTest:

    def __init__(self):
        print("[DEBUG] MultiAgentTrainer.__init__ started", flush=True)

        self.agents = []
        self.sim_env = None
        self.headless = False  # Set to True for headless mode
        
        # Setup SimulationApp after config is loaded
        renderer_mode = "None" if self.headless else "Hybrid"
        
        SimulationApp({
            "headless": self.headless,
            "renderer": renderer_mode,
            "hide_ui":  False,  # Only hide UI if headless
            "window_width": 1280,
            "window_height": 720,
        })
        print(f"[DEBUG] SimulationApp initialized (headless={self.headless}, renderer={renderer_mode})", flush=True)
        
        from tools import log 
        self.log = log

    def wait_for_stage_ready(self, timeout=10.0):

        print("[DEBUG] Waiting for stage to be ready...", flush=True)
        app = omni.kit.app.get_app()
        timeline = omni.timeline.get_timeline_interface()
        from isaacsim.core.utils.stage import is_stage_loading 

        t0 = time.time()
        while is_stage_loading() or not timeline:
            if time.time() - t0 > timeout:
                raise RuntimeError("Timeout waiting for stage to be ready")
            print("[ENV] Waiting for stage...", flush=True)
            app.update()
            time.sleep(0.1)
        print("[DEBUG] Stage is ready.", flush=True)


    def setup_environment_and_agents(self):
        from environment import Environment

        self.log("[DEBUG] Setting up environment and agents...", True)

        try:
            self.sim_env = Environment()
            self.log("[DEBUG] Environment object created", True)

            self.sim_env.add_training_grounds(n=1, size=20.0)
            self.log(f"[DEBUG] {len(self.sim_env.training_grounds)} training grounds added", True)

            self.sim_env.add_bittles(n=1,flush=True)
            self.log(f"[DEBUG] {1} Bittles added", True)

        except Exception as e:
            self.log("[ERROR] Error setting up environment or spawning agents", True)
            traceback.print_exc()

        self.agents.clear()

        self.log("[DEBUG] All agents set up", True)



if __name__ == "__main__":
    print("[DEBUG] Starting training script", flush=True)

    try:
        trainer = SensorTest()
        trainer.setup_environment_and_agents()

        # Start and maintain simulation
        timeline = omni.timeline.get_timeline_interface()
        if not timeline.is_playing():
            timeline.play()
            print("[DEBUG] Timeline started.", flush=True)

        print("[DEBUG] Entering infinite simulation loop...", flush=True)
        app = omni.kit.app.get_app()
        while True:
            app.update()
            trainer.sim_env.bittles[0].print_info()
            time.sleep(0.01)

    except Exception as e:
        print("[FATAL] Unhandled exception during training:", flush=True)
        traceback.print_exc()


