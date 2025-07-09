import time
import omni.kit.app
from isaacsim.core.utils.stage import is_stage_loading

class SimulationCoordinator:
    def __init__(self, envs, world, timestep=1/60.0):
        """
        Args:
            envs: list of gym_env instances (each controlling one agent)
            world: the shared Isaac Sim world (from Environment singleton)
            timestep: simulation step interval (default 60Hz)
        """
        self.gym_envs = envs
        self.world = world
        self.timestep = timestep
        self.ready_to_step = False  # Delay stepping until first env.step() is triggered

    def run_forever(self):
        print("[Coordinator] Starting main loop", flush=True)
        try:
            while True:
                print("[Coordinator] About to check is_ready()", flush=True)
                if self.is_ready():
                    self.step_if_ready()
                else:
                    print("[Coordinator] Waiting for world to be ready...", flush=True)
                time.sleep(self.timestep)
        except Exception as e:
            import traceback
            print("[Coordinator] FATAL ERROR in run_forever:", e, flush=True)
            traceback.print_exc()

    def is_ready(self):
        print("in ready check", flush=True)
        try:
            if self.world is None:
                print("[ReadyCheck] world is None", flush=True)
                return False
            print("[ReadyCheck] world exists")

            if not hasattr(self.world, 'physics_sim_view'):
                print("[ReadyCheck] no physics_sim_view attr", flush=True)
                return False
            print("[ReadyCheck] has physics_sim_view attr")

            if self.world.physics_sim_view is None:
                print("[ReadyCheck] physics_sim_view is None", flush=True)
                return False
            print("[ReadyCheck] physics_sim_view OK")

            if not self.world.is_playing():
                print("[ReadyCheck] world is not playing", flush=True)
                return False
            print("[ReadyCheck] world is playing ✅", flush=True)

            return True
        except Exception as e:
            print("[Coordinator] Readiness check failed:", e, flush=True)
            return False

    def step_if_ready(self):
        print("[Coordinator] step_if_ready() entered", flush=True)

        if all(env._step_called for env in self.gym_envs):
            print("[Coordinator] All envs ready — applying actions and stepping", flush=True)
            self.ready_to_step = True

            for env in self.gym_envs:
                try:
                    env.apply_pending_action()
                except Exception as e:
                    print("[Coordinator] Error applying action:", e, flush=True)

            try:
                self.world.step(render=True)
            except Exception as e:
                print("[Coordinator] world.step() crashed:", e, flush=True)

            for env in self.gym_envs:
                try:
                    env.post_step()
                except Exception as e:
                    print("[Coordinator] post_step() failed:", e, flush=True)

        elif self.ready_to_step:
            print("[Coordinator] Waiting for next action(s)...", flush=True)
        else:
            print("[Coordinator] Not all envs called step() yet", flush=True)
