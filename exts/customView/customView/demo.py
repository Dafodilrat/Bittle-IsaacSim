# demo.py

from omni.isaac.kit import SimulationApp
import os
import json
import time
import traceback
import glob
import omni.kit.app
import torch as th

class MultiAgentDemo:
    """
    Loads and executes trained policies for multiple Bittles in Isaac Sim.
    No training is done — agents run in inference/demo mode only.
    """
    def __init__(self, config_file="params.json"):
        self.config_file = config_file
        self.agents = []
        self.sim_env = None

        # === Load Configuration ===
        self.load_config()

        # === Launch SimulationApp ===
        renderer_mode = "None" if self.headless else "Hybrid"
        SimulationApp({
            "headless": self.headless,
            "renderer": renderer_mode,
            "hide_ui": True,
            "window_width": 1280,
            "window_height": 720,
        })

        # === Safe to import tools post-launch ===
        from tools import get_free_gpu, ensure_dir_exists, log, wait_for_stage_ready
        self.get_free_gpu = get_free_gpu
        self.ensure_dir_exists = ensure_dir_exists
        self.log = log
        self.wait_for_stage_ready = wait_for_stage_ready

        from PPO import PPOAgent
        from Dp3d import DDPGAgent
        from Td3 import TD3Agent
        from A2C import A2CAgent

        self.agent_classes = {
            "ppo": PPOAgent,
            "dp3d": DDPGAgent,
            "td3": TD3Agent,
            "a2c": A2CAgent,
        }

        self.log("[DEMO] MultiAgentDemo initialized", True)

    def load_config(self):
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Configuration file '{self.config_file}' not found.")

        with open(self.config_file, "r") as f:
            config = json.load(f)

        self.all_weights = config["params"]
        self.all_joint_states = config["joint_states"]
        self.agent_algorithms = config["algorithms"]
        self.num_agents = config["num_agents"]
        self.headless = config.get("headless", False)

        print("[DEMO] Loaded configuration:")
        print(json.dumps(config, indent=2), flush=True)

    def setup_environment_and_agents(self):
        from environment import Environment
        from TrainingGround import TrainingGround

        self.log("[DEMO] Setting up environment...", True)

        # Enable consistent relative spawn pattern
        TrainingGround.set_sync(True, seed=42)

        self.sim_env = Environment()
        self.sim_env.add_training_grounds(n=self.num_agents, size=12.0)
        self.sim_env.add_bittles(n=self.num_agents, flush=True)

        self.agents.clear()

        for i, bittle in enumerate(self.sim_env.bittles):
            algo = self.agent_algorithms[i].lower()
            weights = self.all_weights[i]
            joint_states = self.all_joint_states[i]
            ground = self.sim_env.training_grounds[i]

            agent_class = self.agent_classes.get(algo)
            if not agent_class:
                raise ValueError(f"[DEMO] Unsupported algorithm: {algo}")

            self.log(f"[DEMO] Loading agent {i} ({algo})...", True)

            agent = agent_class(
                bittle=bittle,
                weights=weights,
                sim_env=self.sim_env,
                joint_states=joint_states,
                grnd=ground,
                device=self.get_free_gpu(),
                log=False
            )

            # Load checkpoint
            ckpt = agent._load_latest_checkpoint(prefix=algo)
            if ckpt:
                agent.model.set_parameters(ckpt["path"])
                self.log(f"[DEMO] Loaded checkpoint: {ckpt['path']} (step {ckpt['step']})", True)
            else:
                self.log(f"[DEMO] No checkpoint found for agent {i} ({algo})", True)

            agent.reset()
            self.agents.append(agent)

        self.log(f"[DEMO] {len(self.agents)} agents ready for inference", True)

    def run(self):
        self.wait_for_stage_ready()

        timeline = omni.timeline.get_timeline_interface()
        if not timeline.is_playing():
            timeline.play()

        app = omni.kit.app.get_app()

        try:
            self.log("[DEMO] Isaac Sim running in inference-only mode. Press Ctrl+C to stop.", True)

            while True:
                app.update()
                for i, agent in enumerate(self.agents):

                    if agent.gym_env.is_terminated():
                        self.log(f"[DEMO] Agent {i} terminated. Respawning...", True)
                        agent.reset()
                        continue

                    action = agent.predict_action(agent.obs)
                    agent.step(action)

                self.sim_env.get_world().step(render=True)

        except KeyboardInterrupt:
            self.log("[DEMO] Exiting on keyboard interrupt.", True)

        except Exception as e:
            self.log(f"[DEMO] Exception in simulation loop: {e}", True)
            traceback.print_exc()


if __name__ == "__main__":
    try:
        demo = MultiAgentDemo()
        demo.setup_environment_and_agents()
        demo.run()
    except Exception as e:
        print("[FATAL] Failed to launch demo:", flush=True)
        traceback.print_exc()
