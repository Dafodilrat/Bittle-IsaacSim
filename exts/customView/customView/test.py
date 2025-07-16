from omni.isaac.kit import SimulationApp
import os
import json
import time
import traceback

# Global SimulationApp (must be created in main thread)
simulation_app = None

class MultiAgentTrainer:
    def __init__(self, config_file="params.json", headless=False):
        print("[DEBUG] MultiAgentTrainer.__init__ started", flush=True)

        self.config_file = config_file
        self.agents = []
        self.sim_env = None
        self.steps_per_episode = 500
        self.num_episodes = 100
        self.isaac_root = os.environ.get("ISAACSIM_PATH")

        print("[DEBUG] Calling load_config()", flush=True)
        self.load_config()
        print("[DEBUG] Config loaded. Headless =", self.headless, flush=True)

        try:
            print("[DEBUG] Importing Environment and Agents", flush=True)
            from world import Environment
            from PPO import PPOAgent
            from Dp3d import DDPGAgent

            self.Environment = Environment
            self.agent_classes = {
                "ppo": PPOAgent,
                "dp3d": DDPGAgent,
            }
        except Exception as e:
            print("[ERROR] Failed to import environment or agents", flush=True)
            traceback.print_exc()

    def wait_for_stage_ready(self, timeout=10.0):
        print("[DEBUG] Waiting for stage to be ready...", flush=True)
        import omni.kit.app
        from isaacsim.core.utils.stage import is_stage_loading
        app = omni.kit.app.get_app()
        timeline = omni.timeline.get_timeline_interface()

        t0 = time.time()
        while is_stage_loading() or not timeline:
            if time.time() - t0 > timeout:
                raise RuntimeError("Timeout waiting for stage to be ready")
            print("[ENV] Waiting for stage...", flush=True)
            app.update()
            time.sleep(0.1)
        print("[DEBUG] Stage is ready.", flush=True)

    def load_config(self):
        print(f"[DEBUG] Loading config file: {self.config_file}", flush=True)
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Parameter file '{self.config_file}' not found.")

        with open(self.config_file, "r") as f:
            config = json.load(f)

        self.all_weights = config["params"]
        self.all_joint_states = config["joint_states"]
        self.agent_algorithms = config.get("algorithms", ["ppo"] * len(self.all_weights))
        self.num_agents = config.get("num_agents", len(self.all_weights))
        self.steps_per_episode = config.get("steps_per_episode", self.steps_per_episode)
        self.num_episodes = config.get("num_episodes", self.num_episodes)
        self.headless = config.get("headless", False)

        print("[DEBUG] Config values loaded:", flush=True)
        print(json.dumps(config, indent=2), flush=True)

    def setup_environment_and_agents(self):
        print("[DEBUG] Setting up environment and agents...", flush=True)
        try:
            self.sim_env = self.Environment()
            print("[DEBUG] Environment object created", flush=True)
            self.sim_env.add_bittles(n=self.num_agents)
            print(f"[DEBUG] {self.num_agents} Bittles added", flush=True)
        except Exception as e:
            print("[ERROR] Error setting up environment or spawning agents", flush=True)
            traceback.print_exc()

        self.agents.clear()
        for i, bittle in enumerate(self.sim_env.bittlles):
            weights = self.all_weights[i]
            joint_states = self.all_joint_states[i] if i < len(self.all_joint_states) else {}
            algo = self.agent_algorithms[i].lower()

            agent_class = self.agent_classes.get(algo)
            if agent_class is None:
                raise ValueError(f"Unsupported algorithm: {algo}")

            print(f"[DEBUG] Initializing Agent {i} with algo '{algo}'", flush=True)
            agent = agent_class(weights=weights, bittle=bittle, sim_env=self.sim_env, joint_states=joint_states)
            obs, _ = agent.gym_env.reset()
            self.agents.append(agent)

        print("[DEBUG] All agents set up", flush=True)

    def train(self):
        print("[DEBUG] Training started", flush=True)
        self.wait_for_stage_ready()

        global_step = 0

        for episode in range(self.num_episodes):
            print(f"[DEBUG] Starting episode {episode + 1}/{self.num_episodes}", flush=True)
            step_count = 0

            while step_count < self.steps_per_episode:
                actions = [agent.predict_action(agent.obs) for agent in self.agents]

                for agent, action in zip(self.agents, actions):
                    agent.gym_env.step(action)

                self.sim_env.get_world().step(render=True)

                for agent, action in zip(self.agents, actions):
                    agent.post_step(action)
                    agent.train()

                step_count += 1
                global_step += 1

                if global_step % 1000 == 0:
                    print(f"[DEBUG] Saving models at global step {global_step}", flush=True)
                    for i, agent in enumerate(self.agents):
                        algo = self.agent_algorithms[i].lower()
                        path = f"{self.isaac_root}/{algo}_agent_{i}_step_{global_step}.pth"
                        print(f"[DEBUG] Saving {algo} agent {i} to {path}", flush=True)
                        agent.save(path)

            print(f"[DEBUG] Episode {episode + 1} complete. Resetting agents...", flush=True)
            for agent in self.agents:
                agent.reset()

        print("[DEBUG] Training complete. Saving final models...", flush=True)
        for i, agent in enumerate(self.agents):
            algo = self.agent_algorithms[i].lower()
            final_path = f"{self.isaac_root}/{algo}_agent_{i}_final.pth"
            print(f"[DEBUG] Saving final model for {algo} agent {i} to {final_path}", flush=True)
            agent.save(final_path)

        print("[DEBUG] Final models saved.", flush=True)



if __name__ == "__main__":
    print("[DEBUG] Starting training script", flush=True)

    try:
        simulation_app = SimulationApp({
            "headless": False,
            "hide_ui": True ,
            "window_width": 1280,
            "window_height": 720,
        })
        print("[DEBUG] SimulationApp initialized", flush=True)

        trainer = MultiAgentTrainer()
        trainer.setup_environment_and_agents()
        trainer.train()

    except Exception as e:
        print("[FATAL] Unhandled exception during training:", flush=True)
        traceback.print_exc()

    finally:
        if simulation_app:
            simulation_app.close()
            print("[DEBUG] SimulationApp closed", flush=True)
