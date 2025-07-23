from omni.isaac.kit import SimulationApp

import os
import json
import time
import traceback
import omni.kit.app
import pynvml
import torch as th

class MultiAgentTrainer:

    def __init__(self, config_file="params.json"):
        print("[DEBUG] MultiAgentTrainer.__init__ started", flush=True)

        self.config_file = config_file
        self.agents = []
        self.sim_env = None
        self.steps_per_episode = 1000
        self.num_episodes = 100
        self.save_file = os.path.join(os.environ.get("ISAACSIM_PATH"), "alpha", "checkpoints")
        self.save_step = 10000
        os.makedirs(self.save_file, exist_ok=True)

        # Load config to determine headless/rendering behavior
        self.load_config()

        # Setup SimulationApp after config is loaded
        renderer_mode = "None" if self.headless else "Hybrid"
        
        SimulationApp({
            "headless": self.headless,
            "renderer": renderer_mode,
            "hide_ui": False,  # Only hide UI if headless
            "window_width": 1280,
            "window_height": 720,
        })
        print(f"[DEBUG] SimulationApp initialized (headless={self.headless}, renderer={renderer_mode})", flush=True)

        from PPO import PPOAgent
        from Dp3d import DDPGAgent

        self.agent_classes = {
            "ppo": PPOAgent,
            "dp3d": DDPGAgent,
        }

    def select_training_gpu(self):
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()

        free_gpus = []
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            used = mem_info.used / mem_info.total
            free_gpus.append((i, used))
        
            best_gpu = sorted(free_gpus, key=lambda x: x[1])[0][0]
            pynvml.nvmlShutdown()
            return f"cuda:{best_gpu}" if th.cuda.is_available() else "cpu"
        
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
        
        from environment import Environment

        print("[DEBUG] Setting up environment and agents...", flush=True)
        
        try:
            
            self.sim_env = Environment()
            print("[DEBUG] Environment object created", flush=True)
            
            self.sim_env.add_training_grounds(n=self.num_agents, size=20.0)
            print(f"[DEBUG] {len(self.sim_env.training_grounds)} training grounds added", flush=True)

            self.sim_env.add_bittles(n=self.num_agents)
            print(f"[DEBUG] {self.num_agents} Bittles added", flush=True)
        
        except Exception as e:
            print("[ERROR] Error setting up environment or spawning agents", flush=True)
            traceback.print_exc()

        self.agents.clear()

        for i, bittle in enumerate(self.sim_env.bittles):

            weights = self.all_weights[i]
            joint_states = self.all_joint_states[i] if i < len(self.all_joint_states) else {}
            algo = self.agent_algorithms[i].lower()

            agent_class = self.agent_classes.get(algo)
            if agent_class is None:
                raise ValueError(f"Unsupported algorithm: {algo}")

            print(f"[DEBUG] Initializing Agent {i} with algo '{algo}'", flush=True)
            agent = agent_class(weights=weights, bittle=bittle, sim_env=self.sim_env, joint_states=joint_states, grnd=self.sim_env.training_grounds[i] ,device=self.select_training_gpu(),log=True)
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
            episode_rewards = [0.0 for _ in self.agents]

            # Log start-of-episode info
            for i, agent in enumerate(self.agents):
                info = agent.gym_env.generate_info()
                print(f"[Agent {i}] Start Info:", flush=True)
                print(f"  Position : {info['pose']}", flush=True)
                print(f"  Goal     : {info['goal']}", flush=True)

            while step_count < self.steps_per_episode:
                actions = [agent.predict_action(agent.obs) for agent in self.agents]

                for agent, action in zip(self.agents, actions):
                    agent.gym_env.step(action)

                self.sim_env.get_world().step(render=True)

                for i, (agent, action) in enumerate(zip(self.agents, actions)):
                    agent.post_step(action)
                    agent.train()
                    episode_rewards[i] += agent.gym_env._last_reward  # reward from gym_env.post_step()

                step_count += 1
                global_step += 1

                if global_step % self.save_step == 0:
                    print(f"[DEBUG] Saving models at global step {global_step}", flush=True)
                    for i, agent in enumerate(self.agents):

                        agent.save(step_increment=self.save_step)

            # Log end-of-episode summary
            for i, agent in enumerate(self.agents):
                info = agent.gym_env.generate_info()
                print(f"[Agent {i}] Episode {episode + 1} Summary:", flush=True)
                print(f"  Final Position      : {info['pose']}", flush=True)
                print(f"  Distance to Goal    : {info['distance_to_goal']:.2f}", flush=True)
                print(f"  Total Episode Reward: {episode_rewards[i]:.2f}", flush=True)

            # Reset all agents
            print(f"[DEBUG] Episode {episode + 1} complete. Resetting agents...", flush=True)
            for agent in self.agents:
                agent.reset()

        print("[DEBUG] Training complete. Saving final models...", flush=True)
        for i, agent in enumerate(self.agents):
            algo = self.agent_algorithms[i].lower()
            final_path = f"{self.save_file}/{algo}_step_{global_step}.pth"
            print(f"[DEBUG] Saving final model for {algo} agent {i} to {final_path}", flush=True)
            agent.save(final_path)


        print("[DEBUG] Final models saved.", flush=True)



if __name__ == "__main__":
    print("[DEBUG] Starting training script", flush=True)

    try:
        trainer = MultiAgentTrainer()
        trainer.setup_environment_and_agents()
        
        # while omni.kit.app.get_app().is_running():
        #     omni.kit.app.get_app().update()

        trainer.train()

    except Exception as e:
        print("[FATAL] Unhandled exception during training:", flush=True)
        traceback.print_exc()

