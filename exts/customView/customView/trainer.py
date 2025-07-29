from omni.isaac.kit import SimulationApp
import os
import json
import time
import traceback
import omni.kit.app
import torch as th

# Delay tools import until after SimulationApp is initialized

class MultiAgentTrainer:
    """
    Orchestrates training of multiple agents in Isaac Sim using specified algorithms.
    Responsible for environment setup, agent initialization, episode scheduling, and checkpointing.
    """
    def __init__(self, config_file="params.json"):
        self.config_file = config_file
        self.agents = []
        self.sim_env = None
        self.steps_per_episode = 1000
        self.num_episodes = 10000
        self.save_file = os.path.join(os.environ.get("ISAACSIM_PATH"), "alpha", "checkpoints")
        self.save_every_n_episodes = 100  # ← Save every 10 episodes

        # === Load configuration from JSON ===
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

        # === Now safe to import tools ===
        from tools import get_free_gpu, ensure_dir_exists, log, wait_for_stage_ready
        self.get_free_gpu = get_free_gpu
        self.ensure_dir_exists = ensure_dir_exists
        self.log = log
        self.wait_for_stage_ready = wait_for_stage_ready

        self.log("[DEBUG] MultiAgentTrainer.__init__ started", True)
        self.ensure_dir_exists(self.save_file)
        self.log(f"[DEBUG] SimulationApp initialized (headless={self.headless}, renderer={renderer_mode})", True)

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

    def load_config(self):
        """Parse agent and simulation parameters from config JSON file."""
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

        self.log("[DEBUG] Setting up environment and agents...", True)

        try:
            self.sim_env = Environment()
            self.log("[DEBUG] Environment object created", True)

            self.sim_env.add_training_grounds(n=self.num_agents, size=20.0)
            self.log(f"[DEBUG] {len(self.sim_env.training_grounds)} training grounds added", True)

            self.sim_env.add_bittles(n=self.num_agents)
            self.log(f"[DEBUG] {self.num_agents} Bittles added", True)

        except Exception as e:
            self.log("[ERROR] Error setting up environment or spawning agents", True)
            traceback.print_exc()

        self.agents.clear()

        for i, bittle in enumerate(self.sim_env.bittles):
            weights = self.all_weights[i]
            joint_states = self.all_joint_states[i] if i < len(self.all_joint_states) else {}
            algo = self.agent_algorithms[i].lower()

            agent_class = self.agent_classes.get(algo)
            if agent_class is None:
                raise ValueError(f"Unsupported algorithm: {algo}")

            self.log(f"[DEBUG] Initializing Agent {i} with algo '{algo}'", True)
            agent = agent_class(
                weights=weights,
                bittle=bittle,
                sim_env=self.sim_env,
                joint_states=joint_states,
                grnd=self.sim_env.training_grounds[i],
                device=self.get_free_gpu(),
                log=False
            )
            obs, _ = agent.gym_env.reset()
            self.agents.append(agent)

        self.log("[DEBUG] All agents set up", True)

    def train(self):
        self.log("[DEBUG] Training started", True)
        self.wait_for_stage_ready()

        global_step = 0

        for episode in range(self.num_episodes):
            self.log(f"[DEBUG] Starting episode {episode + 1}/{self.num_episodes}", True)
            step_count = 0
            episode_rewards = [0.0 for _ in self.agents]

            for i, agent in enumerate(self.agents):
                info = agent.gym_env.generate_info()
                self.log(f"[Agent {i}] Start Info:", True)
                self.log(f"  Position : {info['pose']}", True)
                self.log(f"  Goal     : {info['goal']}", True)

            while step_count < self.steps_per_episode:
                actions = [agent.predict_action(agent.obs) for agent in self.agents]

                for agent, action in zip(self.agents, actions):
                    agent.gym_env.step(action)

                self.sim_env.get_world().step(render=True)

                for i, (agent, action) in enumerate(zip(self.agents, actions)):
                    agent.post_step(action)
                    agent.train()
                    episode_rewards[i] += agent.gym_env._last_reward

                step_count += 1
                global_step += 1

            if (episode + 1) % self.save_every_n_episodes == 0:
                for i, agent in enumerate(self.agents):
                    path = os.path.join(self.save_file, f"{self.agent_algorithms[i].lower()}_step_{global_step}.pth")
                    agent.model.save(path)
                    self.log(f"[DEBUG] Saved model for {self.agent_algorithms[i].lower()} agent {i} at global step {global_step} to {path}", True)

            for i, agent in enumerate(self.agents):
                info = agent.gym_env.generate_info()
                self.log(f"[Agent {i}] Episode {episode + 1} Summary:", True)
                self.log(f"  Final Position      : {info['pose']}", True)
                self.log(f"  Distance to Goal    : {info['distance_to_goal']:.2f}", True)
                self.log(f"  Total Episode Reward: {episode_rewards[i]:.2f}", True)

            self.log(f"[DEBUG] Episode {episode + 1} complete. Resetting agents...", True)
            for agent in self.agents:
                agent.reset()

        self.log("[DEBUG] Training complete. Saving final models...", True)
        for i, agent in enumerate(self.agents):
            algo = self.agent_algorithms[i].lower()
            final_path = f"{self.save_file}/{algo}_step_{global_step}.pth"
            self.log(f"[DEBUG] Saving final model for {algo} agent {i} to {final_path}", True)
            agent.save(final_path)

        self.log("[DEBUG] Final models saved.", True)


if __name__ == "__main__":
    try:
        trainer = MultiAgentTrainer()
        trainer.setup_environment_and_agents()
        trainer.train()
    except Exception as e:
        traceback.print_exc()
