from omni.isaac.kit import SimulationApp
import os
import json
import time

class MultiAgentTrainer:
    def __init__(self, config_file="params.json", headless=False):
        
        self.config_file = config_file
        self.agents = []
        self.sim_env = None
        self.steps_per_episode = 500
        self.num_episodes = 100
        
        self.load_config()

        self.sim_app = SimulationApp({
            "headless": self.headless,
            "hide_ui": True,
            "window_width": 1280,
            "window_height": 720,
            "width": 1280,
            "height": 720
        })

        from world import Environment
        from PPO import PPOAgent
        from Dp3d import DDPGAgent

        self.Environment = Environment
        self.agent_classes = {
            "ppo": PPOAgent,
            "dp3d": DDPGAgent,
        }

    def wait_for_stage_ready(self, timeout=10.0):
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

    def load_config(self):
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

    def setup_environment_and_agents(self):
        self.sim_env = self.Environment()
        self.sim_env.add_bittles(n=self.num_agents)

        self.agents.clear()
        for i, bittle in enumerate(self.sim_env.bittlles):
            weights = self.all_weights[i]
            joint_states = self.all_joint_states[i] if i < len(self.all_joint_states) else {}
            algo = self.agent_algorithms[i].lower()

            agent_class = self.agent_classes.get(algo)
            if agent_class is None:
                raise ValueError(f"Unsupported algorithm: {algo}")

            agent = agent_class(weights=weights, bittle=bittle, sim_env=self.sim_env, joint_states=joint_states)
            obs, _ = agent.gym_env.reset()
            # agent.set_obs(obs)
            self.agents.append(agent)

    def train(self):
        self.wait_for_stage_ready()

        for episode in range(self.num_episodes):
            print(f"Starting episode {episode + 1}/{self.num_episodes}")
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

            for agent in self.agents:
                agent.reset()

        print("Training complete.")
        for i, agent in enumerate(self.agents):
            agent.save(f"ppo_bittle_agent_{i}")

if __name__ == "__main__":
    trainer = MultiAgentTrainer()
    trainer.setup_environment_and_agents()
    trainer.train()
