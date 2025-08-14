from omni.isaac.kit import SimulationApp
import os
import json
import traceback
import omni.kit.app
import torch as th


class MultiAgentTrainer:

    def __init__(self, config_file: str = "params.json"):
        self.agents = []
        self.sim_env = None
        self.steps_per_episode = 1000
        self.num_episodes = 10000
        self.render_every = 4            # throttle rendering (1 = every frame)

        # Checkpointing
        self.save_dir = os.path.join(os.environ.get("ISAACSIM_PATH", "."), "alpha", "checkpoints")
        os.makedirs(self.save_dir, exist_ok=True)

        # Early stopping (per agent)
        self.early_stop_patience = 10     # episodes in each window
        self.early_stop_threshold = 1e-3  # minimum improvement between windows
        self.agent_rewards_history = []   # per-agent list of episode returns
        self.agent_stop_flags = []        # per-agent boolean flags

        # === Load GUI config ===
        self.config_file = config_file
        self._load_config()

        self._launch_simulation()
        self._init_agent_classes()

    # ---------------------- Config ----------------------
    def _load_config(self):
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Parameter file '{self.config_file}' not found.")
        with open(self.config_file, "r") as f:
            cfg = json.load(f)
        # From PyQt GUI (see pyqt_interface.get_config)
        self.all_weights = cfg["params"]
        self.all_joint_states = cfg["joint_states"]
        self.agent_algorithms = cfg.get("algorithms", ["ppo"] * len(self.all_weights))
        self.num_agents = cfg.get("num_agents", len(self.all_weights))
        self.headless = cfg.get("headless", False)
        # Optional fields we ignore here: training_mode, demo_ckpt_step

    # ---------------------- Boot / Utilities ----------------------
    def _launch_simulation(self):
        SimulationApp({"headless": self.headless, "renderer": "None" if self.headless else "Hybrid"})
        from tools import get_free_gpu, ensure_dir_exists, log, wait_for_stage_ready
        self.get_free_gpu = get_free_gpu
        self.ensure_dir_exists = ensure_dir_exists
        self.log = log
        self.wait_for_stage_ready = wait_for_stage_ready
        self.ensure_dir_exists(self.save_dir)

    def _init_agent_classes(self):
        from PPO import PPOAgent
        from Dp3d import DDPGAgent
        from Td3 import TD3Agent
        from A2C import A2CAgent
        self.agent_classes = {"ppo": PPOAgent, "dp3d": DDPGAgent, "td3": TD3Agent, "a2c": A2CAgent}

    # ---------------------- Setup ----------------------
    def setup_environment_and_agents(self):
        from environment import Environment

        self.sim_env = Environment()
        self.sim_env.add_training_grounds(n=self.num_agents, size=30.0)
        self.sim_env.add_bittles(n=self.num_agents)

        self.agents.clear()
        self.agent_rewards_history = []
        self.agent_stop_flags = []

        for i, bittle in enumerate(self.sim_env.bittles):
            algo = self.agent_algorithms[i].lower()
            weights = self.all_weights[i]
            joint_states = self.all_joint_states[i] if i < len(self.all_joint_states) else {}

            agent_class = self.agent_classes[algo]
            agent = agent_class(
                weights=weights,
                bittle=bittle,
                sim_env=self.sim_env,
                joint_states=joint_states,
                grnd=self.sim_env.training_grounds[i],
                device=self.get_free_gpu(),
                log=False,
            )
            agent.load_model(step=-1)
            agent.reset()
            self.agents.append(agent)
            self.agent_rewards_history.append([])
            self.agent_stop_flags.append(False)

    # ---------------------- Early stopping ----------------------
    def _should_stop_agent(self, agent_idx):
        rewards = self.agent_rewards_history[agent_idx]
        k = self.early_stop_patience
        if len(rewards) < 2 * k:
            return False
        recent_avg = sum(rewards[-k:]) / k
        prev_avg = sum(rewards[-2*k:-k]) / k
        return abs(recent_avg - prev_avg) < self.early_stop_threshold

    # ---------------------- Logging ----------------------

    def _log_agent_info(self, agent_idx, agent, prefix="", episode_reward=None):
        """Logs agent position, goal, and optionally reward/distance to goal."""
        info = agent.gym_env.generate_info()
        self.log(f"[Agent {agent_idx}] {prefix} Info:", True)
        self.log(f"  Position         : {info['pose']}", True)
        self.log(f"  Goal             : {info['goal']}", True)
        self.log(f"  Distance to Goal : {info.get('distance_to_goal', 0):.2f}", True)
        if episode_reward is not None:
            self.log(f"  Total Reward     : {episode_reward:.2f}", True)

    # ---------------------- Training loop ----------------------
    def train(self):
        self.wait_for_stage_ready()
        world = self.sim_env.get_world()

        try:
            for ep in range(self.num_episodes):
                ep_returns = [0.0 for _ in self.agents]
                step = 0

                # Log episode start state
                for i, agent in enumerate(self.agents):
                    if not self.agent_stop_flags[i]:
                        self._log_agent_info(i, agent, prefix=f"Episode {ep+1} START")

                while step < self.steps_per_episode:
                    with th.no_grad():
                        actions = [agent.predict_action(agent.obs) for agent in self.agents]

                    for i, (agent, action) in enumerate(zip(self.agents, actions)):
                        if not self.agent_stop_flags[i]:
                            agent.gym_env.step(action)

                    do_render = (not self.headless) and (step % self.render_every == 0)
                    world.step(render=do_render)

                    for i, (agent, action) in enumerate(zip(self.agents, actions)):
                        if self.agent_stop_flags[i]:
                            continue
                        agent.post_step(action)
                        agent.train()
                        ep_returns[i] += agent.gym_env._last_reward

                    step += 1

                # End-of-episode logging + early stop checks + save
                for i, agent in enumerate(self.agents):
                    if not self.agent_stop_flags[i]:
                        self.agent_rewards_history[i].append(ep_returns[i])
                        if self._should_stop_agent(i):
                            self.agent_stop_flags[i] = True
                            self.log(f"[EARLY-STOP] Agent {i} halted (Δ<{self.early_stop_threshold}).", True)

                    algo = agent.__class__.__name__.replace("Agent", "").lower()
                    agent.save(step_increment=self.steps_per_episode, prefix=algo)

                    # Log summary
                    self._log_agent_info(i, agent, prefix=f"Episode {ep+1} END", episode_reward=ep_returns[i])

                # Compact episode summary
                self.log(
                    "[TRAIN] Ep {}/{} — ".format(ep + 1, self.num_episodes) +
                    ", ".join(f"A{i}:{r:.1f}{'*' if self.agent_stop_flags[i] else ''}"
                            for i, r in enumerate(ep_returns)),
                    True,
                )

                if all(self.agent_stop_flags):
                    self.log("[TRAIN] All agents satisfied early-stopping — ending.", True)
                    break

                for i, agent in enumerate(self.agents):
                    if not self.agent_stop_flags[i]:
                        agent.reset()

        except KeyboardInterrupt:
            self.log("[TRAIN] Interrupted — saving & exiting.", True)
        except Exception as e:
            self.log(f"[TRAIN] Exception: {e}", True)
            traceback.print_exc()
        finally:
            for i, agent in enumerate(self.agents):
                algo = agent.__class__.__name__.replace("Agent", "").lower()
                agent.save(step_increment=0, prefix=algo)
            self.log("[TRAIN] Final checkpoints saved.", True)


if __name__ == "__main__":
    trainer = MultiAgentTrainer("params.json")
    trainer.setup_environment_and_agents()
    trainer.train()
