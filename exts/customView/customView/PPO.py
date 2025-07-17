import numpy as np
#!pip install stable_baselines3
import sys
import os

sb3_path =  os.environ.get("ISAACSIM_PATH")+"/kit/python/lib/python3.10/site-packages"
if sb3_path not in sys.path:
    sys.path.append(sb3_path)
    print("Manually added stable-baselines3 path to sys.path")

# ppo_agent.py

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from GymWrapper import gym_env
import torch as th



class PPOAgent:
    def __init__(self, bittle, weights, sim_env, joint_states, device="cpu"):
        self.should_stop = False

        print("[PPO] set to device:", device, flush=True)

        self.gym_env = gym_env(
            bittle=bittle,
            env=sim_env,
            weights=weights,
            joint_lock_dict=joint_states
        )
        # Initialize PPO model
        self.model = PPO(
            policy="MlpPolicy",
            env=DummyVecEnv([lambda: self.gym_env]),
            verbose=0,
            device=device,
        )

        # Easy access
        self.policy = self.model.policy
        self.buffer = self.model.rollout_buffer

        # Used to track current rollout step
        self.obs,_ = self.gym_env.reset()
        self.dones = [False]

    def predict_action(self, obs):
        """Runs policy to predict an action from current observation"""
        action, _ = self.policy.predict(obs, deterministic=False)
        return action


    def add_to_buffer(self, obs, action, reward, done, value=None, log_prob=None):
        obs_tensor = th.as_tensor(obs).float().to(self.model.device).unsqueeze(0)

        if value is None:
            value = self.policy.predict_values(obs_tensor)

        if log_prob is None:
            action_tensor = th.as_tensor(action).float().to(self.model.device).unsqueeze(0)
            log_prob = self.policy.get_distribution(obs_tensor).log_prob(action_tensor)

        value = value.detach()
        log_prob = log_prob.detach()

        self.buffer.add(obs, action, reward, done, value, log_prob)

    def reset(self):
        self.obs, _ = self.gym_env.reset()

    def train_if_ready(self):
        """Train only if rollout buffer is full"""
        if self.buffer.full:
            self.model.train()
            self.buffer.reset()

    def stop_training(self):
        self.should_stop = True

    def save(self, path):
        self.model.save(path)

    def post_step(self, action):
        """Performs post-step processing and adds to buffer."""
        obs, reward, done, info = self.gym_env.post_step()

        self.add_to_buffer(
            obs=self.obs,      # previous observation
            action=action,
            reward=reward,
            done=done
        )

        self.obs = obs      # update current obs

        if done:
            self.obs, _ = self.gym_env.reset()


    def step(self, action, sim_step_fn=None):
        """Performs a full agent-environment step cycle:
        - Applies action
        - (Optionally) steps simulation
        - Performs post-step and buffer updates
        """
        self.gym_env.step(action)

        if sim_step_fn is not None:
            sim_step_fn()
            self.post_step(action)
    
    def train(self):
        """Train if rollout buffer is full"""
        if self.buffer.full:
            self.model.policy.train()
            self.buffer.reset()