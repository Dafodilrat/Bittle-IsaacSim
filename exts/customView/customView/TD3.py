import os
import sys
import glob
import numpy as np
import torch as th

from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from GymWrapper import gym_env
from tools import log

sb3_path = os.environ.get("ISAACSIM_PATH") + "/kit/python/lib/python3.10/site-packages"
if sb3_path not in sys.path:
    sys.path.append(sb3_path)
    print("Manually added stable-baselines3 path to sys.path")

class TD3Agent:
    def __init__(self, bittle, weights, sim_env, joint_states, grnd, device="cpu", log=False):
        self.should_stop = False
        self.device = device
        self.log_enabled = log
        self.save_dir = os.path.join(os.environ["ISAACSIM_PATH"], "alpha", "checkpoints")
        os.makedirs(self.save_dir, exist_ok=True)

        self.step_count = 0
        self.gym_env = gym_env(
            bittle=bittle,
            env=sim_env,
            weights=weights,
            joint_lock_dict=joint_states,
            grnd=grnd
        )

        self.model = TD3(
            policy="MlpPolicy",
            env=DummyVecEnv([lambda: self.gym_env]),
            verbose=0,
            device=self.device
        )

        latest_ckpt = self._load_latest_checkpoint("td3")
        if latest_ckpt:
            self.model.set_parameters(latest_ckpt["path"])
            self.step_count = latest_ckpt["step"]
            self.log(f"[TD3] Loaded checkpoint from {latest_ckpt['path']} at step {self.step_count}", flush=True)

        self.policy = self.model.policy
        self.buffer = self.model.replay_buffer
        self.obs, _ = self.gym_env.reset()
        self.dones = [False]

    def log(self, *args, **kwargs):
        if self.log_enabled:
            print(*args, **kwargs)

    def _load_latest_checkpoint(self, prefix):
        files = glob.glob(os.path.join(self.save_dir, f"{prefix}_step_*.pth"))
        if not files:
            return None
        files.sort(key=lambda p: int(p.split("_step_")[-1].split(".")[0]), reverse=True)
        path = files[0]
        step = int(path.split("_step_")[-1].split(".")[0])
        return {"path": path, "step": int(step)}

    def predict_action(self, obs):
        action, _ = self.policy.predict(obs, deterministic=False)
        return action

    def add_to_buffer(self, obs, action, reward, done, next_obs):
        self.buffer.add(obs, next_obs, action, reward, done)

    def reset(self):
        self.obs, _ = self.gym_env.reset()

    def post_step(self, action):
        obs, reward, done, info = self.gym_env.post_step()
        self.add_to_buffer(self.obs, action, reward, done, obs)
        self.obs = obs
        if done:
            self.obs, _ = self.gym_env.reset()

    def step(self, action, sim_step_fn=None):
        self.gym_env.step(action)
        if sim_step_fn:
            sim_step_fn()
            self.post_step(action)

    def train(self):
        # Train the model if enough transitions are collected
        if self.buffer.size() >= self.model.batch_size:
            self.model.train(batch_size=self.model.batch_size)

    def stop_training(self):
        self.should_stop = True

    def save(self, step_increment=1, prefix="td3"):
        self.step_count += step_increment
        path = os.path.join(self.save_dir, f"{prefix}_step_{self.step_count}.pth")
        self.model.save(path)
        self.log(f"[TD3] Saved model to {path}", flush=True)
