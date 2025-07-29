import os
import sys
import glob
from cv2 import log
import numpy as np
import torch as th

from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from GymWrapper import gym_env
from tools import log as logger

sb3_path = os.environ.get("ISAACSIM_PATH") + "/kit/python/lib/python3.10/site-packages"
if sb3_path not in sys.path:
    sys.path.append(sb3_path)
    print("Manually added stable-baselines3 path to sys.path")

class A2CAgent:
    def __init__(self, bittle, weights, sim_env, joint_states, grnd, device="cpu", log = False):
        self.should_stop = False
        self.device = "cpu"
        self.save_dir = os.path.join(os.environ["ISAACSIM_PATH"], "alpha", "checkpoints")
        self.log = logger
        self.log_enabled = log
        os.makedirs(self.save_dir, exist_ok=True)

        self.step_count = 0
        self.gym_env = gym_env(
            bittle=bittle,
            env=sim_env,
            weights=weights,
            joint_lock_dict=joint_states,
            grnd=grnd
        )

        if "cuda" in self.device:
            device_idx = int(self.device.split(":")[-1])
            th.cuda.set_device(device_idx)

        self.model = A2C(
            policy="MlpPolicy",
            env=DummyVecEnv([lambda: self.gym_env]),
            verbose=0,
            device="cpu"
        )

        latest_ckpt = self._load_latest_checkpoint("a2c")
        if latest_ckpt:
            self.model.set_parameters(latest_ckpt["path"])
            self.step_count = latest_ckpt["step"]
            self.log(f"[A2C] Loaded checkpoint from {latest_ckpt['path']} at step {self.step_count}", flush=self.log_enabled)

        self.policy = self.model.policy
        self.buffer = self.model.rollout_buffer
        self.obs, _ = self.gym_env.reset()
        self.dones = [False]

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

    def add_to_buffer(self, obs, action, reward, done, value=None, log_prob=None):
        obs_tensor = th.as_tensor(obs).float().to(self.model.device).unsqueeze(0)

        if value is None:
            value = self.policy.predict_values(obs_tensor)

        if log_prob is None:
            action_tensor = th.as_tensor(action).float().to(self.model.device).unsqueeze(0)
            log_prob = self.policy.get_distribution(obs_tensor).log_prob(action_tensor)

        self.buffer.add(obs, action, reward, done, value.detach(), log_prob.detach())

    def reset(self):
        self.obs, _ = self.gym_env.reset()

    def post_step(self, action):
        obs, reward, done, info = self.gym_env.post_step()
        self.add_to_buffer(self.obs, action, reward, done)
        self.obs = obs
        if done:
            self.obs, _ = self.gym_env.reset()

    def step(self, action, sim_step_fn=None):
        self.gym_env.step(action)
        if sim_step_fn:
            sim_step_fn()
            self.post_step(action)

    def train(self):
        if self.buffer.full:
            self.model.policy.train()
            self.buffer.reset()

    def stop_training(self):
        self.should_stop = True

    def save(self, step_increment=1, prefix="a2c"):
        self.step_count += step_increment
        path = os.path.join(self.save_dir, f"{prefix}_step_{self.step_count}.pth")
        self.model.save(path)
        self.log(f"[A2C] Saved model to {path}", flush=self.log_enabled)
