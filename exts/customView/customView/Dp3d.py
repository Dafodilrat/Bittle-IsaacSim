import os
import sys
import glob
import numpy as np
import torch as th

from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from GymWrapper import gym_env
from tools import log as logger
from stable_baselines3.common.logger import configure


sb3_path = os.environ.get("ISAACSIM_PATH") + "/kit/python/lib/python3.10/site-packages"
if sb3_path not in sys.path:
    sys.path.append(sb3_path)
    print("Manually added stable-baselines3 path to sys.path")

class DDPGAgent:
    """
    Deep Deterministic Policy Gradient (DDPG) agent wrapper.
    Includes checkpointing, adaptive training step control, and buffer management.
    """

    def __init__(self, bittle, weights, sim_env, joint_states, grnd, device="cpu", log=False):
        self.should_stop = False
        self.device = device
        self.log_enabled = log
        self.log = logger
        self.save_dir = os.path.join(os.environ["ISAACSIM_PATH"], "alpha", "checkpoints")
        os.makedirs(self.save_dir, exist_ok=True)

        # === Parameters ===
        self.step_count = 0
        self.adaptive_step_scale = 1.0
        self.train_every = 1
        self.global_step = 0
        self.gradient_steps = 1

        if "cuda" in self.device:
            device_idx = int(self.device.split(":")[-1])
            th.cuda.set_device(device_idx)

        self.gym_env = gym_env(
            bittle=bittle,
            env=sim_env,
            weights=weights,
            joint_lock_dict=joint_states,
            grnd=grnd
        )

        self.model = DDPG(
            policy="MlpPolicy",
            env=DummyVecEnv([lambda: self.gym_env]),
            verbose=0,
            device=self.device
        )

        latest_ckpt = self._load_latest_checkpoint("dp3d")
        if latest_ckpt:
            self.model.set_parameters(latest_ckpt["path"])
            self.step_count = latest_ckpt["step"]
            self.log(f"[DDPG] Loaded checkpoint from {latest_ckpt['path']} at step {self.step_count}", flush=self.log_enabled)

        self.policy = self.model.policy
        self.buffer = self.model.replay_buffer
        self.obs, _ = self.gym_env.reset()
        self.dones = [False]

        if not hasattr(self.model, "_logger"):
            self.model._logger = configure()  # Creates default stdout logger
            self.model._current_progress_remaining = 1.0  # or 0.5 if you want halfway-through LR


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
        self.buffer.add(obs, next_obs, action, reward, done, [{}])

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
        """
        Trigger training step if replay buffer has enough data.
        Scales number of gradient steps with buffer size.
        """
        self.global_step += 1
        if self.buffer.size() >= self.model.batch_size:
            if self.global_step % self.train_every == 0:
                # Adapt number of gradient steps based on buffer fill ratio
                fill_ratio = self.buffer.size() / self.buffer.buffer_size
                self.gradient_steps = int(self.adaptive_step_scale * fill_ratio * 10) + 1

                self.model.train(batch_size=self.model.batch_size,gradient_steps=self.gradient_steps)

                self.log(f"[DDPG] Trained {self.gradient_steps}x at step {self.global_step}", flush=self.log_enabled)

    def stop_training(self):
        self.should_stop = True

    def save(self, step_increment=1, prefix="dp3d"):
        self.step_count += step_increment
        path = os.path.join(self.save_dir, f"{prefix}_step_{self.step_count}.pth")
        self.model.save(path)
        self.log(f"[DDPG] Saved model to {path}", flush=self.log_enabled)
