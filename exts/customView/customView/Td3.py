import os
import sys
import glob
import numpy as np
import torch as th

from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from GymWrapper import gym_env
from tools import log as logger
from tools import save_checkpoint, load_latest_checkpoint, format_joint_locks
from stable_baselines3.common.logger import configure


sb3_path = os.environ.get("ISAACSIM_PATH") + "/kit/python/lib/python3.10/site-packages"
if sb3_path not in sys.path:
    sys.path.append(sb3_path)
    print("Manually added stable-baselines3 path to sys.path")

class TD3Agent:
    """
    Twin Delayed DDPG (TD3) agent wrapper.
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
            self.log(f"[TD3] Loaded checkpoint from {latest_ckpt['path']} at step {self.step_count}", flush=self.log_enabled)

        self.policy = self.model.policy
        self.buffer = self.model.replay_buffer
        self.obs, _ = self.gym_env.reset()
        self.dones = [False]

        if not hasattr(self.model, "_logger"):
            self.model._logger = configure()  # Creates default stdout logger
            self.model._current_progress_remaining = 1.0  # or 0.5 if you want halfway-through LR

    def save(self, step_increment=1, prefix="td3"):
        self.step_count += step_increment
        save_checkpoint(
            model=self.model,
            algo=prefix,
            joint_lock_dict=self.gym_env.joint_lock_dict,
            step_count=self.step_count,
            save_dir=self.save_dir,
            log_fn=self.log if self.log_enabled else print
        )

    def _load_latest_checkpoint(self, prefix):
        ckpt = load_latest_checkpoint(
            algo=prefix,
            joint_lock_dict=self.gym_env.joint_lock_dict,
            save_dir=self.save_dir
        )
        return ckpt

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
                fill_ratio = self.buffer.size() / self.buffer.buffer_size
                self.gradient_steps = int(self.adaptive_step_scale * fill_ratio * 10) + 1

                # Inject logger if missing (fixes the AttributeError)
                from stable_baselines3.common.logger import configure
                if not hasattr(self.model, "_logger"):
                    self.model._logger = configure()
                    self.model._current_progress_remaining = 1.0  # Assume full progress for now

                self.model.train(gradient_steps=self.gradient_steps,batch_size=self.model.batch_size)

                self.log(f"[TD3] Trained {self.gradient_steps}x at step {self.global_step}", flush=self.log_enabled)

    def stop_training(self):
        self.should_stop = True

