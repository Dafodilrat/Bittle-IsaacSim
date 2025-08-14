import os
import sys
import glob
import numpy as np
import torch as th

from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from GymWrapper import gym_env
from tools import log as logger
from tools import save_checkpoint, load_checkpoint, format_joint_locks, RunLogger
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
    def post_init_(self):
        self.tensorboard_log = RunLogger(base_dir=os.path.join(os.environ.get("ISAACSIM_PATH"), "alpha", "logs"), agent_name="DDPG", joint_lock_dict=self.gym_env.joint_lock_dict)
        self.policy = self.model.policy
        self.buffer = self.model.replay_buffer
        self.obs, _ = self.gym_env.reset()
        self.dones = [False]
        self.set_lr_offpolicy(1e-4)

        if "cuda" in self.device:
            device_idx = int(self.device.split(":")[-1])
            th.cuda.set_device(device_idx)

        if not hasattr(self.model, "_logger"):
            self.model._logger = configure()  # Creates default stdout logger
            self.model._current_progress_remaining = 1.0  # or 0.5 if you want halfway-through LR

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

        self.model = DDPG(
            policy="MlpPolicy",
            env=DummyVecEnv([lambda: self.gym_env]),
            verbose=0,
            device=self.device,
            tensorboard_log=os.path.join(os.environ.get("ISAACSIM_PATH"),"alpha","logs") 
        )

        self.post_init_()

    def load_model(self, step=-1):

        ckpt = load_checkpoint("dp3d", self.gym_env.joint_lock_dict, self.save_dir, step=step)
        
        if ckpt:
            self.model=DDPG.load(ckpt["path"],env=DummyVecEnv([lambda: self.gym_env]),device=self.device)
            self.step_count = ckpt["step"]
            self.log(f"[DDPG] Loaded checkpoint from {ckpt['path']} at step {self.step_count}", flush=self.log_enabled)
            self.post_init_()
            self.tensorboard_log.set_step_offset(self.step_count)
        else:
            self.log(f"[DDPG] No chekpoint to load", flush=self.log_enabled)

    def save(self, step_increment=1, prefix="dp3d"):
        self.step_count += step_increment
        save_checkpoint(
            model=self.model,
            algo=prefix,
            joint_lock_dict=self.gym_env.joint_lock_dict,
            step_count=self.step_count,
            save_dir=self.save_dir,
            log_fn=self.log if self.log_enabled else print
        )

        obs, reward, done, info = self.gym_env.post_step()

        self.tensorboard_log.log_many(
            {
                "reward": float(reward),
                "dist_to_goal": float(info.get("distance_to_goal", 0.0)),
                "z_height": float(info.get("pose", [0,0,0])[2]) if "pose" in info else 0.0,
                "done": 1.0 if done else 0.0,
            },
            step=self.step_count
        )

    def set_lr_offpolicy(self, lr: float):
        # Make future SB3 updates use a constant LR
        self.model.lr_schedule = lambda _: lr
        # Apply immediately to current optimizers
        for opt in (self.model.actor.optimizer, self.model.critic.optimizer):
            for g in opt.param_groups:
                g["lr"] = lr

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