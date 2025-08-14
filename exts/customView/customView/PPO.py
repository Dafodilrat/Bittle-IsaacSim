import os
import sys
import glob
import numpy as np
import torch as th

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from GymWrapper import gym_env
from tools import log as logger
from tools import save_checkpoint, load_checkpoint, format_joint_locks, RunLogger
from stable_baselines3.common.callbacks import CheckpointCallback

# Ensure stable-baselines3 is loaded from Isaac Sim path if needed
sb3_path = os.environ.get("ISAACSIM_PATH") + "/kit/python/lib/python3.10/site-packages"
if sb3_path not in sys.path:
    sys.path.append(sb3_path)
    print("Manually added stable-baselines3 path to sys.path")

class PPOAgent:

    def post_init_(self):

        self.tensorboard_log = RunLogger(base_dir=os.path.join(os.environ.get("ISAACSIM_PATH"), "alpha", "logs"), 
                                         agent_name="PPO", 
                                         joint_lock_dict=self.gym_env.joint_lock_dict)
        self.policy = self.model.policy
        self.buffer = self.model.rollout_buffer
        self.obs, _ = self.gym_env.reset()
        self.dones = [False]
        self.set_lr_onpolicy(1e-4)

    def __init__(self, bittle, weights, sim_env, joint_states, grnd, device="cpu", log=True):
        # Initialize PPO agent with Gym environment and checkpoint management
        self.should_stop = False
        self.device = "cpu"
        self.log = logger
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

        if "cuda" in self.device:
            device_idx = int(self.device.split(":")[-1])
            th.cuda.set_device(device_idx)
        
        self.model = PPO(
            policy="MlpPolicy",
            env=DummyVecEnv([lambda: self.gym_env]),
            verbose=0,
            device="cpu"
        )

        self.post_init_()

    def load_model(self, step=-1):

        ckpt = load_checkpoint("ppo", self.gym_env.joint_lock_dict, self.save_dir, step=step)
        
        if ckpt:
            self.model=PPO.load(ckpt["path"],env=DummyVecEnv([lambda: self.gym_env]),device="cpu")
            self.step_count = ckpt["step"]
            self.log(f"[PPO] Loaded checkpoint from {ckpt['path']} at step {self.step_count}", flush=self.log_enabled)
            self.post_init_()
            self.tensorboard_log.set_step_offset(self.step_count)
        else :
            self.log(f"[PPO] No chekpoint to load", flush=self.log_enabled)

    def set_lr_onpolicy(self, lr: float):
        self.model.lr_schedule = lambda _: lr
        for g in self.model.policy.optimizer.param_groups:
            g["lr"] = lr
    
    def save(self, step_increment=1, prefix="ppo"):
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

    def predict_action(self, obs):
        # Use current policy to predict the next action
        action, _ = self.policy.predict(obs, deterministic=False)
        return action

    def add_to_buffer(self, obs, action, reward, done, value=None, log_prob=None):
        # Add transition data to rollout buffer
        obs_tensor = th.as_tensor(obs).float().to(self.model.device).unsqueeze(0)

        if value is None:
            value = self.policy.predict_values(obs_tensor)

        if log_prob is None:
            action_tensor = th.as_tensor(action).float().to(self.model.device).unsqueeze(0)
            log_prob = self.policy.get_distribution(obs_tensor).log_prob(action_tensor)

        self.buffer.add(obs, action, reward, done, value.detach(), log_prob.detach())

    def reset(self):
        # Reset environment and agent observation
        self.obs, _ = self.gym_env.reset()


    def post_step(self, action):
        # Perform post-step update and reset if done
        obs, reward, done, info = self.gym_env.post_step()
        self.add_to_buffer(self.obs, action, reward, done)
        self.obs = obs
        if done:
            self.obs, _ = self.gym_env.reset()

    def step(self, action, sim_step_fn=None):
        # Apply action to environment and run post-step processing
        self.gym_env.step(action)
        if sim_step_fn:
            sim_step_fn()
            self.post_step(action)

    def train(self):
        # Train the model if the buffer is full
        if self.buffer.full:
            self.model.policy.train()
            self.buffer.reset()

    def stop_training(self):
        # External call to stop training loop
        self.should_stop = True