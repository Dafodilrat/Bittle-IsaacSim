import os
import sys
import glob
import numpy as np
import torch as th

from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from GymWrapper import gym_env

sb3_path = os.environ.get("ISAACSIM_PATH") + "/kit/python/lib/python3.10/site-packages"
if sb3_path not in sys.path:
    sys.path.append(sb3_path)
    print("Manually added stable-baselines3 path to sys.path")

class DDPGAgent:
    def __init__(self, bittle, weights, sim_env, joint_states, device="cpu"):
        self.device = device
        self.should_stop = False
        self.gradient_steps = 1
        self.save_dir = os.path.join(os.environ["ISAACSIM_PATH"], "alpha", "checkpoints")
        os.makedirs(self.save_dir, exist_ok=True)

        self.step_count = 0
        self.gym_env = gym_env(
            bittle=bittle,
            env=sim_env,
            weights=weights,
            joint_lock_dict=joint_states
        )

        self.model = DDPG(
            policy="MlpPolicy",
            env=DummyVecEnv([lambda: self.gym_env]),
            verbose=0,
            device=self.device,
        )

        latest_ckpt = self._load_latest_checkpoint("dp3d")
        if latest_ckpt:
            self.model.set_parameters(latest_ckpt["path"])
            self.step_count = latest_ckpt["step"]
            print(f"[DDPG] Loaded checkpoint from {latest_ckpt['path']} at step {self.step_count}", flush=True)

        self.policy = self.model.policy
        self.buffer = self.model.replay_buffer
        self.obs, _ = self.gym_env.reset()
        self.dones = [False]

    def _load_latest_checkpoint(self, prefix):
        files = glob.glob(os.path.join(self.save_dir, f"{prefix}_step_*.pth"))
        if not files:
            return None
        files.sort(key=lambda p: int(p.split("_step_")[-1].split(".")[0]), reverse=True)
        path = files[0]
        step = int(path.split("_step_")[-1].split(".")[0])
        return {"path": path, "step": step}

    def predict_action(self, obs):
        action, _ = self.policy.predict(obs, deterministic=False)
        return action

    def add_to_buffer(self, obs, action, reward, done):
        obs_prev = self.gym_env.get_previous_observation()
        obs_next = self.gym_env.get_current_observation()

        obs_np = np.asarray(obs_prev, dtype=np.float32).reshape(1, -1)
        next_obs_np = np.asarray(obs_next, dtype=np.float32).reshape(1, -1)
        action_np = np.asarray(action, dtype=np.float32).reshape(1, -1)
        reward_np = np.asarray([reward], dtype=np.float32)
        done_np = np.asarray([done], dtype=np.float32)

        self.buffer.add(obs_np, next_obs_np, action_np, reward_np, done_np, [{}])
        self.obs = obs_next

        if done:
            self.obs, _ = self.gym_env.reset()

    def train_if_ready(self):
        if self.buffer.size() >= self.model.batch_size:
            self.model.learn(total_timesteps=self.gradient_steps, reset_num_timesteps=False)
            self.buffer.reset()

    def post_step(self, action):
        obs, reward, done, info = self.gym_env.post_step()
        self.add_to_buffer(self.obs, action, reward, done)

    def step(self, action, sim_step_fn=None):
        self.gym_env.step(action)
        if sim_step_fn:
            sim_step_fn()
            self.post_step(action)

    def train(self):
        self.train_if_ready()

    def reset(self):
        self.obs, _ = self.gym_env.reset()
        self.buffer.reset()

    def stop_training(self):
        self.should_stop = True

    def save(self, step_increment=1, prefix="dp3d"):
        self.step_count += step_increment
        path = os.path.join(self.save_dir, f"{prefix}_step_{self.step_count}.zip")
        self.model.save(path)
        print(f"[DDPG] Saved model to {path}", flush=True)
