import numpy as np
#!pip install stable_baselines3
import sys
import os

sb3_path = "/home/dafodilrat/Documents/bu/RASTIC/isaac-sim-standalone@4.5.0-rc.36+release.19112.f59b3005.gl.linux-x86_64.release/kit/python/lib/python3.10/site-packages"
if sb3_path not in sys.path:
    sys.path.append(sb3_path)
    print("Manually added stable-baselines3 path to sys.path")


from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.wrappers import TimeLimit

from .GymWrapper import gym_env

class StopCallback(BaseCallback):
    def __init__(self, trainer_ref, verbose=0):
        super().__init__(verbose)
        self.trainer_ref = trainer_ref  # This should be a reference to the PPO instance

    def _on_step(self) -> bool:
        # Stop training when the flag is set
        return not self.trainer_ref.should_stop

class stb3_PPO():

    def __init__(self, params, bittle, env):
        env = gym_env(bittle=bittle, env=env, weights=params)
        env = DummyVecEnv([lambda: TimeLimit(env, max_episode_steps=500)])

        self.model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            tensorboard_log="./ppo_logs",
            device="cpu"
        )
        self.should_stop = False  # <-- This is the mutable flag used by the callback

    def start_training(self):
        print("starting the training loop", flush=True)
        callback = StopCallback(trainer_ref=self)

        try:
            self.model.learn(total_timesteps=1_000_000, callback=callback)
            self.model.save("ppo_bittle")
        except Exception as e:
            import traceback
            print("[TRAINING ERROR] Exception during training:", e)
            traceback.print_exc()
            self.should_stop = True

    def stop_training(self):
        print("Training stop requested.")
        self.should_stop = True
