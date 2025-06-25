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


from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})  # start the simulation app, with GUI open

from .world import Environment
from .GymWrapper import gym_env
from gymnasium.wrappers import TimeLimit
from .Bittle import Bittle

class train():

    def __init__(self,params, bittle, env):
        
        env = gym_env(bittle = bittle ,env=env ,weights = params)
        
        # check_env(env)                                      
        env = DummyVecEnv([lambda: TimeLimit(env, max_episode_steps=500)])

        self.model = PPO(
            policy="MlpPolicy",         # Use multilayer perceptron
            env = env,                    # Your wrapped Gym env
            verbose = 1,                  # Show training info
            tensorboard_log="./ppo_logs",  # Optional
            device="cpu"
        )

    def start(self):

        print("starting the training loop", flush=True)
        # callback = StopCallback(should_stop_fn=stop_fn)
        self.model.learn(total_timesteps=1_000_000)
        self.model.save("ppo_bittle")
    

if __name__ == "__main__":
    w=Environment()
    b=Bittle()
    print("done",flush=True)
    # while simulation_app.is_running():
    #     simulation_app.update()
    t=train([100,10,10,0.5,0.2,10],b,w) 
    t.start()
