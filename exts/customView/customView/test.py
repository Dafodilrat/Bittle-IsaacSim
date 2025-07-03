
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})  # start the simulation app, with GUI open

from isaacsim import SimulationApp
from PPO import stb3_PPO
from world import Environment

import json
import os

if __name__ == "__main__":

    # --- Load training parameters from JSON ---
    param_file = "params.json"
    if not os.path.exists(param_file):
        raise FileNotFoundError(f"Parameter file '{param_file}' not found.")

    with open(param_file, "r") as f:
        param_dict = json.load(f)

    # Maintain the order expected by PPO
    ordered_keys = [
        "Correct Posture Bonus",
        "Smooth Bonus Weight",
        "Incorrect Posture Penalty",
        "Jerking Movement Penalty (x10)",
        "High Joint Velocity Penalty (x10)",
        "Distance to Goal Penalty"
    ]

    # Scale ×10 values back to normal
    params = []
    for key in ordered_keys:
        value = param_dict[key]
        params.append(value)

    # --- Initialize environment and agents ---
    agents = []
    w = Environment()
    n = 1
    w.add_bittles(n=n)

    print("Environment setup complete.", flush=True)

    for bittle in w.bittlles:
        t = stb3_PPO(params=params, bittle=bittle, env=w)
        agents.append(t)

    for agent in agents:
        agent.start_training()
