from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})  # GUI-enabled

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
        config = json.load(f)

    param_dict = config["params"]
    joint_states = config.get("joint_states")  # default to empty if not provided

    # Maintain the order expected by PPO
    ordered_keys = [
        "Correct Posture Bonus",
        "Smooth Bonus Weight",
        "Incorrect Posture Penalty",
        "Jerking Movement Penalty (x10)",
        "High Joint Velocity Penalty (x10)",
        "Distance to Goal Penalty"
    ]

    # Scale ×10 values already handled in GUI, so no need to rescale here
    weights = [param_dict[k] for k in ordered_keys]

    # --- Initialize environment and agents ---
    agents = []
    w = Environment()
    n = 1
    w.add_bittles(n=n)

    print("Environment setup complete.", flush=True)

    for bittle in w.bittlles:
        
        t = stb3_PPO(weights=weights, bittle=bittle, env=w, joints=joint_states)
        agents.append(t)

    for agent in agents:
        agent.start_training()
