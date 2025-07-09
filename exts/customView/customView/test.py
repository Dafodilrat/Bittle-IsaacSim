from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import threading
import json
import os

from PPO import stb3_PPO
from world import Environment
from GymWrapper import gym_env
from cordinator import SimulationCoordinator
import omni.kit.app
from isaacsim.core.utils.stage import is_stage_loading
import time

def wait_for_stage_ready(timeout=10.0):
    """Wait for stage to be properly loaded"""
    app = omni.kit.app.get_app()
    timeline = omni.timeline.get_timeline_interface()
    
    t0 = time.time()
    while is_stage_loading() or not timeline:
        if time.time() - t0 > timeout:
            raise RuntimeError("Timeout waiting for stage to be ready")
        print("[ENV] Waiting for stage...", flush=True)
        app.update()
        time.sleep(0.1)


if __name__ == "__main__":

    # --- Load training parameters ---
    param_file = "params.json"
    if not os.path.exists(param_file):
        raise FileNotFoundError(f"Parameter file '{param_file}' not found.")

    with open(param_file, "r") as f:
        config = json.load(f)

    param_dict = config["params"]
    joint_states = config.get("joint_states", {})

    ordered_keys = [
        "Correct Posture Bonus",
        "Smooth Bonus Weight",
        "Incorrect Posture Penalty",
        "Jerking Movement Penalty (x10)",
        "High Joint Velocity Penalty (x10)",
        "Distance to Goal Penalty"
    ]

    weights = [param_dict[k] for k in ordered_keys]

    # --- Setup environment and agents ---
    env = Environment()
    env.add_bittles(n=1)

    envs = []
    agents = []

    print("[ENV] num of Bittles:", len(env.bittlles), flush=True)

    for idx, bittle in enumerate(env.bittlles):
        
        
        agent_env = gym_env(
            bittle=bittle,
            env=env,
            weights=weights,
            joint_lock_dict=joint_states
        )
        envs.append(agent_env)

        trainer = stb3_PPO(
            gym_env=agent_env
        )
        agents.append(trainer)

    # --- Launch training threads ---
    wait_for_stage_ready()
    
    
    for agent in agents:
        thread = threading.Thread(target=agent.start_training)
        thread.start()

    # --- Run simulation coordinator (blocks main thread) ---
    coordinator = SimulationCoordinator(envs=envs, world=env.get_world())
    threading.Thread(target=coordinator.run_forever, daemon=True).start()


