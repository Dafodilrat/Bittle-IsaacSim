# test.py

from isaacsim import SimulationApp
from torch.package.package_exporter import ActionHook
simulation_app = SimulationApp({"headless": False})

import os
import json
import time

from world import Environment
from cordinator import SimulationCoordinator
from PPO import PPOAgent  # <-- new import

MAX_STEPS = 1_000_000
ROLLOUT_LENGTH = 2048  # SB3 PPO default

def wait_for_stage_ready(timeout=10.0):
    import omni.kit.app
    from isaacsim.core.utils.stage import is_stage_loading
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

    # === Load parameters ===
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

    # === Environment and Agents ===
    sim_env = Environment()
    num_agents = 2  # 🔁 Adjust for number of Bittles to spawn
    sim_env.add_bittles(n=num_agents)

    agents = []

    for bittle in sim_env.bittlles:

        agent = PPOAgent(weights=weights, bittle=bittle, sim_env=sim_env, joint_states=joint_states)
        agents.append(agent)

    # === Start Coordinator ===
    wait_for_stage_ready()

    # === Training Loop ===
    step_count = 0

    while step_count < MAX_STEPS:

        # --- Predict actions
        actions = [agent.predict_action(agent.obs) for agent in agents]

        # --- Step each environment (applies action immediately)
        for agent, action in zip(agents, actions):
            agent.step(action)
            
        sim_env.get_world().step(render=True)      
        
        # --- Post-step processing and training
        
        for agent, action in zip(agents, actions):
            agent.post_step(action)
            agent.train()

            step_count += 1

    print("Training complete.")
    for i, agent in enumerate(agents):
        agent.save(f"ppo_bittle_agent_{i}")
