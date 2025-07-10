# test.py

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import os
import json
import time

from world import Environment
from PPO import PPOAgent

MAX_STEPS = 1_000_000

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

    all_weights = config["params"]
    all_joint_states = config["joint_states"]
    num_agents = config.get("num_agents", len(all_weights))

    # === Environment and Agents ===
    sim_env = Environment()
    sim_env.add_bittles(n=num_agents)

    agents = []

    for i, bittle in enumerate(sim_env.bittlles):
        weights = all_weights[i]
        joint_states = all_joint_states[i] if i < len(all_joint_states) else {}

        agent = PPOAgent(weights=weights, bittle=bittle, sim_env=sim_env, joint_states=joint_states)
        obs, _ = agent.gym_env.reset()
        # agent.set_obs(obs)
        agents.append(agent)

    wait_for_stage_ready()

    # === Training Loop ===
    step_count = 0

    while step_count < MAX_STEPS:
        actions = [agent.predict_action(agent.obs) for agent in agents]

        for agent, action in zip(agents, actions):
            agent.gym_env.step(action)

        sim_env.get_world().step(render=True)

        for agent, action in zip(agents, actions):
            agent.post_step(action)
            agent.train()

        step_count += 1

    print("Training complete.")
    for i, agent in enumerate(agents):
        agent.save(f"ppo_bittle_agent_{i}")
