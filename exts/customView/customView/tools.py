import os
import time
import torch as th
import omni.kit.app

from omni.isaac.core.simulation_context import SimulationContext
from isaacsim.core.utils.prims import is_prim_path_valid
from pxr import Gf

def log(msg, flush=False):
    if flush:
        print(msg, flush=True)

def ensure_dir_exists(path):
    """Ensure a directory exists."""
    if not os.path.exists(path):
        os.makedirs(path)

def get_free_gpu():
    """Return 'cuda' if GPU is available, otherwise 'cpu'."""
    return "cuda" if th.cuda.is_available() else "cpu"

def wait_for_prim(path, timeout=5.0):
    """Wait for a given prim path to become valid on stage."""
    from isaacsim.core.utils.prims import is_prim_path_valid
    start_time = time.time()
    while not is_prim_path_valid(path):
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Timeout waiting for prim at path: {path}")
        time.sleep(0.05)

def wait_for_stage_ready(timeout=10.0):
    """Wait until Isaac Sim stage is loaded and timeline is initialized."""
    from isaacsim.core.utils.stage import is_stage_loading
    app = omni.kit.app.get_app()
    timeline = omni.timeline.get_timeline_interface()

    t0 = time.time()
    while is_stage_loading() or not timeline:
        if time.time() - t0 > timeout:
            raise RuntimeError("Timeout waiting for stage to be ready")
        log("[ENV] Waiting for stage...", flush=True)
        app.update()
        time.sleep(0.1)

def wait_for_physics(timeout=5.0, prim_path="/World/PhysicsScene", flush=False):
    """Wait for physics context to be ready at given prim path."""
    sim = SimulationContext(physics_prim_path=prim_path)
    t0 = time.time()
    while sim.physics_sim_view is None or sim._physics_context is None:
        sim.initialize_physics()
        if time.time() - t0 > timeout:
            raise RuntimeError(f"Timeout waiting for physics context at {prim_path}")
        if flush:
            print(f"[WAIT] Waiting for physics at {prim_path}...", flush=True)
        time.sleep(0.1)
