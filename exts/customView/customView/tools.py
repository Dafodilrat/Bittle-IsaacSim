import os
import time
import torch as th
import omni.kit.app

from omni.isaac.core.simulation_context import SimulationContext
from isaacsim.core.utils.prims import is_prim_path_valid
from pxr import Gf

import torch as th
import pynvml
import glob

def log(msg, flush=False):
    if flush:
        print(msg, flush=True)

def ensure_dir_exists(path):
    """Ensure a directory exists."""
    if not os.path.exists(path):
        os.makedirs(path)

def get_free_gpu():
    """
    Selects the best available GPU by considering both memory and compute usage.
    Returns 'cuda:X' or 'cpu' if no GPU is available.
    """
    if not th.cuda.is_available():
        return "cpu"

    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        best_gpu = None
        best_score = float('-inf')

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)

            mem_free_ratio = mem.free / mem.total
            util_score = 1.0 - util.gpu / 100.0  # 1 = unused, 0 = fully busy

            score = mem_free_ratio * 0.7 + util_score * 0.3  # weight memory more

            if score > best_score:
                best_score = score
                best_gpu = i

        pynvml.nvmlShutdown()
        return f"cuda:{best_gpu}"

    except Exception as e:
        print(f"[get_free_gpu] Error: {e}")
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

def format_joint_locks(joint_lock_dict):
    """
    Converts joint lock dictionary into a filename-safe suffix.
    Example: {'joint1': True, 'joint2': False} → 'joint1_joint2'
    """
    locked = [name for name, locked in joint_lock_dict.items() if locked]
    return "_".join(sorted(locked)) if locked else "all_free"


def get_checkpoint_filename(algo, joint_lock_dict, step=None):
    """
    Generates a checkpoint filename.
    Example:
      - With locked joints: ppo_joint1_joint2_step_1000.pth
      - No joints locked:   ppo_step_1000.pth
    """
    suffix = format_joint_locks(joint_lock_dict)
    if suffix == "all_free":
        return f"{algo}_step_{step}.pth" if step is not None else f"{algo}.pth"
    return f"{algo}_{suffix}_step_{step}.pth" if step is not None else f"{algo}_{suffix}.pth"


def save_checkpoint(model, algo, joint_lock_dict, step_count, save_dir, log_fn=print):
    """
    Saves the model to a uniquely named file.
    """
    os.makedirs(save_dir, exist_ok=True)
    filename = get_checkpoint_filename(algo, joint_lock_dict, step=step_count)
    path = os.path.join(save_dir, filename)
    model.save(path)
    log_fn(f"[{algo.upper()}] Saved model to {path}", flush=True)
    return path


def load_latest_checkpoint(algo, joint_lock_dict, save_dir):
    """
    Finds the latest checkpoint for a given algo and joint lock combo.
    """
    suffix = format_joint_locks(joint_lock_dict)
    pattern = os.path.join(save_dir, f"{algo}_{suffix}_step_*.pth")
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=lambda p: int(p.split("_step_")[-1].split(".")[0]), reverse=True)
    return {
        "path": files[0],
        "step": int(files[0].split("_step_")[-1].split(".")[0])
    }

