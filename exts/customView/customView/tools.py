import os
import time
import pynvml
import torch as th
import omni.kit.app

from isaacsim.core.utils.prims import is_prim_path_valid


def log(*args, flush=True, **kwargs):
    """Safe logging utility used across agent and trainer scripts."""
    print(*args, **kwargs, flush=flush)


def ensure_dir_exists(path):
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def get_free_gpu():
    """Return the ID of the GPU with most free memory. Defaults to CPU if no GPU is available."""
    if not th.cuda.is_available():
        return "cpu"

    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()

    free_gpus = []
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used_ratio = mem_info.used / mem_info.total
        free_gpus.append((i, used_ratio))

    pynvml.nvmlShutdown()
    best_gpu = sorted(free_gpus, key=lambda x: x[1])[0][0]
    return f"cuda:{best_gpu}"


def wait_for_stage_ready(timeout=10.0):
    """
    Waits for the USD stage to finish loading and for the timeline to initialize.
    Safe to call before simulation or prim creation.
    """
    from isaacsim.core.utils.stage import is_stage_loading
    app = omni.kit.app.get_app()
    timeline = omni.timeline.get_timeline_interface()

    t0 = time.time()
    while is_stage_loading() or not timeline:
        if time.time() - t0 > timeout:
            raise RuntimeError("Timeout waiting for stage to be ready")
        log("[TOOLS] Waiting for stage to load...", flush=True)
        app.update()
        time.sleep(0.1)
    log("[TOOLS] Stage ready.", flush=True)


def wait_for_prim(path, timeout=5.0):
    """
    Waits for a USD prim to appear at the specified path.
    """
    t0 = time.time()
    while not is_prim_path_valid(path):
        if time.time() - t0 > timeout:
            raise RuntimeError(f"Timed out waiting for prim: {path}")
        time.sleep(0.05)
