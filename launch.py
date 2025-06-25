from isaacsim import SimulationApp
import socket
import time
from PIL import Image

kit_app = SimulationApp({"headless": True})
from isaacsim.core.utils.stage import add_reference_to_stage, open_stage
open_stage("/home/dafodilrat/Documents/bu/RASTIC/rl_world.usd")

import omni.kit
import omni.kit.viewport.utility

viewport = omni.kit.viewport.utility.get_active_viewport()
viewport.set_active_camera("/OmniverseKit_Persp")

print("[Sim] Viewport initialized.")

HOST, PORT = "127.0.0.1", 65435
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((HOST, PORT))
sock.listen(1)

print(f"[Sim] Waiting for viewer client on {HOST}:{PORT}...")
conn, _ = sock.accept()

def capture_viewport():
    img_data = viewport.get_texture()
    if img_data is None:
        return None

    np_img = np.frombuffer(img_data, dtype=np.uint8)
    np_img = np_img.reshape((viewport.resolution[1], viewport.resolution[0], 4))  # RGBA
    img = Image.fromarray(np_img[:, :, :3])  # Drop alpha
    with io.BytesIO() as output:
        img.save(output, format="JPEG")
        return output.getvalue()

while kit_app.is_running():
    kit_app.update()
    jpeg = capture_viewport()
    if jpeg:
        conn.sendall(len(jpeg).to_bytes(4, 'big'))
        conn.sendall(jpeg)
    time.sleep(1 / 15)

import json
from PPO import train
# Load saved params
with open("params.json", "r") as f:
    param_dict = json.load(f)

# Convert to list format if needed
param_list = list(param_dict.values())

# Start training
t = train(param_list)
t.start()
