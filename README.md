# 🦾 Isaac Sim Bittle RL Training Extension

This project provides a custom GUI extension for NVIDIA Isaac Sim that enables real-time tuning and training of a reinforcement learning (RL) policy to control the Bittle quadruped robot using PPO (Proximal Policy Optimization) via Stable Baselines3. The extension includes sliders for reward parameter configuration and a "Start Training" button to launch training directly from within the Isaac Sim interface.

![Isaac SIM setup](images/IsaacSim.png)  
![Extension GUI](images/extension.png)

---

## 📦 Features

- ✅ Isaac Sim GUI extension with integrated sliders for RL reward tuning  
- ✅ PPO training pipeline using Stable Baselines3  
- ✅ Custom Gym environment (`gym_env`) with live feedback and IMU-based observation  
- ✅ Real-time reward shaping with posture, jerk, velocity, and goal distance penalties  
- ✅ Uses `ArticulationView` and IMU sensors for robot state control and observation  
- ✅ Generates TensorBoard logs for training visualization (`ppo_logs/`)

---

## 🛠️ Get Started

Follow these steps to set up your environment and begin training:

1. **Download Isaac Sim 4.5.0**  
   Use the standalone version from NVIDIA's developer site.

2. **Clone the Repository**  
   ```bash
   git clone https://github.com/your-username/isaac-bittle-rl-extension.git
   cd isaac-bittle-rl-extension/alpha
   ```

3. **Install Required Packages**  
   Use Isaac Sim's Python environment:
   ```bash
   ./python.sh -m pip install stable-baselines3[extra] gymnasium scipy numpy
   ```

4. **Enable the Extension in Isaac Sim**  
   - Launch Isaac Sim
   - Go to `Window > Extensions`
   - Search for **RL Viewport** under the "3rd Party" tab
   - Manually enable it

5. **Run with Custom Extension Folder**  
   If you want to launch Isaac Sim with the custom extension directly:
   ```bash
   ./isaac-sim.sh --ext-folder alpha/exts
   ```
   Replace `alpha` with the folder name where the repo was cloned.

---

## 📁 Path Configuration

You must update two hardcoded paths in the following files before using the extension:

1. **`PPO.py`**  
   Modify the `sb3_path` to point to the `site-packages` directory in your Isaac Sim install:
   ```python
   sb3_path = "/your/local/path/to/isaac-sim/kit/python/lib/python3.10/site-packages"
   ```

2. **`Bittle.py` – `spawn_bittle()`**  
   Update the path to the Bittle `.usd` robot file:
   ```python
   usd_path = "/your/local/path/to/isaac-bittle-rl-extension/alpha/Bittle_URDF/bittle/bittle.usd"
   ```
   This is critical for spawning the robot into the simulation.

---

## 🚀 How to Use

1. **Launch Isaac Sim**  
   Start Isaac Sim and manually enable the RL extension as described above.

2. **Tune Parameters**  
   Use the GUI sliders to:
   - Increase reward for upright posture  
   - Penalize jerky motions or high joint velocity  
   - Adjust penalty based on distance to goal  

3. **Start Training**  
   Click the **"Start Training"** button to begin PPO training.  
   Trained model will be saved as:
   ```
   ./ppo_bittle.zip
   ```

4. **Visualize with TensorBoard**  
   ```bash
   tensorboard --logdir=ppo_logs/
   ```

---

## 🧪 Testing the Training Setup

You can also test training logic outside the GUI by running:

```bash
./python.sh -m alpha.exts.customView.customView.test
```

Make sure to:
- Launch Isaac Sim manually with GUI
- Enable the extension
- Ensure all paths are correct

Clicking the **Start Training** button in the extension panel should reproduce the training behavior (or surface any errors).

---

## ⚙️ Requirements

- Isaac Sim 4.5.0 (Standalone version)  
- Python 3.10+  
- `stable-baselines3`, `gymnasium`, `scipy`, `numpy`  

To install dependencies:
```bash
./python.sh -m pip install stable-baselines3[extra] gymnasium scipy numpy
```

---

## 🧠 Reward Function Breakdown

```python
reward = (
    + upright_bonus * weight[0]
    + smooth_bonus * weight[1]
    - posture_penalty * weight[2]
    - jerk_penalty * weight[3]
    - velocity_penalty * weight[4]
    - distance_to_goal * weight[5]
)
```

Weights are set via the GUI before training begins.

---

## 📌 Notes

- Do not run `PPO.py` standalone — it expects an active Isaac Sim simulation.
- IMU data is used to infer robot pose (roll, pitch, yaw).
- All training must be launched either from the GUI or using `test.py`.

---

## 📜 License

MIT License – see `LICENSE` file for details.

---

## 🙏 Acknowledgments

Built using:  
- [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim)  
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)  
- [Mini Pupper / Bittle by Petoi](https://www.petoi.com)
