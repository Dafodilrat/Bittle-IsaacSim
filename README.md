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

## 🗂️ Project Structure

```
alpha/
├── Bittle.py         # Interface for loading, controlling, and observing the Bittle robot
├── PPO.py            # PPO training class with Stable Baselines3
├── GymWrapper.py     # Custom Gym environment wrapping Isaac Sim control
├── ext.py            # Omniverse extension with PySide2 GUI and training launcher
└── rl_world.usd      # Isaac Sim stage (loaded separately)
```

---

## 🚀 How to Use

1. **Launch Isaac Sim**  
   Ensure you launch Isaac Sim 4.5.0 and enable this extension.

2. **Open the World**
   The extension will load your predefined stage:
   ```
   /home/dafodilrat/Documents/bu/RASTIC/rl_world.usd
   ```

3. **Tune Parameters**
   Use the GUI sliders to:
   - Increase reward for upright posture
   - Penalize jerky motions or high joint velocity
   - Adjust penalty based on distance to goal

4. **Start Training**
   Click the **"Start Training"** button to begin training the PPO agent.
   Trained model is saved as:
   ```
   ./ppo_bittle.zip
   ```

5. **Visualize with TensorBoard**
   ```
   tensorboard --logdir=ppo_logs/
   ```

---

## ⚙️ Requirements

- Isaac Sim 4.5.0 (Independant stand alone version)
- Python 3.10+
- `stable-baselines3`
- `gymnasium`
- `scipy`, `numpy`

To install Stable Baselines3 manually on the Isaac Sim python env:
```bash
./python.sh -m pip install stable-baselines3[extra]
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

- You must launch the training **from within the GUI**. Do not run `PPO.py` standalone—it relies on an already running simulation context.
- IMU data is used to infer robot pose (roll, pitch, yaw).

---

## 📜 License

MIT License – see `LICENSE` file for details.

---

## 🙏 Acknowledgments

Built using:
- [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim)
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [Mini Pupper / Bittle by Petoi](https://www.petoi.com)
