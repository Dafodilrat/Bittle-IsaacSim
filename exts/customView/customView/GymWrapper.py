import gymnasium
from gymnasium import spaces
import numpy as np

import gymnasium
from gymnasium import spaces
import numpy as np

from world import Environment

class gym_env(gymnasium.Env):
    
    def __init__(self, bittle, env, weights=[100, 10, 10, 0.5, 0.2, 10], joint_lock_dict=None):
        super().__init__()

        self.weights = weights
        self.bittle = bittle
        self.environment = env 
        print("[GymEnv] env type", type(env), flush=True)

        self.joint_lock_dict = joint_lock_dict or {}
        internal_joint_names = self.bittle.get_joint_names()
        self.joint_lock_mask = np.array([
            self.joint_lock_dict.get(name, False) for name in internal_joint_names
        ], dtype=bool)

        dof, low, high = self.bittle.get_robot_dof()
        self.prev_action = np.zeros(dof, dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, shape=(dof,), dtype=np.float32)

        obs_low = np.concatenate([
            np.array([-np.inf, -np.inf, -np.inf]),
            -np.ones(3) * np.pi,
            low,
            -np.ones(dof) * 10.0,
        ])
        obs_high = np.concatenate([
            np.array([np.inf, np.inf, np.inf]),
            np.ones(3) * np.pi,
            high,
            np.ones(dof) * 10.0,
        ])
        self.observation_space = spaces.Box(low=obs_low - 0.01, high=obs_high + 0.01, dtype=np.float64)
 
        self.goals = self.environment.get_valid_positions_on_terrain()
        self.current_goal = self.goals[np.random.choice(len(self.goals))]

        self.total_rewards = 0
        self.observations = [[0, 0, 0], [0, 0, 0], [0] * dof, [0] * dof]
        self.prev_distance = 0
        self.delta = 0

        self._step_called = False
        self._pending_action = None
        self._last_obs = np.zeros_like(obs_low)
        self._last_reward = 0.0
        self._last_done = False
        self._last_info = {}

    def step(self, action):

        print("[STEP] ppo called step", flush=True)
        print("[STEP] received action:", action, flush=True)

        self._pending_action = action
        self._step_called = True
        return self._last_obs, self._last_reward, self._last_done, False, self._last_info

    def apply_pending_action(self):
        if self._pending_action is None:
            print("[GymEnv] No action to apply", flush=True)
            return

        try:
            print("[GymEnv] Applying action:", self._pending_action, flush=True)
            action = np.where(self.joint_lock_mask, 0.0, self._pending_action)
            self.bittle.set_robot_action(action)
        except Exception as e:
            print("[GymEnv] Failed to apply action:", e, flush=True)


    def post_step(self):
        self.observations = self.bittle.get_robot_observation()
        reward = self.calculate_reward(self._pending_action)
        done = self.is_terminated()
        info = self.generate_info()

        self._last_obs = np.concatenate(self.observations)
        self._last_reward = reward
        self._last_done = done
        self._last_info = info

        self._step_called = False
        self._pending_action = None

    def generate_info(self):
        pos, orientation, joint_angles, joint_velocities = self.observations
        return {
            "goal": self.current_goal,
            "pose": pos,
            "orientation": orientation,
            "joint_angles": joint_angles,
            "joint_vel": joint_velocities,
            "new": False,
            "total_reward": self.total_rewards,
            "distance_to_goal": np.linalg.norm(np.array(pos[:2]) - np.array(self.current_goal[:2])),
            "delta movement": self.delta
        }

    def is_terminated(self):
        pos, *_ = self.observations
        return np.linalg.norm(np.array(pos[:2]) - np.array(self.current_goal[:2])) < 0.1

    def is_truncated(self):
        return not self.env.is_running()

    def reset(self, *, seed=None, options=None):
   
        print("[GymEnv] reset called", flush=True)
        super().reset(seed=seed)

        if len(self.goals) > 1:
            self.goals.remove(self.current_goal)
        else:
            self.goals = self.environment.get_valid_positions_on_terrain()

        self.current_goal = self.goals[np.random.choice(len(self.goals))]
        print("[GymEnv] calling bittle.reset_simulation()", flush=True)
        self.bittle.reset_simulation()
        print("[GymEnv] finished reset_simulation()", flush=True)

        dof, _, _ = self.bittle.get_robot_dof()
        self.prev_action = np.zeros(dof, dtype=np.float32)
        self.total_rewards = 0
        self.prev_distance = 0
        self.delta = 0

        self.observations = self.bittle.get_robot_observation()
        self._last_obs = np.concatenate(self.observations)

        info = self.generate_info()
        info["new"] = True

        print("------------------RESET---------------------")
        return self._last_obs, info

    def calculate_reward(self, action):
        pos, orientation, joint_angles, joint_velocities = self.observations
        params = self.weights

        z = pos[2]
        roll, pitch, _ = orientation
        delta = np.abs(action - self.prev_action)

        roll_penalty = max(0.0, abs(roll) - 1.2)
        pitch_penalty = max(0.0, abs(pitch) - 1.2)
        posture_penalty = roll_penalty ** 2 + pitch_penalty ** 2
        jerk_penalty = np.sum(np.clip(delta - 0.1, 0, None))
        velocity_penalty = np.sum(np.clip(np.abs(joint_velocities) - 90.0, 0, None))
        z_penalty = 0.4 - z if z < 0.4 else 0.0

        upright_bonus = 1.0 if abs(roll) < 1 and abs(pitch) < 1 else 0.0
        smooth_bonus = 1.0 if np.all(delta < 0.5) else 0.0

        dist_to_goal = np.linalg.norm(self.current_goal[:2] - pos[:2])
        self.delta = abs(self.prev_distance - dist_to_goal)

        reward = 0.0
        reward += params[0] * upright_bonus
        reward += params[1] * smooth_bonus
        reward -= params[2] * posture_penalty
        reward -= params[3] * jerk_penalty
        reward -= params[4] * velocity_penalty
        reward -= z_penalty
        reward -= params[5] * dist_to_goal

        if self.delta < 0.05 and self.prev_distance != 0:
            reward -= 200

        self.prev_action = action.copy()
        self.prev_distance = dist_to_goal

        return reward
