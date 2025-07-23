import gymnasium
from gymnasium import spaces
import numpy as np
from pxr import UsdGeom, Gf
from isaacsim.core.utils.stage import get_current_stage, is_stage_loading
from isaacsim.core.utils.prims import is_prim_path_valid, get_prim_at_path


class gym_env(gymnasium.Env):

    def __init__(self, bittle, env, grnd, weights=[100, 10, 10, 0.5, 0.2, 10], joint_lock_dict=None):
        super().__init__()

        self.weights = weights
        self.bittle = bittle
        self.grnd = grnd
        self.environment = env

        self.joint_lock_dict = joint_lock_dict or {}
        joint_names = self.bittle.get_joint_names()
        self.joint_lock_mask = np.array([
            self.joint_lock_dict.get(name, False) for name in joint_names
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

        self.prev_distance = 0
        self.total_rewards = 0
        self.delta = 0

        self._last_obs = None
        self._last_reward = 0.0
        self._last_done = False
        self._last_info = {}
        self.goal_marker_path = f"/World/GoalMarker_{self.bittle.robot_prim.split('/')[-1]}"
        self.create_or_update_goal_marker(self.grnd.get_point())


    def step(self, action):
    
        action = np.where(self.joint_lock_mask, 0.0, action)
        self.bittle.set_robot_action(action)

        return self.get_previous_observation(), self._last_reward, self._last_done, False, self._last_info

    def post_step(self):
        self.observations = self.bittle.get_robot_observation()
        reward = self.calculate_reward(self.prev_action)
        done = self.is_terminated()
        info = self.generate_info()

        self._last_obs = np.concatenate(self.observations)
        self._last_reward = reward
        self._last_done = done
        self._last_info = info

        return self._last_obs, reward, done, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.current_goal = self.grnd.get_point()

        self.bittle.reset_simulation()

        self.prev_action = np.zeros_like(self.prev_action)
        self.prev_distance = 0
        self.total_rewards = 0
        self.delta = 0

        self.observations = self.bittle.get_robot_observation()
        self._last_obs = np.concatenate(self.observations)
        self._last_reward = 0.0
        self._last_done = False
        self._last_info = self.generate_info()
        self._last_info["new"] = True

        self.create_or_update_goal_marker(self.current_goal)

        return self._last_obs, self._last_info

    def is_terminated(self):
        pos, *_ = self.observations

        # Check goal reached
        goal_reached = np.linalg.norm(np.array(pos[:2]) - np.array(self.current_goal[:2])) < 2

        # Check collision with other Bittles
        collided_paths = self.environment.get_collided_bittle_prim_paths()
        self_collided = self.bittle.robot_prim in collided_paths
        fall = pos[2] < 0  # Check if robot is below ground level

        if goal_reached:
            print(f"[TERMINATED] Goal reached by {self.bittle.robot_prim}", flush=True)
        if self_collided:
            print(f"[TERMINATED] Collision detected for {self.bittle.robot_prim}", flush=True)
        if pos[2] < 0:
            print(f"[TERMINATED] {self.bittle.robot_prim} fell below ground level", flush=True)

        return goal_reached or self_collided or fall


    def generate_info(self):
        pos, orientation, joint_angles, joint_velocities = self.observations
        return {
            "goal": self.current_goal,
            "pose": pos,
            "orientation": orientation,
            "joint_angles": joint_angles,
            "joint_vel": joint_velocities,
            "total_reward": self.total_rewards,
            "distance_to_goal": np.linalg.norm(np.array(pos[:2]) - np.array(self.current_goal[:2])),
            "delta movement": self.delta,
        }

    def calculate_reward(self, action):
        pos, orientation, joint_angles, joint_velocities = self.observations
        params = self.weights  # Expected to be a list of 8 weights

        roll, pitch, yaw = orientation
        delta = np.abs(action - self.prev_action)

        # --- Reward components ---
        posture_penalty = (max(0, abs(roll) - 0.2) ** 2 + max(0, abs(pitch) - 0.2) ** 2)
        jerk_penalty = np.linalg.norm(delta)
        velocity_penalty = np.sum(np.tanh(np.abs(joint_velocities) / 100))
        z_penalty = max(0.0, abs(-0.2 - pos[2]))

        # Upright and smooth movement bonuses
        upright_bonus = np.clip(1.5 - (abs(roll) + abs(pitch)), 0, 1.5)
        smooth_bonus = np.exp(-np.linalg.norm(delta))

        # Distance-based shaping
        dist_to_goal = np.linalg.norm(self.current_goal[:2] - pos[:2])
        delta_dist = abs(self.prev_distance - dist_to_goal)
        self.delta = delta_dist

        # Goal heading alignment
        goal_vector = np.array(self.current_goal[:2]) - np.array(pos[:2])
        robot_forward = np.array([np.cos(yaw), np.sin(yaw)])
        goal_alignment_bonus = max(0.0, np.dot(goal_vector, robot_forward) / (np.linalg.norm(goal_vector) + 1e-6))

        # Arrival bonus if upright and at goal
        at_goal = dist_to_goal < 0.1 and abs(roll) < 0.3 and abs(pitch) < 0.3
        goal_arrival_bonus = 20.0 if at_goal else 0.0

        # --- Tipping detection ---
        is_tipped = abs(roll) > 0.8 or abs(pitch) > 0.8
        tipping_penalty = 5.0 if is_tipped else 0.0

        # Recovery bonus if it was tipped in last step and now isn't
        was_tipped = getattr(self, "was_tipped_last", False)
        recovering_bonus = 2.0 if was_tipped and not is_tipped else 0.0
        self.was_tipped_last = is_tipped

        # --- Final reward calculation ---
        reward = 0.0
        reward += params[0] * upright_bonus
        reward += params[1] * smooth_bonus
        reward -= params[2] * posture_penalty
        reward -= params[3] * jerk_penalty
        reward -= params[4] * velocity_penalty
        reward -= params[5] * z_penalty
        reward -= params[6] * dist_to_goal
        reward += params[7] * goal_alignment_bonus
        reward += goal_arrival_bonus
        reward -= tipping_penalty
        reward += recovering_bonus

        # Save state
        self.prev_action = action.copy()
        self.prev_distance = dist_to_goal

        return reward

    def get_previous_observation(self):
        return self._last_obs

    def get_current_observation(self):
        return np.concatenate(self.bittle.get_robot_observation())
    
    def create_or_update_goal_marker(self, position):
        stage = get_current_stage()

        # Elevate Z a bit for visibility
        elevated_pos = (position[0], position[1], position[2] + 0.15)

        if not is_prim_path_valid(self.goal_marker_path):
            # Create the marker if it doesn't exist
            sphere = UsdGeom.Sphere.Define(stage, self.goal_marker_path)
            sphere.CreateRadiusAttr(0.1)

            # Set position
            xform = UsdGeom.Xformable(sphere)
            xform.AddTranslateOp().Set(Gf.Vec3d(*elevated_pos))

            # Set bright green color
            sphere.GetDisplayColorAttr().Set([Gf.Vec3f(*(0.0, 1.0, 0.0))])

        else:
            # Move existing marker to new elevated goal position
            prim = stage.GetPrimAtPath(self.goal_marker_path)
            xform = UsdGeom.Xformable(prim)
            ops = xform.GetOrderedXformOps()
            if ops:
                ops[0].Set(Gf.Vec3d(*elevated_pos))
            else:
                xform.AddTranslateOp().Set(Gf.Vec3d(*elevated_pos))


