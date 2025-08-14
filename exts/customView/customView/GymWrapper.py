import gymnasium
from gymnasium import spaces
import numpy as np
from pxr import UsdGeom, Gf
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.prims import is_prim_path_valid, get_prim_at_path
from tools import wait_for_prim, wait_for_stage_ready

class gym_env(gymnasium.Env):
    """
    Custom Gymnasium environment for Bittle robot training in Isaac Sim.
    Encapsulates robot control, observation, reward shaping, and termination.
    """

    # ===== Fixed reward weights (tune here) =====
    UPRIGHT_W           = 1.0    # reward for being upright
    SMOOTH_W            = 0.5    # reward for smooth actions (small deltas)
    POSTURE_W           = 2.0    # penalty for excessive roll/pitch
    JERK_W              = 0.5    # penalty for action changes
    JOINT_VEL_W         = 0.3    # penalty for high joint velocities
    VERT_MOTION_W       = 1.0    # penalty for vertical velocity/acceleration (anti-hop)
    DIST_W              = 1.5    # penalty for distance-to-goal
    HEADING_W           = 1.0    # reward for heading toward goal

    GOAL_ARRIVAL_BONUS  = 20.0
    TIPPING_PENALTY     = 5.0
    RECOVERY_BONUS      = 2.0
    ALIVE_BONUS         = 0.0    # small per-step bonus if desired (e.g., 0.02)

    # vertical-motion deadbands (per-step units)
    VZ_DEADBAND         = 0.005  # tolerate ~5 mm per-step z motion
    AZ_DEADBAND         = 0.010  # tolerate small per-step z accel
    UPWARD_BIAS         = 1.5    # penalize upward motion slightly more than downward

    def __init__(self, bittle, env, grnd, weights=None, joint_lock_dict=None):
        """
        `weights` is kept for backward compatibility but is ignored by this version.
        """
        super().__init__()

        # === Inputs and Config ===
        self.bittle = bittle
        self.environment = env
        self.grnd = grnd
        self.weights = weights  # not used
        self.joint_lock_dict = joint_lock_dict or {}

        # === Joint masking ===
        joint_names = self.bittle.get_joint_names()
        self.joint_lock_mask = np.array(
            [self.joint_lock_dict.get(name, False) for name in joint_names],
            dtype=bool
        )

        # === Action & Observation spaces ===
        dof, low, high = self.bittle.get_robot_dof()
        self.prev_action = np.zeros(dof, dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, shape=(dof,), dtype=np.float32)

        obs_low = np.concatenate([
            [-np.inf] * 3,                # Position
            -np.ones(3) * np.pi,          # Orientation (roll, pitch, yaw)
            low,                          # Joint angles
            -np.ones(dof) * 10.0,         # Joint velocities
        ])
        obs_high = np.concatenate([
            [np.inf] * 3,
            np.ones(3) * np.pi,
            high,
            np.ones(dof) * 10.0,
        ])
        self.observation_space = spaces.Box(low=obs_low - 0.01, high=obs_high + 0.01, dtype=np.float64)

        # === State ===
        self.prev_distance = 0.0
        self.total_rewards = 0.0
        self.delta = 0.0

        self._last_obs = None
        self._last_reward = 0.0
        self._last_done = False
        self._last_info = {}

        # vertical-motion memory
        self._prev_base_z = 0.0
        self._prev_base_vz = 0.0

        # === Visualization ===
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
        self.prev_distance = 0.0
        self.total_rewards = 0.0
        self.delta = 0.0

        self.observations = self.bittle.get_robot_observation()
        pos, _, _, _ = self.observations
        self._prev_base_z = float(pos[2])
        self._prev_base_vz = 0.0

        self._last_obs = np.concatenate(self.observations)
        self._last_reward = 0.0
        self._last_done = False
        self._last_info = self.generate_info()
        self._last_info["new"] = True

        self.create_or_update_goal_marker(self.current_goal)

        return self._last_obs, self._last_info

    def is_terminated(self):
        pos, *_ = self.observations

        goal_reached = np.linalg.norm(np.array(pos[:2]) - np.array(self.current_goal[:2])) < 0.3
        collided_paths = self.environment.get_collided_bittle_prim_paths()
        self_collided = self.bittle.robot_prim in collided_paths
        fall = pos[2] < 0.0

        if goal_reached:
            print(f"[TERMINATED] Goal reached by {self.bittle.robot_prim}", flush=True)
        if self_collided:
            print(f"[TERMINATED] Collision detected for {self.bittle.robot_prim}", flush=True)
        if fall:
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
        roll, pitch, yaw = orientation
        delta = np.abs(action - self.prev_action)

        # --- Base posture and smoothness ---
        posture_penalty = (max(0.0, abs(roll) - 0.2) ** 2 +
                           max(0.0, abs(pitch) - 0.2) ** 2)
        jerk_penalty = np.linalg.norm(delta)
        velocity_penalty = np.sum(np.tanh(np.abs(joint_velocities) / 100.0))

        upright_bonus = np.clip(1.5 - (abs(roll) + abs(pitch)), 0.0, 1.5)
        smooth_bonus = np.exp(-np.linalg.norm(delta))

        # --- Goal terms ---
        dist_to_goal = np.linalg.norm(self.current_goal[:2] - pos[:2])
        self.delta = abs(self.prev_distance - dist_to_goal)

        goal_vector = np.array(self.current_goal[:2]) - np.array(pos[:2])
        robot_forward = np.array([np.cos(yaw), np.sin(yaw)])
        goal_alignment_bonus = max(0.0, np.dot(goal_vector, robot_forward) /
                                   (np.linalg.norm(goal_vector) + 1e-6))

        at_goal = (dist_to_goal < 0.1) and (abs(roll) < 0.3) and (abs(pitch) < 0.3)
        goal_arrival_bonus = self.GOAL_ARRIVAL_BONUS if at_goal else 0.0

        # --- Tip / recover ---
        is_tipped = (abs(roll) > 0.8) or (abs(pitch) > 0.8)
        tipping_penalty = self.TIPPING_PENALTY if is_tipped else 0.0
        was_tipped = getattr(self, "was_tipped_last", False)
        recovering_bonus = self.RECOVERY_BONUS if was_tipped and not is_tipped else 0.0
        self.was_tipped_last = is_tipped

        # --- NEW: Vertical motion penalty (anti-hop / anti-crawl via bounce suppression) ---
        z = float(pos[2])
        vz = z - self._prev_base_z                 # per-step vertical velocity
        az = vz - self._prev_base_vz               # per-step vertical accel

        vz_term = max(0.0, abs(vz) - self.VZ_DEADBAND) * (self.UPWARD_BIAS if vz > 0 else 1.0)
        az_term = max(0.0, abs(az) - self.AZ_DEADBAND)
        vertical_motion_penalty = 0.5 * vz_term + 0.5 * az_term

        # --- Combine reward ---
        reward = 0.0
        reward += self.UPRIGHT_W      * upright_bonus
        reward += self.SMOOTH_W       * smooth_bonus
        reward -= self.POSTURE_W      * posture_penalty
        reward -= self.JERK_W         * jerk_penalty
        reward -= self.JOINT_VEL_W    * velocity_penalty
        reward -= self.VERT_MOTION_W  * vertical_motion_penalty
        reward -= self.DIST_W         * dist_to_goal
        reward += self.HEADING_W      * goal_alignment_bonus
        reward += goal_arrival_bonus + recovering_bonus + self.ALIVE_BONUS
        reward -= tipping_penalty

        # --- Update histories ---
        self.prev_action = action.copy()
        self.prev_distance = dist_to_goal
        self._prev_base_vz = vz
        self._prev_base_z = z

        return float(np.clip(reward, -50.0, 50.0))

    def get_previous_observation(self):
        return self._last_obs

    def get_current_observation(self):
        return np.concatenate(self.bittle.get_robot_observation())

    def create_or_update_goal_marker(self, position):
        stage = get_current_stage()
        elevated_pos = (position[0], position[1], position[2] + 0.15)

        if not is_prim_path_valid(self.goal_marker_path):
            sphere = UsdGeom.Sphere.Define(stage, self.goal_marker_path)
            sphere.CreateRadiusAttr(0.1)
            xform = UsdGeom.Xformable(sphere)
            xform.AddTranslateOp().Set(Gf.Vec3d(*elevated_pos))
            sphere.GetDisplayColorAttr().Set([Gf.Vec3f(*(0.0, 1.0, 0.0))])
        else:
            prim = stage.GetPrimAtPath(self.goal_marker_path)
            xform = UsdGeom.Xformable(prim)
            ops = xform.GetOrderedXformOps()
            if ops:
                ops[0].Set(Gf.Vec3d(*elevated_pos))
            else:
                xform.AddTranslateOp().Set(Gf.Vec3d(*elevated_pos))
