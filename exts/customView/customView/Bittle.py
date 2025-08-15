from isaacsim.core.utils.prims import is_prim_path_valid, get_prim_at_path
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.sensors.physics import _sensor
from isaacsim.sensors.physics import IMUSensor  # kept if you use it elsewhere
from isaacsim.core.api.controllers import ArticulationController
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.api.materials import PhysicsMaterial

from pxr import UsdPhysics, PhysxSchema, UsdGeom, Sdf, Gf, UsdShade

import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import random

from tools import log, wait_for_prim, wait_for_physics


class Bittle:
    """
    Represents a single Bittle robot in Isaac Sim.
    Adds hobby-servo-like behavior (MG90-ish) via rate-limited, torque-capped position control.
    """

    def __init__(self, cords, id, world, flush=False):
        self.flush = flush
        self.robot_prim = "/World/bittle" + str(id)
        self.world = world
        self.spawn_cords = cords
        self.robot_view = None
        self.ctrl = None  # ArticulationController instance
        self.color = tuple(random.uniform(0.4, 1.0) for _ in range(3))  # RGB in [0.4–1.0]

        # --- Servo emulation state ---
        self._servo_cfg = None
        self._last_cmd = None
        self._cmd_filt = None

    # ---------------------------
    # Utilities / lifecycle
    # ---------------------------

    def log(self, *args, **kwargs):
        if self.flush:
            print(*args, **kwargs)

    def reset(self):
        """Fully reset and reinitialize the robot."""
        self.log("[Bittle] reset called")
        self.respawn_bittle()
        wait_for_prim(self.robot_prim)
        self.log("Simulation started")

    def reset_simulation(self):
        """Alias for reset."""
        self.reset()

    def remove_prim_at_path(self, prim_path):
        stage = get_current_stage()
        if stage.GetPrimAtPath(prim_path).IsValid():
            self.log(f"Removing prim at: {prim_path}")
            stage.RemovePrim(prim_path)
        else:
            self.log(f"No valid prim at: {prim_path}")

    # ---------------------------
    # Spawning / articulation
    # ---------------------------

    def spawn_bittle(self):
        """Create a fresh Bittle robot from USD, apply pose and appearance."""
        usd_path = os.environ.get("ISAACSIM_PATH") + "/alpha/Bittle_URDF/bittle/bittle.usd"

        if is_prim_path_valid(self.robot_prim):
            self.remove_prim_at_path(self.robot_prim)

        self.log(f"[Bittle] Referencing robot from {usd_path}")
        add_reference_to_stage(usd_path=usd_path, prim_path=self.robot_prim)
        wait_for_prim(self.robot_prim)

        prim = get_prim_at_path(self.robot_prim)
        if not prim.HasAttribute("articulation:root"):
            attr = prim.CreateAttribute("articulation:root", Sdf.ValueTypeNames.Bool)
            attr.Set(True)
            self.log("[Bittle] Marked as articulation root")
        else:
            self.log("[Bittle] Already has articulation:root =", prim.GetAttribute("articulation:root").Get())

        # Initial spawn pose (drop from z=5 to settle)
        x, y, z = self.spawn_cords
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(x, y, 1.0))
        wait_for_physics()

    def set_articulation(self):
        """Create Articulation + controller and configure MG90-like servo profile."""
        self.robot_view = Articulation(self.robot_prim)
        self.robot_view.initialize()

        self.ctrl = ArticulationController()
        self.ctrl.initialize(self.robot_view)
        self.ctrl.switch_control_mode("position")

        # Base gains before servo config (will be overwritten by configure_mg90_profile)
        n = self.robot_view.num_dof
        self.ctrl.set_gains(
            kps=np.full((n,), 12.0, np.float32),
            kds=np.full((n,), 1.2,  np.float32),
            save_to_usd=False
        )

        # Apply MG90-like behavior (defaults can be tweaked inside the function)
        self.configure_mg90_profile()

    # ---------------------------
    # Servo profile & joint setup
    # ---------------------------

    def _get_joint_prim(self, joint_name):
        """Find a joint prim by common URDF-imported path convention."""
        # Typical URDF import puts joints under /joints/<name>. Adjust if yours differs.
        joint_path = f"{self.robot_prim}/joints/{joint_name}"
        prim = get_prim_at_path(joint_path)
        return prim if prim.IsValid() else None

    def configure_mg90_profile(
        self,
        speed_deg_s=600.0,      # ≈0.10 s per 60°
        stall_torque_nm=0.25,   # MG90S-ish
        kp=12.0,
        kd=1.2,
        deadband_deg=1.0,
        cmd_alpha=0.1           # LPF on commands (0=no filter, 1=overwrite)
    ):
        """Configure USD drives, velocity limits, and controller gains to emulate a hobby servo."""
        self._servo_cfg = dict(
            speed_rad_s=np.deg2rad(speed_deg_s),
            torque_limit=stall_torque_nm,
            kp=kp, kd=kd,
            deadband_rad=np.deg2rad(deadband_deg),
            cmd_alpha=cmd_alpha,
        )

        # Controller gains (used by ArticulationController)
        n = self.robot_view.num_dof
        kps = np.full((n,), kp, dtype=np.float32)
        kds = np.full((n,), kd, dtype=np.float32)
        self.ctrl.set_gains(kps=kps, kds=kds, save_to_usd=False)

        # PhysX velocity limit + drive limits per joint
        dof_names = list(self.robot_view.dof_names)
        for name in dof_names:
            prim = self._get_joint_prim(name)
            if not prim:
                continue

            # Try to set a velocity limit attribute (rad/s)
            vel_attr = prim.GetAttribute("physics:joint:velocityLimit")
            if not vel_attr.IsValid():
                vel_attr = prim.CreateAttribute("physics:joint:velocityLimit", Sdf.ValueTypeNames.Float)
            try:
                vel_attr.Set(float(self._servo_cfg["speed_rad_s"]))
            except Exception:
                pass

            # Drive API on revolute joints: channel "angular"
            try:
                drive_api = UsdPhysics.DriveAPI.Apply(prim, "angular")
                drive_api.CreateStiffnessAttr(kp)
                drive_api.CreateDampingAttr(kd)
                drive_api.CreateMaxForceAttr(stall_torque_nm)  # N·m cap
            except Exception:
                # Some joints or schemas may not support DriveAPI—skip safely
                pass

        # Initialize command memories
        if self._last_cmd is None:
            self._last_cmd = np.zeros(self.robot_view.num_dof, dtype=np.float32)
        if self._cmd_filt is None:
            self._cmd_filt = self._last_cmd.copy()

    # ---------------------------
    # Control / observations
    # ---------------------------

    def set_robot_action(self, target_positions):
        """
        Emulate analog hobby servo motion with:
          - command low-pass filtering,
          - slew-rate limiting (speed cap),
          - small deadband near goal,
        then send as position targets. Torque/force is capped via DriveAPI.
        """
        if self._servo_cfg is None:
            # Fallback to direct control if not configured (should not happen if set_articulation() used)
            self.ctrl.apply_action(ArticulationAction(joint_positions=target_positions))
            return

        # Physics dt
        try:
            dt = float(self.world.get_physics_dt())
        except Exception:
            dt = 1.0 / 60.0

        cfg = self._servo_cfg

        # Low-pass the incoming command to remove chatter
        cmd = np.asarray(target_positions, dtype=np.float32)
        self._cmd_filt = (1.0 - cfg["cmd_alpha"]) * self._cmd_filt + cfg["cmd_alpha"] * cmd

        # Slew-rate limit: clamp delta target so the servo can't “teleport”
        max_step = cfg["speed_rad_s"] * dt
        delta = np.clip(self._cmd_filt - self._last_cmd, -max_step, +max_step)
        next_cmd = self._last_cmd + delta

        # Apply small deadband near the filtered target so joints settle calmly
        deadband = cfg["deadband_rad"]
        settle_mask = np.abs(self._cmd_filt - next_cmd) < deadband
        next_cmd = np.where(settle_mask, self._cmd_filt, next_cmd).astype(np.float32)

        # Send to controller (force cap comes from DriveAPI; controller sends pos targets)
        self.ctrl.apply_action(ArticulationAction(joint_positions=next_cmd))

        # Remember
        self._last_cmd = next_cmd

    def get_robot_dof(self):
        """Return DOF count, lower and upper limits."""
        num_dofs = self.robot_view.num_dof
        limits = self.robot_view.get_dof_limits()[0]
        return num_dofs, limits[:, 0], limits[:, 1]

    def get_joint_names(self):
        """Return DOF joint name list."""
        return self.robot_view.dof_names

    def get_curr_robot_pose(self):
        """Estimate robot pose from IMU quaternion and world position."""
        imu = _sensor.acquire_imu_sensor_interface()
        imu_data = imu.get_sensor_reading(self.robot_prim + "/base_frame_link/Imu_Sensor")
        quat = imu_data.orientation

        # Handle zero-norm quaternion safely
        norm = np.linalg.norm(quat)
        if norm < 1e-6:
            print(f"[WARNING] Zero-norm quaternion detected for {self.robot_prim}. Using identity rotation.")
            quat = [0.0, 0.0, 0.0, 1.0]

        r = R.from_quat(quat)
        roll, pitch, yaw = r.as_euler('xyz', degrees=False)

        pos, _ = self.robot_view.get_world_poses()
        return pos[0], [roll, pitch, yaw]

    def get_robot_observation(self):
        """Return observation tuple: position, orientation, joint angles, velocities."""
        pos, ori = self.get_curr_robot_pose()
        angles = self.robot_view.get_joint_positions()[0]
        vel = self.robot_view.get_joint_velocities()[0]
        return [pos, ori, angles, vel]

    def respawn_bittle(self):
        """Reset robot joint positions and teleport to spawn."""
        self.log("[Bittle] respawn_bittle() entered")
        n, _, _ = self.get_robot_dof()
        self.robot_view.set_joint_positions(np.zeros(n))
        self.robot_view.set_joint_velocities(np.zeros(n))
        self.robot_view.set_world_poses(
            positions=[[self.spawn_cords[0], self.spawn_cords[1], 1.0]],
            orientations=[[1.0, 0.0, 0.0, 0.0]]
        )
        self.log("[Bittle] respawn_bittle() completed")

    def print_info(self):
        """Print current pose and IMU-derived orientation."""
        self.log("[INFO] Fetching robot pose and IMU orientation...")
        try:
            pos, ori = self.get_curr_robot_pose()
            self.log(f"[INFO] Position: x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}", flush=True)
            self.log(f"[INFO] Orientation (rpy): roll={ori[0]:.3f}, pitch={ori[1]:.3f}, yaw={ori[2]:.3f}", flush=True)
        except Exception as e:
            self.log(f"[ERROR] Failed to retrieve pose or orientation: {e}", flush=True)


if __name__ == "__main__":
    print("[Bittle] This module is meant to be used as part of the simulation pipeline.")
