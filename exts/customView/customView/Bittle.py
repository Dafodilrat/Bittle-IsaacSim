from isaacsim.core.utils.prims import is_prim_path_valid, get_prim_at_path
from isaacsim.core.prims import Articulation
from isaacsim.sensors.physics import _sensor
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.sensors.physics import IMUSensor
from pxr import UsdPhysics, PhysxSchema, UsdGeom, Sdf, Gf, UsdShade
from omni.kit.commands import execute
from omni.isaac.core.simulation_context import SimulationContext

import os
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import random

from tools import log, wait_for_prim, wait_for_physics

class Bittle:
    """
    Represents a single Bittle robot in Isaac Sim.
    Handles spawning, resetting, control, coloring, and IMU-based pose estimation.
    """

    def __init__(self, cords, id, world, flush=False):
        self.flush = flush
        self.robot_prim = "/World/bittle" + str(id)
        self.world = world
        self.spawn_cords = cords
        self.robot_view = None
        self.color = tuple(random.uniform(0.4, 1.0) for _ in range(3))  # RGB in [0.4–1.0]

    def log(self, *args, **kwargs):
        if self.flush:
            print(*args, **kwargs)

    def reset(self):
        """Fully reset and reinitialize the robot."""
        self.log("[Bittle] reset called")
        self.respawn_bittle()
        wait_for_prim(self.robot_prim)
        self.enforce_vel_limits()
        self.log("Simulation started")

    def set_robot_action(self, action):
        """Apply joint action to the robot."""
        self.robot_view.set_joint_positions(action)

    def get_robot_dof(self):
        """Return DOF count, lower and upper limits."""
        num_dofs = self.robot_view.num_dof
        joint_names = self.robot_view.dof_names
        limits = self.robot_view.get_dof_limits()[0]
        return num_dofs, limits[:, 0], limits[:, 1]

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

    def reset_simulation(self):
        """Alias for reset."""
        self.reset()

    def enforce_vel_limits(self):
        """Clamp all joint velocity limits to 90 deg/sec."""
        joint_names = self.robot_view.dof_names
        for name in joint_names:
            joint_path = f"{self.robot_prim}/joints/{name}"
            joint_prim = get_prim_at_path(joint_path)
            if not joint_prim.IsValid():
                self.log(f"[Warning] Joint not found: {joint_path}")
                continue
            attr = joint_prim.GetAttribute("physics:joint:velocityLimit")
            if not attr.IsValid():
                attr = joint_prim.CreateAttribute("physics:joint:velocityLimit", Sdf.ValueTypeNames.Float)
            attr.Set(90.0)
            self.log(f"Enforced velocity limit on {name}")

    def set_articulation(self):
        """Attach articulation API and initialize robot."""
        self.robot_view = Articulation(self.robot_prim)
        self.robot_view.initialize()

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

        prim.SetInstanceable(False)

        # Remove material bindings and apply color
        self.unbind_materials()
        self.apply_display_color()

        # Initial spawn pose
        x, y, z = self.spawn_cords
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(x, y, 1))
        wait_for_physics()

    def apply_display_color(self):
        """Recursively assign color to all meshes."""
        def recursive_color(prim):
            if prim.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(prim)
                mesh.CreateDisplayColorAttr().Set([Gf.Vec3f(*self.color)])
            for child in prim.GetChildren():
                recursive_color(child)

        prim = get_prim_at_path(self.robot_prim)
        recursive_color(prim)
        self.log(f"[COLOR] Applied color {self.color} to {self.robot_prim}")

    def unbind_materials(self):
        """Remove all material bindings recursively."""
        def recursive_unbind(prim):
            binding = UsdShade.MaterialBindingAPI(prim)
            binding.UnbindDirectBinding()
            for child in prim.GetChildren():
                recursive_unbind(child)

        prim = get_prim_at_path(self.robot_prim)
        recursive_unbind(prim)
        self.log(f"[COLOR] Unbound materials for {self.robot_prim}")

    def respawn_bittle(self):
        """Reset robot joint positions and teleport to spawn."""
        self.log("[Bittle] respawn_bittle() entered")
        n, _, _ = self.get_robot_dof()
        self.robot_view.set_joint_positions(np.zeros(n))
        self.robot_view.set_joint_velocities(np.zeros(n))
        self.robot_view.set_world_poses(
            positions=[self.spawn_cords],
            orientations=[[1, 0, 0, 0]]
        )
        self.log("[Bittle] respawn_bittle() completed")

    def remove_prim_at_path(self, prim_path):
        stage = get_current_stage()
        if stage.GetPrimAtPath(prim_path).IsValid():
            self.log(f"Removing prim at: {prim_path}")
            stage.RemovePrim(prim_path)
        else:
            self.log(f"No valid prim at: {prim_path}")

    def get_joint_names(self):
        """Return DOF joint name list."""
        return self.robot_view.dof_names

    def print_info(self):
        """Print current pose and IMU-derived orientation."""
        self.log("[INFO] Fetching robot pose and IMU orientation...")

        try:
            pos, ori = self.get_curr_robot_pose()
            self.log(f"[INFO] Position: x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}",flush=True)
            self.log(f"[INFO] Orientation (rpy): roll={ori[0]:.3f}, pitch={ori[1]:.3f}, yaw={ori[2]:.3f}",flush=True)
        except Exception as e:
            self.log(f"[ERROR] Failed to retrieve pose or orientation: {e}",flush=True)

if __name__ == "__main__":
    print("[Bittle] This module is meant to be used as part of the simulation pipeline.")