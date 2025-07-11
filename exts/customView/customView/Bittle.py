from isaacsim.core.utils.prims import is_prim_path_valid, get_prim_at_path
from isaacsim.core.prims import Articulation, SingleArticulation
from isaacsim.sensors.physics import _sensor  
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.sensors.physics import IMUSensor
from pxr import UsdPhysics, PhysxSchema
from omni.kit.commands import execute
from omni.isaac.core.simulation_context import SimulationContext

import os
from pxr import UsdGeom, Sdf, Gf
from scipy.spatial.transform import Rotation as R
import time
import numpy as np
from sympy import true

class Bittle():

    def __init__(self,cords,id,world):
        
        self.robot_prim = "/World/bittle"+str(id)
        self.world = world
        self.spawn_cords = cords
        self.robot_view = None
        # self.spawn_bittle()
        # self.world.reset()
        # ph = PhysicsContext(prim_path = "/World/PhysicsScene")
        # print("[BITTLE] ",ph.get_current_physics_scene_prim(),flush=True)
        # print("[BITTLE] initializing bittle object "+self.robot_prim,flush=True)
        # self.wait_for_physics()

    def reset(self):

        print("[Bittle] reset called", flush=True)
        self.respawn_bittle()
        print("[Bittle] finished respawn_bittle()", flush=True)
        self.wait_for_prim(self.robot_prim)
        print("[Bittle] finished wait_for_prim()", flush=True)
        self.enforce_vel_limits()
        print("[Bittle] finished enforce_vel_limits()", flush=True)
        print("Simulation started", flush=True)

    def wait_for_prim(self, path, timeout=5.0):
        t0 = time.time()
        while not is_prim_path_valid(path):
            if time.time() - t0 > timeout:
                raise RuntimeError(f"Timed out waiting for prim: {path}")
            time.sleep(0.05)

    def set_robot_action(self, action):

        self.robot_view.set_joint_positions(action)

    def get_robot_dof(self):
        num_dofs = self.robot_view.num_dof
        joint_names = self.robot_view.dof_names
        limits = self.robot_view.get_dof_limits()[0]
        return num_dofs, limits[:, 0], limits[:, 1]

    def get_curr_robot_pose(self):

        imu = _sensor.acquire_imu_sensor_interface()
        imu_data = imu.get_sensor_reading(self.robot_prim + "/base_frame_link/Imu_Sensor")
        quat = imu_data.orientation
        r = R.from_quat([quat[0], quat[1], quat[2], quat[3]])
        roll, pitch, yaw = r.as_euler('xyz', degrees=False)
        pos, _ = self.robot_view.get_world_poses()
        return pos[0], [roll, pitch, yaw]

    def get_robot_observation(self):

        pos, ori = self.get_curr_robot_pose()
        angles = self.robot_view.get_joint_positions()[0]
        vel = self.robot_view.get_joint_velocities()[0]
        return [pos, ori, angles, vel]

    def reset_simulation(self):
        self.reset()

    def wait_for_physics(self, timeout=5.0):
        """
        Wait for SimulationContext's physics context and sim view to be ready.

        Args:
            timeout (float): Maximum time to wait in seconds.

        Raises:
            RuntimeError: If physics sim view or context are not initialized in time.
        """
        sim = SimulationContext(physics_prim_path="/World/PhysicsScene")
        t0 = time.time()
        while sim.physics_sim_view is None or sim._physics_context is None:
            sim.initialize_physics()
            if time.time() - t0 > timeout:
                raise RuntimeError("Timeout waiting for physics sim view and context to initialize.")
            print("[Bittle] Waiting for physics...", flush=True)
            time.sleep(0.1)
        return

    def enforce_vel_limits(self):

        joint_names = self.robot_view.dof_names
        
        for name in joint_names:
            joint_path = f"{self.robot_prim}/joints/{name}"
            joint_prim = get_prim_at_path(joint_path)
            if not joint_prim.IsValid():
                print(f"[Warning] Joint not found: {joint_path}")
                continue
            attr = joint_prim.GetAttribute("physics:joint:velocityLimit")
            if not attr.IsValid():
                attr = joint_prim.CreateAttribute("physics:joint:velocityLimit", Sdf.ValueTypeNames.Float)
            attr.Set(90.0)
            print(f"Enforced velocity limit on {name}")

    def set_articulation(self):

        self.robot_view = Articulation(self.robot_prim)
        self.robot_view.initialize()

    def spawn_bittle(self):

        usd_path =  os.environ.get("ISAACSIM_PATH") + "/alpha/Bittle_URDF/bittle/bittle.usd"
        
        imu_path = self.robot_prim + "/base_frame_link/Imu_Sensor"

        # Add robot to the stage if it's not already present
        if is_prim_path_valid(self.robot_prim):

            self.remove_prim_at_path(self.robot_prim)
        
        print(f"[Bittle] Referencing robot from {usd_path}")
        add_reference_to_stage(usd_path=usd_path, prim_path=self.robot_prim)
        
        self.wait_for_prim(self.robot_prim)
        
        prim = get_prim_at_path(self.robot_prim)

        if not prim.HasAttribute("articulation:root"):
            attr = prim.CreateAttribute("articulation:root", Sdf.ValueTypeNames.Bool)
            attr.Set(True)
            print("[Bittle] Marked as articulation root")
        else:
            print("[Bittle] Already has articulation:root =", prim.GetAttribute("articulation:root").Get())

        x, y, z = self.spawn_cords
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(x, y, 1))
    
    def respawn_bittle(self):
        print("[Bittle] respawn_bittle() entered", flush=True)
        n, _, _ = self.get_robot_dof()
        self.robot_view.set_joint_positions(np.zeros(n))
        self.robot_view.set_joint_velocities(np.zeros(n))
        self.robot_view.set_world_poses(
            positions=[self.spawn_cords],
            orientations=[[1, 0, 0, 0]]
        )
        print("[Bittle] respawn_bittle() completed", flush=True)


    def remove_prim_at_path(self,prim_path):
        
        stage = get_current_stage()
        
        if stage.GetPrimAtPath(prim_path).IsValid():
            print(f"Removing prim at: {prim_path}")
            stage.RemovePrim(prim_path)
        else:
            print(f"No valid prim at: {prim_path}")
    
    def get_joint_names(self):
        """
        Return the list of joint names in the order used by get_robot_dof() and set_robot_action().
        """
        return self.robot_view.dof_names




if __name__ == "__main__":

    b = Bittle()
    