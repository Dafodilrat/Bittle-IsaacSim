from isaacsim.core.utils.prims import is_prim_path_valid, get_prim_at_path
from isaacsim.core.prims import Articulation
from isaacsim.core.api import World
from isaacsim.sensors.physics import _sensor  
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.sensors.physics import IMUSensor
from pxr import UsdPhysics, PhysxSchema
from omni.kit.commands import execute
from omni.isaac.core.physics_context import PhysicsContext
from omni.isaac.core.simulation_context import SimulationContext

from pxr import UsdGeom, Sdf, Gf
from scipy.spatial.transform import Rotation as R
import time
import numpy as np

class Bittle():

    def __init__(self,cords,id,world):
        
        self.robot_prim = "/World/bittle"+str(id)
        self.world = world
        self.spawn_cords = cords
        self.spawn_bittle()
        self.world.reset()
        # self.wait_for_physics()
        self.robot_view = Articulation(self.robot_prim)
        self.robot_view.initialize()

    def reset(self):

        self.respawn_bittle()
        self.wait_for_prim(self.robot_prim)
        self.enforce_vel_limits()

        print("Simulation started")

    def wait_for_prim(self, path, timeout=5.0):
        t0 = time.time()
        while not is_prim_path_valid(path):
            if time.time() - t0 > timeout:
                raise RuntimeError(f"Timed out waiting for prim: {path}")
            time.sleep(0.05)

    def set_robot_action(self, action):
        # Optional: uncomment if step still crashes
        # if getattr(self.sim, "_physics_context", None) is None:
        #     print("[Warning] Physics not ready")
        #     return
        self.robot_view.set_joint_positions(action)
        self.world.step(render=True)

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

    def is_running(self):
        return self.world.is_playing()

    def reset_simulation(self):
        self.reset()

    def wait_for_physics(self, timeout=10.0):
        """
        Wait for SimulationContext's physics context and sim view to be ready.

        Args:
            timeout (float): Maximum time to wait in seconds.

        Raises:
            RuntimeError: If physics sim view or context are not initialized in time.
        """
        sim = SimulationContext()
        t0 = time.time()
        while sim.physics_sim_view is None or sim._physics_context is None:
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

    def spawn_bittle(self):

        usd_path = "/home/dafodilrat/Documents/bu/RASTIC" \
        "/isaac-sim-standalone@4.5.0-rc.36+release.19112.f59b3005.gl.linux-x86_64.release" \
        "/alpha/Bittle_URDF/bittle/bittle.usd"
        
        imu_path = self.robot_prim + "/base_frame_link/Imu_Sensor"

        # Add robot to the stage if it's not already present
        if not is_prim_path_valid(self.robot_prim):
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

        # UsdPhysics.ArticulationRootAPI.Apply(prim)
        # PhysxSchema.PhysxArticulationAPI.Apply(prim)

        x, y, z = self.spawn_cords
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(x, y, 1))

        # Check if IMU exists
        if is_prim_path_valid(imu_path):
            print(f"[IMU] Found existing IMU at {imu_path}")
        else:
            print(f"[IMU] IMU not found at {imu_path}. Creating...")

            # Create IMU sensor prim
            imu_sensor = self.world.scene.add(
                IMUSensor(
                    prim_path=imu_path,
                    name="imu",
                    frequency=60,
                    translation=np.array([0, 0, 0]),  
                )
            )

        self.wait_for_prim(imu_path)
    
    def respawn_bittle(self):
        n,_,_ = self.get_robot_dof()

        self.robot_view.set_joint_positions(np.zeros(n))
        self.robot_view.set_joint_velocities(np.zeros(n))
            
        self.robot_view.set_world_poses(positions = [self.spawn_cords], orientations = [[1, 0, 0, 0]])
        
        # prim = get_prim_at_path(self.robot_prim)
        
        # x, y, z = self.spawn_cords
        # xform = UsdGeom.Xformable(prim)
        # xform.ClearXformOpOrder()
        # xform.AddTranslateOp().Set(Gf.Vec3d(x, y, 1))


        
        



if __name__ == "__main__":

    b = Bittle()
    