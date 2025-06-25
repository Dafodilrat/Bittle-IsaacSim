from isaacsim.core.utils.prims import is_prim_path_valid, get_prim_at_path
from isaacsim.core.prims import Articulation
from isaacsim.core.api import World
from isaacsim.sensors.physics import _sensor
from pxr import UsdGeom, Sdf
from scipy.spatial.transform import Rotation as R
import time
import numpy as np

class Bittle():

    def __init__(self):
        
        self.robot_prim = "/World/bittle"
        self.world = World(stage_units_in_meters=1.0)
        self.wait_for_prim(self.robot_prim)
        self.reset()
    
    def reset(self):
        
        # Wait for prims


        prim = get_prim_at_path(self.robot_prim)
        if not prim.HasAttribute("articulation:root"):
            attr = prim.CreateAttribute("articulation:root", Sdf.ValueTypeNames.Bool)
            attr.Set(True)
            print("Marked as articulation root")
        else:
            print("Already has articulation:root =", prim.GetAttribute("articulation:root").Get())

        # simulation_view = tensors.create_simulation_view("torch")
        # articulation_view = simulation_view.create_articulation_view("/world/bittle*")

        # Safe physics init
        # self.sim.initialize_physics()
        # phy = SimulationManager.get_physics_sim_view()
        # print("physics :",phy)
        # print("device :",SimulationManager.get_physics_sim_device())
        
        self.world.reset()
        self.robot_view = Articulation(self.robot_prim)
        self.robot_view.initialize()

        self.world.reset()
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


if __name__ == "__main__":

    b = Bittle()
    