import time
import numpy as np
from isaacsim.core.api import World
from pxr import UsdGeom, UsdPhysics, PhysxSchema, UsdShade
from isaacsim.core.utils.stage import get_current_stage, open_stage
from isaacsim.core.utils.prims import is_prim_path_valid, get_prim_at_path
from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.core.prims import RigidPrim

class Environment:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Environment, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return  # Avoid re-initializing on repeated calls

        self._initialized = True

        open_stage("/home/dafodilrat/Documents/bu/RASTIC/rl_world.usd")

        self.grnd_plane = "/World/GroundPlane"
        self.stage = get_current_stage()
        self.world = World(stage_units_in_meters=1.0)
        
        grnd = GroundPlane(prim_path=self.grnd_plane, size=10, color=np.array([0.5, 0.5, 0.5])) 
        self.wait_for_prim(self.grnd_plane)
        self.set_grnd_coeffs()

        self.wait_for_prim("/World/PhysicsScene")
        

    def wait_for_prim(self, path, timeout=5.0):
        t0 = time.time()
        while not is_prim_path_valid(path):
            if time.time() - t0 > timeout:
                raise RuntimeError(f"Timed out waiting for prim: {path}")
            time.sleep(0.05)

    def get_valid_positions_on_terrain(self, n_samples=10):
        
        ground_prim = get_prim_at_path(self.grnd_plane)
        mesh_prim = ground_prim.GetChild("geom")

        # if mesh_prim.GetTypeName() != "CollissionPlane":
        #     raise RuntimeError(f"Expected Mesh prim, got: {mesh_prim.GetTypeName()} at {mesh_prim.GetPath()}")

        mesh = UsdGeom.Mesh(mesh_prim)
        points = mesh.GetPointsAttr().Get()

        if not points:
            raise RuntimeError("Mesh points attribute could not be read.")

        point_array = np.array([[p[0], p[1], p[2]] for p in points])
        min_bounds = np.min(point_array, axis=0)
        max_bounds = np.max(point_array, axis=0)
        z_const = min_bounds[2]

        positions = [
            (np.random.uniform(min_bounds[0], max_bounds[0]),
            np.random.uniform(min_bounds[1], max_bounds[1]),
            z_const)
            for _ in range(n_samples)
        ]

        return positions
    
    def set_grnd_coeffs(self):

        ground_prim = self.stage.GetPrimAtPath(self.grnd_plane)

        # Get bound physics material
        binding_api = UsdShade.MaterialBindingAPI(ground_prim)
        material = binding_api.ComputeBoundMaterial()[0]

        if not material:
            raise RuntimeError("No material bound to ground plane.")

        # Get physics material schema from material
        physics_mat = UsdPhysics.MaterialAPI(material)
        physics_mat.CreateStaticFrictionAttr().Set(0.6)
        physics_mat.CreateDynamicFrictionAttr().Set(0.4)
        print("[Environment] Set ground friction successfully.")
