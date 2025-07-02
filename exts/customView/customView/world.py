import time
import numpy as np
from isaacsim.core.api import World, PhysicsContext
from pxr import UsdGeom, UsdPhysics, PhysxSchema, UsdShade, UsdLux, Gf
from isaacsim.core.utils.stage import get_current_stage, open_stage
from isaacsim.core.utils.prims import is_prim_path_valid, get_prim_at_path
from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.core.prims import RigidPrim
from isaacsim.core.utils.stage import is_stage_loading
from omni.isaac.core.simulation_context import SimulationContext
from omni.kit.async_engine import run_coroutine

from .Bittle import Bittle

import omni.usd

class Environment:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Environment, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return

        self._initialized = True
        self.physics = "/World/PhysicsScene"
        self.grnd_plane = "/World/GroundPlane"
        self.bitlles_count = 0
        self.bittlles = []
        self.spawn_points = []
        self.stage = None
        self.context = None
        self.clear_stage() 
        #physics_prim_path = self.physics
        self.context = SimulationContext()
        run_coroutine(self.context.initialize_simulation_context_async())
        # self.wait_for_physics_ready()
        
        # self.wait_for_prim(self.physics)

        self.world = World(stage_units_in_meters=1.0,physics_prim_path = self.physics,set_defaults=True)

        print("[ENV] physics context at :",self.world.get_physics_context(),flush=True)
        
        print(self.world.physics_sim_view,flush=True)
        
        self.world.reset()

        self.world.play()

        self.set_grnd_coeffs()
        # self.wait_for_physics()
        self.get_valid_positions_on_terrain()

    @classmethod
    def destroy(cls):
        """Manually clears the singleton instance"""
        cls._instance = None
    
    def clear_stage(self):
        # Create a new empty USD stage
        omni.usd.get_context().new_stage()
        self.stage = omni.usd.get_context().get_stage()

        # Wait for stage load
        while is_stage_loading():
            print("[Environment] Waiting for stage to finish loading", flush=True)
            time.sleep(0.1)

        # Define /World root Xform (required for Isaac Sim to function correctly)
        if not is_prim_path_valid("/World"):
            self.stage.DefinePrim("/World", "Xform")
        self.stage.SetDefaultPrim(self.stage.GetPrimAtPath("/World"))
        self.wait_for_prim("/World")

        # Define physics scene
        if not is_prim_path_valid(self.physics):
            UsdPhysics.Scene.Define(self.stage, self.physics)
            print("[Environment] Added physics scene", flush=True)
        
        self.wait_for_prim(self.physics)

        # Add dome light similar to default startup
        self.create_colored_dome_light()

        # Add ground plane
        if not is_prim_path_valid(self.grnd_plane):
            GroundPlane(prim_path=self.grnd_plane, size=10, color=np.array([0.5, 0.5, 0.5]), z_position=0)
        self.wait_for_prim(self.grnd_plane)

        print("[Environment] Stage reset complete. Default Isaac Sim-like world initialized.")

    
    def wait_for_physics_ready(self, timeout=5.0):
        app = omni.kit.app.get_app()
        t0 = time.time()
        while True:
            physics_ready = self.context.physics_sim_view is not None and self.context._physics_context is not None
            if physics_ready:
                break
            if time.time() - t0 > timeout:
                raise RuntimeError("Timeout waiting for physics to initialize.")
            print("[Environment] Waiting for physics context...", flush=True)
            app.update()
            time.sleep(0.01)

    def wait_for_prim(self, path, timeout=5.0):
        t0 = time.time()
        while not is_prim_path_valid(path):
            if time.time() - t0 > timeout:
                raise RuntimeError(f"Timed out waiting for prim: {path}")
            time.sleep(0.05)

    def get_valid_positions_on_terrain(self, n_samples=10):

        ground_prim = get_prim_at_path(self.grnd_plane)
        mesh_prim = ground_prim.GetChild("geom")

        mesh = UsdGeom.Mesh(mesh_prim)
        points = mesh.GetPointsAttr().Get()

        if not points:
            raise RuntimeError("Mesh points attribute could not be read.")

        point_array = np.array([[p[0], p[1], p[2]] for p in points])
        min_bounds = np.min(point_array, axis=0)
        max_bounds = np.max(point_array, axis=0)
        z_const = min_bounds[2]

        pos = [
            (np.random.uniform(min_bounds[0], max_bounds[0]),
            np.random.uniform(min_bounds[1], max_bounds[1]),
            z_const)
            for _ in range(n_samples)
        ]

        filtered = [p for p in pos if p not in self.spawn_points]

        return filtered

    def set_grnd_coeffs(self):
        ground_prim = self.stage.GetPrimAtPath(self.grnd_plane)
        binding_api = UsdShade.MaterialBindingAPI(ground_prim)
        material = binding_api.ComputeBoundMaterial()[0]

        if not material:
            raise RuntimeError("No material bound to ground plane.")

        physics_mat = UsdPhysics.MaterialAPI(material)
        physics_mat.CreateStaticFrictionAttr().Set(0.6)
        physics_mat.CreateDynamicFrictionAttr().Set(0.4)
        print("[Environment] Set ground friction successfully.")

    def add_bittles(self,n = 1):
        
        self.bitlles_count = n

        for idx in range(self.bitlles_count):
            
            pts =  self.get_valid_positions_on_terrain()

            cord = pts[np.random.choice(len(pts))]
            self.spawn_points.append(cord)

            try:
                b = Bittle(id = idx, cords = cord, world = self.world)
                self.bittlles.append(b)
            
            except Exception as e:
                import traceback
                print("[Environment] Error adding bittle", e)
                traceback.print_exc()


    def create_colored_dome_light(self,path="/Environment/DomeLight", color=(0.5, 0.5, 1.0), intensity=5000.0):

        if not is_prim_path_valid(path):
            dome = UsdLux.DomeLight.Define(self.stage, path)
            print(f"[Light] Created new DomeLight at {path}")
        else:
            dome = UsdLux.DomeLight(get_prim_at_path(path))
            print(f"[Light] DomeLight already exists at {path}")

        # Set light attributes
        dome.CreateColorAttr(Gf.Vec3f(*color))            # RGB color (0.0–1.0)
        dome.CreateIntensityAttr(intensity)               # Brightness
        dome.CreateTextureFileAttr("")                    # No HDRI texture (pure color light)