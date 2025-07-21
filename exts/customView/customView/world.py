import time
import numpy as np
from isaacsim.core.api import World, PhysicsContext
from pxr import UsdGeom, UsdPhysics, PhysxSchema, UsdShade, UsdLux, Gf
from isaacsim.core.utils.stage import get_current_stage, is_stage_loading
from isaacsim.core.utils.prims import is_prim_path_valid, get_prim_at_path
from isaacsim.core.api.objects.ground_plane import GroundPlane
from omni.isaac.core.simulation_context import SimulationContext
from Bittle import Bittle
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
        self.bittles = []
        self.spawn_points = []
        self.goal_points = []
        self.stage = None
        self.context = None
        self.world = None

        self.setup_stage_and_physics()

        self.world = World(stage_units_in_meters=1.0, physics_prim_path=self.physics, set_defaults=True, device="cuda")
        PhysicsContext(prim_path=self.physics)

        self.world.reset()
        self.world.play()

    @classmethod
    def destroy(cls):
        cls._instance = None

    def is_running(self):
        return self.world.is_playing()

    def get_world(self):
        return self.world

    def setup_stage_and_physics(self):
        print("[ENV] Setting up stage and physics for Kit extension...")
        self.clear_stage()
        self.wait_for_stage_ready()
        self.context = SimulationContext(physics_prim_path=self.physics)
        self.wait_for_physics_context()
        self.set_grnd_coeffs()
        print("[ENV] Environment initialization complete!")

    def wait_for_stage_ready(self, timeout=10.0):
        app = omni.kit.app.get_app()
        timeline = omni.timeline.get_timeline_interface()
        t0 = time.time()
        while is_stage_loading() or not timeline:
            if time.time() - t0 > timeout:
                raise RuntimeError("Timeout waiting for stage to be ready")
            print("[ENV] Waiting for stage...", flush=True)
            app.update()
            time.sleep(0.1)

    def wait_for_physics_context(self, timeout=10.0):
        app = omni.kit.app.get_app()
        t0 = time.time()
        while True:
            app.update()
            if (self.context and 
                hasattr(self.context, '_physics_context') and 
                self.context._physics_context is not None):
                print("[ENV] Physics context ready!")
                break
            if time.time() - t0 > timeout:
                raise RuntimeError("Timeout waiting for physics context")
            print("[ENV] Waiting for physics context...", flush=True)
            time.sleep(0.1)

    def clear_stage(self):
        omni.usd.get_context().new_stage()
        
        self.stage = omni.usd.get_context().get_stage()
        
        while is_stage_loading():
            print("[Environment] Waiting for stage to finish loading", flush=True)
            time.sleep(0.1)
        
        if not is_prim_path_valid("/World"):
            self.stage.DefinePrim("/World", "Xform")
        
        self.stage.SetDefaultPrim(self.stage.GetPrimAtPath("/World"))
        self.wait_for_prim("/World")
        
        if not is_prim_path_valid(self.physics):
            UsdPhysics.Scene.Define(self.stage, self.physics)
            print("[Environment] Added physics scene", flush=True)
        self.wait_for_prim(self.physics)
        
        self.create_colored_dome_light()

        if not is_prim_path_valid(self.grnd_plane):
            GroundPlane(prim_path=self.grnd_plane, size=10, color=np.array([0.5, 0.5, 0.5]), z_position=0)
        
        self.wait_for_prim(self.grnd_plane)
        print("[Environment] Stage reset complete. Default Isaac Sim-like world initialized.")

    def wait_for_prim(self, path, timeout=5.0):
        t0 = time.time()
        while not is_prim_path_valid(path):
            if time.time() - t0 > timeout:
                raise RuntimeError(f"Timed out waiting for prim: {path}")
            time.sleep(0.05)

    def get_valid_positions_on_terrain(self, n_samples=20, line=False, axis='x', spacing=1.0, base=(0.0, 0.0, 0.0)):
        if line:
            min_distance = 2.0
            points = []
            i = 0
            while len(points) < n_samples:
                if axis == 'x':
                    pt = (base[0] + i * spacing, base[1], base[2])
                elif axis == 'y':
                    pt = (base[0], base[1] + i * spacing, base[2])
                else:
                    pt = (base[0], base[1], base[2] + i * spacing)

                # Check distance constraint
                if min_distance is None or all(np.linalg.norm(np.array(pt[:2]) - np.array(p[:2])) >= min_distance for p in points):
                    points.append(pt)

                i += 1
                if i > 500:
                    raise RuntimeError(f"Unable to find {n_samples} valid positions with spacing constraint")

            return points
        else:
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
            return [
                (np.random.uniform(min_bounds[0], max_bounds[0]),
                 np.random.uniform(min_bounds[1], max_bounds[1]),
                 z_const)
                for _ in range(n_samples)
            ]

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

    def setup_groundplane_visuals(self, color=(0.0, 0.0, 0.0)):
        """
        Sets the ground plane display color and optionally modifies mesh appearance.
        """
        ground_prim = self.stage.GetPrimAtPath(self.grnd_plane)

        def apply_color_recursively(prim, color):
            if prim.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(prim)
                mesh.CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])
            for child in prim.GetChildren():
                apply_color_recursively(child, color)

        if not ground_prim.IsValid():
            raise RuntimeError(f"Ground plane {self.grnd_plane} is not valid.")

        apply_color_recursively(ground_prim, color)
        print(f"[ENV] Ground plane color set to {color}")


    def add_bittles(self, n=1, m=3):
        self.bitlles_count = n 
        self.bittles = []
        self.spawn_points = self.get_valid_positions_on_terrain(
            n_samples=n, line=True, spacing=1.5, base=(-2.0, 0.0, 0.0)
        )
        self.goal_points = self.get_valid_positions_on_terrain(n_samples=m)

        for idx in range(n):
            cord = self.spawn_points[idx]
            try:
                b = Bittle(id=idx, cords=cord, world=self.world)
                b.spawn_bittle()
                self.world.reset()
                b.set_articulation()
                self.world.step(render=True)
                self.wait_for_stage_ready()
                self.bittles.append(b)
            except Exception as e:
                print("[Environment] Error adding bittle", e)
                import traceback
                traceback.print_exc()

    def create_colored_dome_light(self, path="/Environment/DomeLight", color=(0.4, 0.6, 1.0), intensity=5000.0):
        
        if not is_prim_path_valid(path):
            dome = UsdLux.DomeLight.Define(self.stage, path)
            print(f"[Light] Created new DomeLight at {path}")
        else:
            dome = UsdLux.DomeLight(get_prim_at_path(path))
            print(f"[Light] DomeLight already exists at {path}")
        dome.CreateColorAttr(Gf.Vec3f(*color))
        dome.CreateIntensityAttr(intensity)
        dome.CreateTextureFileAttr("")
    
    def get_collided_bittle_prim_paths(self):
        contact_api = PhysxSchema.PhysxSceneAPI(get_prim_at_path(self.physics))
        collisions = set()
        stage = self.stage
        bittle_paths = [b.robot_prim for b in self.bittles if is_prim_path_valid(b.robot_prim)]

        for b in bittle_paths:
            contact_attr = stage.GetPrimAtPath(b).GetAttribute("physxContactReport:body0")
            if contact_attr and contact_attr.HasAuthoredValue():
                contacts = contact_attr.Get()
                if contacts:
                    for c in contacts:
                        if any(other in c for other in bittle_paths if other != b):
                            collisions.add(b)
        return collisions

