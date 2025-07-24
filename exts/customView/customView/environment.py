import time
import numpy as np
import omni.usd
from isaacsim.core.api import World, PhysicsContext
from pxr import UsdGeom, UsdPhysics, PhysxSchema, UsdShade, UsdLux, Gf
from isaacsim.core.utils.stage import get_current_stage, is_stage_loading
from isaacsim.core.utils.prims import is_prim_path_valid, get_prim_at_path
from isaacsim.core.api.objects.ground_plane import GroundPlane
from omni.isaac.core.simulation_context import SimulationContext

from Bittle import Bittle
from TrainingGround import TrainingGround
from tools import wait_for_prim, wait_for_stage_ready

class Environment:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Environment, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return

        # === Simulation & Stage ===
        self.stage = None
        self.context = None
        self.world = None

        # === Settings ===
        self._initialized: bool = True
        self.physics: str = "/World/PhysicsScene"
        self.bittles_count: int = 0

        # === Scene Elements ===
        self.bittles: list = []
        self.training_grounds: list = []
        self.spawn_points: list = []
        self.goal_points: list = []

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
        wait_for_stage_ready()
        self.context = SimulationContext(physics_prim_path=self.physics)
        self.wait_for_physics_context()
        print("[ENV] Environment initialization complete!")

    def wait_for_physics_context(self, timeout=10.0):
        app = omni.kit.app.get_app()
        t0 = time.time()
        while True:
            app.update()
            if self.context and hasattr(self.context, '_physics_context') and self.context._physics_context is not None:
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

        self._define_world_prim()
        self._define_physics_scene()
        self.create_colored_dome_light()
        print("[Environment] Stage reset complete. Default Isaac Sim-like world initialized.")

    def _define_world_prim(self):
        if not is_prim_path_valid("/World"):
            self.stage.DefinePrim("/World", "Xform")
        self.stage.SetDefaultPrim(self.stage.GetPrimAtPath("/World"))
        wait_for_prim("/World")

    def _define_physics_scene(self):
        if not is_prim_path_valid(self.physics):
            UsdPhysics.Scene.Define(self.stage, self.physics)
            print("[Environment] Added physics scene", flush=True)
        wait_for_prim(self.physics)

    def add_training_grounds(self, n=1, size=10.0):
        """
        Create and register `n` training ground planes of given size.
        """
        self.training_grounds.clear()
        for i in range(n):
            try:
                ground = TrainingGround(size=size)
                self.training_grounds.append(ground)
                print(f"[Environment] Training ground {i} created at {ground.path}", flush=True)
            except Exception as e:
                print(f"[Environment] Error creating training ground {i}:", e)
                import traceback
                traceback.print_exc()

    def add_bittles(self, n=1):
        """
        Spawn `n` Bittle robots, each on a unique training ground.
        """
        self.bittles_count = n
        self.bittles.clear()

        if len(self.training_grounds) < n:
            raise RuntimeError("Not enough training grounds created. Call spawn_training_grounds(n) first.")

        for idx in range(n):
            try:
                ground = self.training_grounds[idx]
                spawn = ground.get_point()

                b = Bittle(id=idx, cords=spawn, world=self.world)
                b.spawn_bittle()
                b.set_articulation()
                self.world.step(render=True)
                wait_for_stage_ready()
                self.bittles.append(b)

            except Exception as e:
                print(f"[Environment] Error adding bittle {idx}:", e)
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
        """
        Check for collisions between spawned Bittle robots and return their prim paths.
        """
        contact_api = PhysxSchema.PhysxSceneAPI(get_prim_at_path(self.physics))
        collisions = set()
        bittle_paths = [b.robot_prim for b in self.bittles if is_prim_path_valid(b.robot_prim)]

        for b in bittle_paths:
            contact_attr = self.stage.GetPrimAtPath(b).GetAttribute("physxContactReport:body0")
            if contact_attr and contact_attr.HasAuthoredValue():
                contacts = contact_attr.Get()
                if contacts:
                    for c in contacts:
                        if any(other in c for other in bittle_paths if other != b):
                            collisions.add(b)
        return collisions
