import numpy as np
import math
import random
import logging
from omni.isaac.kit import SimulationApp

# # Initialize SimulationApp
# app = SimulationApp({
#     "headless": False,
#     "hide_ui": False,
#     "window_width": 1280,
#     "window_height": 720,
# })

from isaacsim.core.api import World
from pxr import UsdGeom, Gf
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.prims import is_prim_path_valid, get_prim_at_path
import omni.usd
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg, TerrainGeneratorCfg
from isaaclab.terrains.height_field import HfRandomUniformTerrainCfg
import isaaclab.sim as sim_utils

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingGround:
    all_bounds = []
    cell_size = 12.0
    z_offset = 0.0
    _next_index = 0
    _grid_size = 1

    sync = False
    _sync_seed = 42

    def __init__(self, size=10.0, color=(0.0, 0.0, 0.0), type="plane", static_friction=None, dynamic_friction=None):
        if type not in ["plane", "rocky"]:
            raise ValueError(f"Invalid terrain type: {type}. Must be 'plane' or 'rocky'.")
        self.size = size
        self.color = color
        self.type = type
        # Set default friction values based on terrain type
        self.static_friction = 0.6 if type == "plane" else 0.8 if static_friction is None else static_friction
        self.dynamic_friction = 0.4 if type == "plane" else 0.5 if dynamic_friction is None else dynamic_friction
        self.z = TrainingGround.z_offset
        TrainingGround.cell_size = self.size + 5.0
        self.stage = get_current_stage()
        self._point_cache = []

        self.prim_path, self.row, self.col, self.x_offset, self.y_offset = self._auto_reserve()
        self.create_ground_plane()
        self.enable_collision()
        self.set_visuals()
        self.set_friction_coeffs()

    @staticmethod
    def set_sync(enabled=True, seed=42):
        TrainingGround.sync = enabled
        TrainingGround._sync_seed = seed
        logger.info(f"[TrainingGround] Sync {'enabled' if enabled else 'disabled'} with seed={seed}")

    def _auto_reserve(self):
        i = TrainingGround._next_index
        TrainingGround._grid_size = max(TrainingGround._grid_size, math.ceil(math.sqrt(i + 1)))
        row, col = divmod(i, TrainingGround._grid_size)
        x = col * TrainingGround.cell_size
        y = row * TrainingGround.cell_size
        path = f"/World/GroundPlane_{i}"
        TrainingGround._next_index += 1
        return path, row, col, x, y

    def create_ground_plane(self):
        sub_terrains = {
            "plane": HfRandomUniformTerrainCfg(
                proportion=1.0 if self.type == "plane" else 0.0,
                noise_range=(0.0, 0.0),
                noise_step=0.1,
                horizontal_scale=0.1,
                vertical_scale=0.005,
                slope_threshold=0.0,
            ),
            "rocky": HfRandomUniformTerrainCfg(
                proportion=1.0 if self.type == "rocky" else 0.0,
                noise_range=(0.05, 0.20),
                noise_step=0.05,
                horizontal_scale=0.05,
                vertical_scale=0.01,
                slope_threshold=0.7,
            )
        }

        gen_cfg = TerrainGeneratorCfg(
            num_rows=1,
            num_cols=1,
            size=(self.size, self.size),
            vertical_scale=0.005,
            color_scheme="none",
            sub_terrains=sub_terrains,
            curriculum=False,
            border_width=0.0,
            border_height=1.0,
            seed=TrainingGround._sync_seed if TrainingGround.sync else None
        )

        imp_cfg = TerrainImporterCfg(
            prim_path=self.prim_path,
            terrain_type="generator",
            terrain_generator=gen_cfg,
            debug_vis=False,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=self.color),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=self.static_friction,
                dynamic_friction=self.dynamic_friction
            )
        )

        importer = TerrainImporter(imp_cfg)

        prim = get_prim_at_path(self.prim_path + "/terrain")
        if not prim.IsValid():
            raise RuntimeError(f"Failed to create terrain at {self.prim_path}/terrain")

        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        translate_op = xform.AddTranslateOp()
        translate_op.Set(Gf.Vec3f(self.x_offset, self.y_offset, self.z))
        logger.debug(f"Applied translation ({self.x_offset}, {self.y_offset}, {self.z}) to {self.prim_path}/terrain")

        self.importer = importer

    def enable_collision(self):
        mesh_path = f"{self.prim_path}"
        mesh_prim = get_prim_at_path(mesh_path)
        if not mesh_prim.IsValid():
            logger.error(f"Ground mesh not found at {mesh_path}")
            return

        collider_cfg = sim_utils.CollisionPropertiesCfg(collision_enabled=True)
        sim_utils.define_collision_properties(mesh_prim.GetPath(), collider_cfg)
        logger.debug(f"Static collision enabled on {mesh_path}")

    def set_visuals(self):
        logger.debug(f"Visuals set for {self.prim_path} with color {self.color}")

    def set_friction_coeffs(self):
        material_path = f"{self.prim_path}/physicsMaterial"
        material_prim = get_prim_at_path(material_path)
        if not material_prim.IsValid():
            logger.warning(f"Physics material not found at {material_path}")
            return

        physics_mat = sim_utils.RigidBodyMaterialCfg(
            static_friction=self.static_friction,
            dynamic_friction=self.dynamic_friction
        )
        sim_utils.bind_physics_material(get_prim_at_path(f"{self.prim_path}/terrain"), material_path)
        physics_mat.func(material_path, physics_mat)
        logger.info(f"Friction set on {material_path} for {self.prim_path}: static={self.static_friction}, dynamic={self.dynamic_friction}")

    def register_bounds(self):
        half = self.size / 2.0
        self.bounds = (
            self.x_offset - half, self.x_offset + half,
            self.y_offset - half, self.y_offset + half
        )
        TrainingGround.all_bounds.append(self.bounds)

    def get_world_translation(self):
        prim = get_prim_at_path(self.prim_path + "/terrain")
        if not prim.IsValid():
            logger.error(f"Invalid prim at: {self.prim_path}/terrain")
            return Gf.Vec3d(0, 0, 0)

        xform = UsdGeom.Xformable(prim)
        for op in xform.GetOrderedXformOps():
            if op.GetOpName() == "xformOp:translate":
                return op.Get()
        return Gf.Vec3d(0, 0, 0)

    def generate_points(self, n=10, spacing=None, margin=1.5):
        if TrainingGround.sync:
            random.seed(TrainingGround._sync_seed)

        points = []
        half = self.size / 2.0
        min_x = -half + margin
        max_x = half - margin
        min_y = -half + margin
        max_y = half - margin
        z_local = 0.4

        center = self.get_world_translation()
        x_offset, y_offset, z_offset = center[0], center[1], center[2]

        attempts = 0
        while len(points) < n:
            x = random.uniform(min_x, max_x)
            y = random.uniform(min_y, max_y)
            pt_local = (x, y)
            if spacing is None or all(
                np.linalg.norm(np.array(pt_local) - np.array(p[:2])) >= spacing for p in points
            ):
                pt_world = (x + x_offset, y + y_offset, z_local + z_offset)
                points.append(pt_world)

            attempts += 1
            if attempts > 500:
                raise RuntimeError(f"[TrainingGround] Failed to sample {n} spaced points for {self.prim_path}")
        self._point_cache = points

    def get_point(self, spacing=None, density_per_m2=0.5):
        if not self._point_cache:
            area = self.size * self.size
            n = int(area * density_per_m2)
            logger.info(f"{self.prim_path}: computing {n} points from area={area:.1f}")
            self.generate_points(n=n, spacing=spacing)
        return self._point_cache.pop()

    def spawn_debug_marker(self):
        stage = get_current_stage()
        marker_path = self.prim_path + "/Marker"
        if not is_prim_path_valid(marker_path):
            sphere = UsdGeom.Sphere.Define(stage, marker_path)
            sphere.CreateRadiusAttr(0.15)
            sphere.GetDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.0, 0.0)])
            UsdGeom.Xformable(sphere).AddTranslateOp().Set(self.get_world_translation())

def main():
    # Initialize World
    world = World(stage_units_in_meters=1.0, device="cuda")
    world.reset()

    n = 4
    size = 10.0
    gap_factor = 2.0
    prim_path = "/World/TerrainGridTest"

    logger.info("Starting terrain grid generation...")
    TrainingGround.set_sync(True, 42)

    # Compute grid dimensions
    num_rows = math.ceil(math.sqrt(n))
    num_cols = math.ceil(n / num_rows)
    logger.debug(f"Computed grid: {num_rows}x{num_cols}")

    # Create terrains with alternating types
    grounds = []
    for i in range(n):
        row, col = divmod(i, num_cols)
        terrain_type = "plane" if (row + col) % 2 == 0 else "rocky"
        color = (0.2, 0.2, 0.2) if terrain_type == "plane" else (0.3, 0.3, 0.3)
        ground = TrainingGround(
            size=size,
            color=color,
            type=terrain_type
        )
        grounds.append(ground)
        logger.info(f"Created {terrain_type} terrain at {ground.prim_path}")

    logger.info(f"Terrain grid spawned successfully with {len(grounds)} terrains")

    # Test point generation
    logger.info("Testing point generation for each training ground...")
    for ground in grounds:
        logger.info(f"Generating points for {ground.prim_path}")
        ground.generate_points(n=5, spacing=2.0, margin=1.0)
        while ground._point_cache:
            point = ground.get_point()
            logger.info(f"Point in {ground.prim_path}: {point}")

    logger.info("Entering simulation update loop...")
    while app.is_running():
        world.step(render=True)
        app.update()

if __name__ == "__main__":

    main()
