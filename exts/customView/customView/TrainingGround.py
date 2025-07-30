import numpy as np
import math
import random
from pxr import UsdGeom, UsdPhysics, UsdShade, Gf, PhysxSchema
from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.prims import is_prim_path_valid, get_prim_at_path
import omni.usd


class TrainingGround:
    all_bounds = []
    cell_size = 12.0
    z_offset = 0.0
    _next_index = 0
    _grid_size = 1

    sync = False             
    _sync_seed = 42          

    def __init__(self, size=10.0, color=(0.0, 0.0, 0.0)):
        self.size = size
        self.color = color
        self.z = TrainingGround.z_offset
        TrainingGround.cell_size = self.size + 2.0
        self.stage = get_current_stage()
        self._point_cache = []

        self.path, self.row, self.col, self.x_offset, self.y_offset = self._auto_reserve()

        self.prim = self.create_ground_plane()
        self.enable_collision()
        self.set_visuals()
        self.set_friction_coeffs()
        self.register_bounds()
    
    @staticmethod
    def set_sync(enabled=True, seed=42):
        TrainingGround.sync = enabled
        TrainingGround._sync_seed = seed
        print(f"[TrainingGround] Sync {'enabled' if enabled else 'disabled'} with seed={seed}")


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
        if not is_prim_path_valid(self.path):
            plane_prim = UsdGeom.Xform.Define(self.stage, self.path)
        else:
            plane_prim = get_prim_at_path(self.path)

        xform = UsdGeom.Xformable(plane_prim)
        xform.ClearXformOpOrder()
        translate_op = xform.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(self.x_offset, self.y_offset, self.z))

        mesh_path = self.path + "/Plane"
        if not is_prim_path_valid(mesh_path):
            mesh = UsdGeom.Mesh.Define(self.stage, mesh_path)
            s = self.size / 2.0
            mesh.CreatePointsAttr([
                Gf.Vec3f(-s, -s, 0),
                Gf.Vec3f(s, -s, 0),
                Gf.Vec3f(s, s, 0),
                Gf.Vec3f(-s, s, 0),
            ])
            mesh.CreateFaceVertexCountsAttr([4])
            mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
            mesh.CreateExtentAttr([Gf.Vec3f(-s, -s, 0), Gf.Vec3f(s, s, 0)])

        return plane_prim

    def enable_collision(self):
        mesh_path = self.path + "/Plane"
        mesh_prim = get_prim_at_path(mesh_path)
        if not mesh_prim.IsValid():
            print(f"[ERROR] Ground mesh not found at {mesh_path}")
            return

        # Apply general RigidBody schema (static)
        UsdPhysics.RigidBodyAPI.Apply(mesh_prim)
        UsdPhysics.CollisionAPI.Apply(mesh_prim)

        rigid_api = UsdPhysics.RigidBodyAPI(mesh_prim)
        rigid_api.CreateRigidBodyEnabledAttr().Set(False)

        # Optional: if PhysxCollisionAPI is available
        try:
            PhysxSchema.PhysxCollisionAPI.Apply(mesh_prim)
            physx_collision = PhysxSchema.PhysxCollisionAPI(mesh_prim)
            physx_collision.CreateCollisionEnabledAttr().Set(True)
            physx_collision.CreateCollisionApproximationAttr().Set("triangleMesh")
        except Exception as e:
            print("[Warning] Could not apply PhysX-specific collision schema:", e)

        print(f"[DEBUG] Static collision enabled on {mesh_path}")



    def set_visuals(self):
        def apply_color_recursively(prim, color):
            if prim.IsA(UsdGeom.Mesh):
                UsdGeom.Mesh(prim).CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])
            for child in prim.GetChildren():
                apply_color_recursively(child, color)

        apply_color_recursively(get_prim_at_path(self.path), self.color)

    def set_friction_coeffs(self, static_friction=0.6, dynamic_friction=0.4):
        root = self.prim
        binding_api = UsdShade.MaterialBindingAPI(root)
        material_tuple = binding_api.ComputeBoundMaterial()
        
        if not material_tuple:
            print(f"[Warning] No material bound to {self.path}, skipping friction setup.")
            return

        material = material_tuple[0]
        if not material or not material.GetPrim().IsValid():
            print(f"[Warning] Material prim invalid for {self.path}")
            return

        physics_mat = UsdPhysics.MaterialAPI(material)
        physics_mat.CreateStaticFrictionAttr().Set(static_friction)
        physics_mat.CreateDynamicFrictionAttr().Set(dynamic_friction)
        print(f"[INFO] Friction set on {material.GetPath()} for {self.path}")


    def register_bounds(self):
        half = self.size / 2.0
        self.bounds = (
            self.x_offset - half, self.x_offset + half,
            self.y_offset - half, self.y_offset + half
        )
        TrainingGround.all_bounds.append(self.bounds)


    def get_world_translation(self):
        prim = get_prim_at_path(self.path)  # this should be the Xform prim
        if not prim.IsValid():
            print(f"[ERROR] Invalid prim at: {self.path}")
            return Gf.Vec3d(0, 0, 0)

        xform = UsdGeom.Xformable(prim)
        for op in xform.GetOrderedXformOps():
            if op.GetOpName() == "xformOp:translate":
                return op.Get()  # <-- This is the Vec3d you're seeing in GUI
        return Gf.Vec3d(0, 0, 0)


    def generate_points(self, n=10, spacing=None, margin=1.5):
        """
        Uniformly samples `n` points within the local XY area of the ground tile.

        - Points are sampled in local tile coordinates centered at (0, 0),
        then translated using the ground tile’s world offset.
        - If `spacing` is set, ensures minimum distance between points.
        """
        if TrainingGround.sync:
            random.seed(TrainingGround._sync_seed)

        points = []
        half = self.size / 2.0

        # Local sampling bounds (centered at 0,0)
        min_x = -half + margin
        max_x = +half - margin
        min_y = -half + margin
        max_y = +half - margin
        z_local = 0.4  # relative to tile height

        # Get world-space tile offset
        center = self.get_world_translation()
        x_offset, y_offset, z_offset = center[0], center[1], center[2]

        attempts = 0
        
        while len(points) < n:
            x = random.uniform(min_x, max_x)
            y = random.uniform(min_y, max_y)
            pt_local = (x, y)

            # Check spacing against already accepted local points
            if spacing is None or all(
                np.linalg.norm(np.array(pt_local) - np.array(p[:2])) >= spacing for p in points
            ):
                # Convert to world coordinates and append
                pt_world = (x + x_offset, y + y_offset, z_local + z_offset)
                points.append(pt_world)

            attempts += 1
            if attempts > 500:
                raise RuntimeError(f"[TrainingGround] Failed to sample {n} spaced points for {self.path}")
        self._point_cache = points

    def get_point(self, spacing=None, density_per_m2=0.5):
        """
        Returns one cached point. Regenerates based on plane area if cache is empty.
        `n` is computed as (area × density_per_m2).
        """
        if not self._point_cache:
            area = self.size * self.size
            n = int(area * density_per_m2)
            print(f"[TrainingGround] {self.path}: computing {n} points from area={area:.1f}")
            self.generate_points(n=n, spacing=spacing)

        return self._point_cache.pop()


    # Optional: add visual markers at each training ground center for debug
    def spawn_debug_marker(self):
        stage = get_current_stage()
        marker_path = self.path + "/Marker"
        if not is_prim_path_valid(marker_path):
            sphere = UsdGeom.Sphere.Define(stage, marker_path)
            sphere.CreateRadiusAttr(0.15)
            sphere.GetDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.0, 0.0)])  # red
            UsdGeom.Xformable(sphere).AddTranslateOp().Set(self.get_world_translation())
