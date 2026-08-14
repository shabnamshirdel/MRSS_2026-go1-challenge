import os
import math
from pathlib import Path

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import omni.usd
from pxr import Usd, UsdGeom, Gf, UsdShade, Sdf, UsdPhysics


def create_arena_usd(output_path: str, arena_size: float = 5.0):
    """Create a 5x5 arena USD file with walls, ArUco tags, and obstacle spawn points."""

    # Create new stage
    stage = Usd.Stage.CreateNew(output_path)

    # Set up scene
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    # Create root prim
    root_prim = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root_prim.GetPrim())

    # #! Create ground plane
    # ground = UsdGeom.Mesh.Define(stage, "/World/ground")
    # ground.CreatePointsAttr(
    #     [
    #         (-arena_size / 2, -arena_size / 2, 0),
    #         (arena_size / 2, -arena_size / 2, 0),
    #         (arena_size / 2, arena_size / 2, 0),
    #         (-arena_size / 2, arena_size / 2, 0),
    #     ]
    # )
    # ground.CreateFaceVertexCountsAttr([4])
    # ground.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    # ground.CreateNormalsAttr([(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)])

    # # Add physics to ground
    # ground_collision = UsdPhysics.CollisionAPI.Apply(ground.GetPrim())

    #! Create walls
    wall_height = 1.0
    wall_thickness = 0.2
    half_size = arena_size / 2
    wall_length = arena_size + wall_thickness

    wall_configs = [
        ("wall_right", (half_size, 0, wall_height / 2), (wall_thickness, wall_length, wall_height)),
        ("wall_left", (-half_size, 0, wall_height / 2), (wall_thickness, wall_length, wall_height)),
        ("wall_front", (0, half_size, wall_height / 2), (wall_length, wall_thickness, wall_height)),
        ("wall_back", (0, -half_size, wall_height / 2), (wall_length, wall_thickness, wall_height)),
    ]

    for name, pos, size in wall_configs:
        wall_prim_path = f"/World/arena/{name}"
        wall = UsdGeom.Cube.Define(stage, wall_prim_path)
        wall.CreateSizeAttr(1.0)  # Unit cube
        wall.CreateExtentAttr([(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)])

        # Transform wall
        wall_xform = UsdGeom.Xformable(wall)
        wall_xform.AddTranslateOp().Set(pos)
        wall_xform.AddScaleOp().Set(size)

        # Add physics
        UsdPhysics.CollisionAPI.Apply(wall.GetPrim())
        rigid_body = UsdPhysics.RigidBodyAPI.Apply(wall.GetPrim())
        rigid_body.CreateKinematicEnabledAttr(True)

    #! ArUco tags - Reference the USD with all tags and transform them
    # Reference the AprilTag USD file containing all tags and materials
    tags_xform = UsdGeom.Xform.Define(stage, "/World/arena/tags")
    apriltag_usd_path = "assets/april_tags.usd"  # Adjust filename as needed
    tags_xform.GetPrim().GetReferences().AddReference(apriltag_usd_path)

    # Define tag positions and transformations
    arena_half_size = arena_size / 2
    tag_size = 0.2 / 100
    corner_offset = 1.25
    wall_offset = wall_thickness / 2
    z_height = 0.3
    high_tag_z = 0.7

    tag_transforms = [
        # (tag_id, position, rotation)
        (0, (arena_half_size - corner_offset, arena_half_size - wall_offset, z_height), (90, 0, 0)),  # top-right
        (1, (-arena_half_size + corner_offset, arena_half_size - wall_offset, z_height), (90, 0, 0)),  # top-left
        (2, (-arena_half_size + wall_offset, arena_half_size - corner_offset, z_height), (0, 90, 90)),  # left-top
        (3, (-arena_half_size + wall_offset, -arena_half_size + corner_offset, z_height), (0, 90, 90)),  # left-bottom
        (4, (arena_half_size - wall_offset, +arena_half_size - corner_offset, z_height), (0, -90, -90)),  # right-top
        (5, (arena_half_size - wall_offset, -arena_half_size + corner_offset, z_height), (0, -90, -90)),  # right-bottom
        (6, (arena_half_size - corner_offset, -arena_half_size + wall_offset, z_height), (-90, 0, 180)),  # bottom-right
        (7, (-arena_half_size + corner_offset, -arena_half_size + wall_offset, z_height), (-90, 0, 180)),  # bottom-left
        # Extra landmarks at the centre of each wall. These improve tag
        # visibility when terrain or obstacles occlude the corner landmarks.
        (8, (0.0, arena_half_size - wall_offset, z_height), (90, 0, 0)),  # top-centre
        (9, (0.0, -arena_half_size + wall_offset, z_height), (-90, 0, 180)),  # bottom-centre
        (10, (-arena_half_size + wall_offset, 0.0, z_height), (0, 90, 90)),  # left-centre
        (11, (arena_half_size - wall_offset, 0.0, z_height), (0, -90, -90)),  # right-centre
        # Higher landmarks remain visible when the lower row is occluded.
        (12, (0.0, arena_half_size - wall_offset, high_tag_z), (90, 0, 0)),  # top-centre high
        (13, (0.0, -arena_half_size + wall_offset, high_tag_z), (-90, 0, 180)),  # bottom-centre high
    ]

    # List of all possible tag IDs (0-13)
    all_tag_ids = list(range(14))  # Adjust range based on your USD file
    active_tag_ids = [tag_id for tag_id, _, _ in tag_transforms]
    unused_tag_ids = [tag_id for tag_id in all_tag_ids if tag_id not in active_tag_ids]

    # Transform and show active tags
    for tag_id, pos, rot in tag_transforms:
        # Get the referenced tag prim
        tag_prim_path = f"/World/arena/tags/tag_{tag_id:02d}"
        tag_prim = stage.GetPrimAtPath(tag_prim_path)

        if tag_prim.IsValid():
            # Apply transform to the tag
            tag_xform = UsdGeom.Xformable(tag_prim)

            # Clear existing transform operations first
            tag_xform.ClearXformOpOrder()

            # Now add new transform operations
            tag_xform.AddTranslateOp().Set(pos)

            print(f"Tag {tag_id} loc: {pos} with rotation {rot}")

            # Apply rotation
            tag_xform.AddRotateXOp().Set(rot[0])
            tag_xform.AddRotateYOp().Set(rot[1])
            tag_xform.AddRotateZOp().Set(rot[2])

            # Apply scaling
            tag_xform.AddScaleOp().Set((tag_size, tag_size, 0.01 / 100))
        else:
            print(f"Warning: Tag prim {tag_prim_path} not found in referenced USD")

    # Hide unused tags
    for tag_id in unused_tag_ids:
        tag_prim_path = f"/World/arena/tags/tag_{tag_id:02d}"
        tag_prim = stage.GetPrimAtPath(tag_prim_path)

        if tag_prim.IsValid():
            # Hide the tag by setting visibility
            imageable = UsdGeom.Imageable(tag_prim)
            imageable.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
            print(f"Hiding unused tag: {tag_prim_path}")

    # Create obstacle spawn points (will be used by event terms)
    spawn_points = [
        (1.0, 1.0, 0.25, "prism_1"),  # Prism 1
        (-1.0, -1.0, 0.25, "prism_2"),  # Prism 2
        (1.0, -1.0, 0.25, "cylinder_1"),  # Cylinder
    ]

    for x, y, z, name in spawn_points:
        # Create placeholder prims that will be replaced by actual obstacles
        placeholder = UsdGeom.Xform.Define(stage, f"/World/{name}_spawn")
        placeholder.AddTranslateOp().Set((x, y, z))

    # Save the stage
    stage.GetRootLayer().Save()
    print(f"Arena USD saved to: {output_path}")


if __name__ == "__main__":
    # Create assets directory if it doesn't exist
    assets_dir = Path(__file__).parent
    # assets_dir.mkdir(exist_ok=True)

    output_file = assets_dir / "arena_5x5.usd"
    create_arena_usd(str(output_file))

    print("Arena generation completed.")
