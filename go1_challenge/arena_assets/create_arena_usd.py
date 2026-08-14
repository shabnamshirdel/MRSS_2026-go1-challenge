import os
import math
from pathlib import Path

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import omni.usd
from pxr import Usd, UsdGeom, Gf, UsdShade, Sdf, UsdPhysics


APRILTAG_MDL = "assets/textures/AprilTag.mdl"


def define_apriltag(stage: Usd.Stage, tag_id: int, mosaic_path: str):
    """Define an additional tag36h11 mesh and its ID-selecting MDL material."""
    tag_name = f"tag_{tag_id:02d}"
    tag_path = f"/World/arena/tags/{tag_name}"
    tag_xform = UsdGeom.Xform.Define(stage, tag_path)
    # Copy the explicit cube topology, normals, and face-varying UVs from the
    # bundled tag. UsdGeom.Cube alone has no authored UVs, which makes the MDL
    # sample a single atlas texel and renders the tag as a blank square.
    template = UsdGeom.Mesh(stage.GetPrimAtPath("/World/arena/tags/tag_00/tag_00"))
    if not template.GetPrim().IsValid():
        raise RuntimeError("Bundled AprilTag template mesh was not found")
    tag_mesh = UsdGeom.Mesh.Define(stage, f"{tag_path}/{tag_name}")
    tag_mesh.CreatePointsAttr(template.GetPointsAttr().Get())
    tag_mesh.CreateFaceVertexCountsAttr(template.GetFaceVertexCountsAttr().Get())
    tag_mesh.CreateFaceVertexIndicesAttr(template.GetFaceVertexIndicesAttr().Get())
    tag_mesh.CreateNormalsAttr(template.GetNormalsAttr().Get())
    tag_mesh.SetNormalsInterpolation(template.GetNormalsInterpolation())
    template_st = UsdGeom.PrimvarsAPI(template).GetPrimvar("st")
    tag_st = UsdGeom.PrimvarsAPI(tag_mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, template_st.GetInterpolation()
    )
    tag_st.Set(template_st.Get())
    tag_mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)

    material = UsdShade.Material.Define(stage, f"/World/arena/tags/Looks/AprilTag_{tag_id:02d}")
    shader = UsdShade.Shader.Define(stage, f"{material.GetPath()}/Shader")
    shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    shader.SetSourceAsset(Sdf.AssetPath(APRILTAG_MDL), "mdl")
    shader.SetSourceAssetSubIdentifier("AprilTag", "mdl")
    shader.CreateInput("tag_id", Sdf.ValueTypeNames.Int).Set(tag_id)
    shader.CreateInput("tag_mosaic", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(mosaic_path))
    shader.CreateInput("tag_size", Sdf.ValueTypeNames.Int).Set(10)
    shader.CreateInput("tags_per_row", Sdf.ValueTypeNames.Int).Set(24)
    shader.CreateInput("spacing", Sdf.ValueTypeNames.Int).Set(1)
    shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    material.CreateDisplacementOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    material.CreateVolumeOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    UsdShade.MaterialBindingAPI.Apply(tag_mesh.GetPrim()).Bind(material)
    return tag_xform.GetPrim()


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

    # Keep the asset self-contained and compatible with the installed Isaac Sim
    # version by overriding the bundled materials to use the checked-in MDL.
    for tag_id in range(14):
        shader = UsdShade.Shader.Get(stage, f"/World/arena/tags/Looks/AprilTag_{tag_id:02d}/Shader")
        shader.SetSourceAsset(Sdf.AssetPath("assets/textures/AprilTag.mdl"), "mdl")

    # Define tag positions and transformations
    arena_half_size = arena_size / 2
    # The referenced tag meshes are 100 units wide. A 0.002 scale therefore
    # produces a 20 cm print (16 cm black square plus its white border).
    tag_size = 0.2 / 100
    tag_width = 0.2
    tag_gap = 0.125  # 12.5 cm clear gap, within the requested 10--15 cm range.
    tag_pitch = tag_width + tag_gap
    wall_offset = wall_thickness / 2
    z_height = 0.5

    def centered_offsets(count: int) -> list[float]:
        """Return evenly pitched offsets whose group is centered on a wall."""
        return [(index - (count - 1) / 2) * tag_pitch for index in range(count)]

    # Fill every wall with 15 unique tags at a 12.5 cm edge-to-edge gap.
    tag_transforms = []
    next_tag_id = 0
    tags_per_wall = 15
    for wall_name, count, rotation in (
        ("front", tags_per_wall, (90, 0, 0)),
        ("left", tags_per_wall, (0, 90, 90)),
        ("right", tags_per_wall, (0, -90, -90)),
        ("back", tags_per_wall, (-90, 0, 180)),
    ):
        for offset in centered_offsets(count):
            if wall_name == "front":
                position = (offset, arena_half_size - wall_offset, z_height)
            elif wall_name == "back":
                position = (offset, -arena_half_size + wall_offset, z_height)
            elif wall_name == "left":
                position = (-arena_half_size + wall_offset, offset, z_height)
            else:
                position = (arena_half_size - wall_offset, offset, z_height)
            tag_transforms.append((next_tag_id, position, rotation))
            next_tag_id += 1

    # The referenced asset supplies IDs 0--13. Define the remaining unique
    # tag36h11 IDs locally; the AprilTag MDL selects each code from the mosaic.
    bundled_tag_count = 14
    all_tag_ids = list(range(len(tag_transforms)))
    for tag_id in all_tag_ids[bundled_tag_count:]:
        define_apriltag(stage, tag_id, "assets/textures/tag36h11.png")
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
