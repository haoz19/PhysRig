"""
FBX to Pipeline Converter (Blender Script)

Imports an FBX file and exports:
  - mesh/mesh_frame_XXXX.obj      (skinned surface mesh per frame, from a mesh object)
  - skeleton_mesh/skeleton_frame_XXXX.obj  (icospheres at bone positions per frame, from an armature)

Usage:
  blender --background --python motionrep/datatools/fbx_to_pipeline.py -- \\
      <fbx_path> <output_dir> [--mesh_obj U3DMesh] [--skeleton_obj Armature]

The output_dir will contain mesh/ and skeleton_mesh/ subdirectories matching the
folder structure expected by infill.sh, points.sh, and train.sh.
"""

import bpy
import os
import sys
import argparse


def parse_args():
    """Parse arguments after '--' in the Blender command line."""
    argv = sys.argv
    if "--" not in argv:
        print("ERROR: No arguments provided after '--'")
        print("Usage: blender --background --python fbx_to_pipeline.py -- <fbx_path> <output_dir> [options]")
        sys.exit(1)
    argv = argv[argv.index("--") + 1:]

    parser = argparse.ArgumentParser(description="Convert FBX to pipeline mesh/skeleton_mesh OBJ sequences.")
    parser.add_argument("fbx_path", help="Path to the input FBX file")
    parser.add_argument("output_dir", help="Output directory (will contain mesh/ and skeleton_mesh/)")
    parser.add_argument("--mesh_obj", default="U3DMesh",
                        help="Name of the mesh object in the FBX (default: U3DMesh)")
    parser.add_argument("--skeleton_obj", default="Armature",
                        help="Name of the armature object in the FBX (default: Armature)")
    parser.add_argument("--icosphere_radius", type=float, default=0.02,
                        help="Radius of icospheres placed at bone positions (default: 0.02)")
    parser.add_argument("--icosphere_subdivisions", type=int, default=1,
                        help="Subdivision level for icospheres (default: 1)")
    return parser.parse_args(argv)


def clear_scene():
    """Remove all objects from the scene."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def import_fbx(fbx_path):
    """Import an FBX file into the scene."""
    print(f"Importing FBX: {fbx_path}")
    bpy.ops.import_scene.fbx(filepath=fbx_path)
    print("Imported objects:")
    for obj in bpy.context.scene.objects:
        print(f"  {obj.name} (type={obj.type})")


def find_object(name, expected_type=None):
    """Find an object by name in the scene."""
    # Try exact match first
    obj = bpy.data.objects.get(name)
    if obj is not None:
        if expected_type and obj.type != expected_type:
            print(f"WARNING: Object '{name}' is type '{obj.type}', expected '{expected_type}'")
        return obj

    # Try case-insensitive / partial match
    for obj in bpy.data.objects:
        if obj.name.lower() == name.lower():
            print(f"Found object '{obj.name}' (case-insensitive match for '{name}')")
            return obj

    # Try partial match (object name contains the search name)
    for obj in bpy.data.objects:
        if name.lower() in obj.name.lower():
            print(f"Found object '{obj.name}' (partial match for '{name}')")
            return obj

    return None


def export_mesh_frame(mesh_obj, output_path):
    """Export a single mesh object to OBJ for the current frame."""
    # Deselect all, then select only the mesh object
    bpy.ops.object.select_all(action="DESELECT")
    mesh_obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_obj

    # Export selected object to OBJ
    bpy.ops.export_scene.obj(
        filepath=output_path,
        use_selection=True,
        use_materials=False,
        use_normals=False,
        use_uvs=False,
        keep_vertex_order=True,
    )


def create_skeleton_mesh_frame(armature_obj, output_path, radius=0.02, subdivisions=1):
    """
    Create icospheres at each bone's head position and export as OBJ.

    Each bone becomes one connected component, so generate_skeleton_centroids.py
    (which does mesh.split() then part.vertices.mean()) will recover one centroid per bone.
    """
    # Collect bone head positions in world space from pose bones
    bone_positions = []
    for pose_bone in armature_obj.pose.bones:
        # pose_bone.head is in armature local space; transform to world space
        world_pos = armature_obj.matrix_world @ pose_bone.head
        bone_positions.append(world_pos)

    if not bone_positions:
        print(f"  WARNING: No bones found in armature '{armature_obj.name}'")
        return

    # Create icospheres at each bone position
    created_objects = []
    for i, pos in enumerate(bone_positions):
        bpy.ops.mesh.primitive_ico_sphere_add(
            radius=radius,
            subdivisions=subdivisions,
            location=pos,
        )
        ico = bpy.context.active_object
        ico.name = f"_bone_ico_{i}"
        created_objects.append(ico)

    # Select only the icospheres and export
    bpy.ops.object.select_all(action="DESELECT")
    for ico in created_objects:
        ico.select_set(True)
    bpy.context.view_layer.objects.active = created_objects[0]

    bpy.ops.export_scene.obj(
        filepath=output_path,
        use_selection=True,
        use_materials=False,
        use_normals=False,
        use_uvs=False,
    )

    # Delete the temporary icospheres
    bpy.ops.object.delete()


def main():
    args = parse_args()

    fbx_path = os.path.abspath(args.fbx_path)
    output_dir = os.path.abspath(args.output_dir)

    if not os.path.isfile(fbx_path):
        print(f"ERROR: FBX file not found: {fbx_path}")
        sys.exit(1)

    # Create output directories
    mesh_dir = os.path.join(output_dir, "mesh")
    skeleton_mesh_dir = os.path.join(output_dir, "skeleton_mesh")
    os.makedirs(mesh_dir, exist_ok=True)
    os.makedirs(skeleton_mesh_dir, exist_ok=True)

    # Clear scene and import FBX
    clear_scene()
    import_fbx(fbx_path)

    # Find the mesh and armature objects
    mesh_obj = find_object(args.mesh_obj, expected_type="MESH")
    armature_obj = find_object(args.skeleton_obj, expected_type="ARMATURE")

    if mesh_obj is None:
        print(f"ERROR: Mesh object '{args.mesh_obj}' not found in FBX.")
        print("Available objects:")
        for obj in bpy.data.objects:
            print(f"  {obj.name} (type={obj.type})")
        sys.exit(1)

    if armature_obj is None:
        print(f"ERROR: Armature object '{args.skeleton_obj}' not found in FBX.")
        print("Available objects:")
        for obj in bpy.data.objects:
            print(f"  {obj.name} (type={obj.type})")
        sys.exit(1)

    print(f"\nMesh object:     {mesh_obj.name} (type={mesh_obj.type})")
    print(f"Armature object: {armature_obj.name} (type={armature_obj.type})")
    print(f"  Bones: {len(armature_obj.pose.bones)}")
    for bone in armature_obj.pose.bones:
        print(f"    - {bone.name}")

    # Get animation range
    start_frame = bpy.context.scene.frame_start
    end_frame = bpy.context.scene.frame_end
    num_frames = end_frame - start_frame + 1
    print(f"\nAnimation range: {start_frame} to {end_frame} ({num_frames} frames)")
    print(f"Output directory: {output_dir}")
    print(f"  mesh/            -> {mesh_dir}")
    print(f"  skeleton_mesh/   -> {skeleton_mesh_dir}")
    print()

    # Export each frame
    for frame in range(start_frame, end_frame + 1):
        # 0-based index for output filenames
        idx = frame - start_frame

        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()

        # Export body mesh
        mesh_path = os.path.join(mesh_dir, f"mesh_frame_{idx:04d}.obj")
        export_mesh_frame(mesh_obj, mesh_path)

        # Export skeleton icospheres
        skeleton_path = os.path.join(skeleton_mesh_dir, f"skeleton_frame_{idx:04d}.obj")
        create_skeleton_mesh_frame(
            armature_obj,
            skeleton_path,
            radius=args.icosphere_radius,
            subdivisions=args.icosphere_subdivisions,
        )

        print(f"Frame {idx}/{num_frames - 1} (blender frame {frame}) -> exported")

    print(f"\nExport complete!")
    print(f"  Mesh frames:     {num_frames} OBJs in {mesh_dir}")
    print(f"  Skeleton frames: {num_frames} OBJs in {skeleton_mesh_dir}")


if __name__ == "__main__":
    main()
