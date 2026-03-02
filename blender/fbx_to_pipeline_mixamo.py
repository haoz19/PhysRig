"""
FBX to Pipeline Converter for Mixamo FBX (Blender Script)

Imports an FBX file and exports:
  - mesh/mesh_frame_XXXX.obj      (skinned surface mesh per frame, from a mesh object)
  - skeleton_mesh/skeleton_frame_XXXX.obj  (icospheres at bone positions per frame, from an armature)

Usage:
  blender --background --python fbx_to_pipeline_mixamo.py -- \\
      <fbx_path> <output_dir> [--mesh_obj <name>] [--skeleton_obj <name>]

The output_dir will contain mesh/ and skeleton_mesh/ subdirectories matching the
folder structure expected by infill.sh, points.sh, and train.sh.
"""

import bpy
import os
import sys
import argparse
import math
from mathutils import Vector


def parse_args():
    """Parse arguments after '--' in the Blender command line."""
    argv = sys.argv
    if "--" not in argv:
        print("ERROR: No arguments provided after '--'")
        print("Usage: blender --background --python fbx_to_pipeline_mixamo.py -- <fbx_path> <output_dir> [options]")
        sys.exit(1)
    argv = argv[argv.index("--") + 1:]

    parser = argparse.ArgumentParser(description="Convert FBX to pipeline mesh/skeleton_mesh OBJ sequences.")
    parser.add_argument("fbx_path", help="Path to the input FBX file")
    parser.add_argument("output_dir", help="Output directory (will contain mesh/ and skeleton_mesh/)")
    parser.add_argument("--mesh_obj", default="",
                        help="Optional mesh object name override; if empty, auto-detect largest skinned mesh")
    parser.add_argument("--skeleton_obj", default="",
                        help="Optional armature object name override; if empty, auto-detect first armature")
    parser.add_argument("--icosphere_radius", type=float, default=0.02,
                        help="Radius of icospheres placed at bone positions (default: 0.02)")
    parser.add_argument("--icosphere_subdivisions", type=int, default=1,
                        help="Subdivision level for icospheres (default: 1)")
    parser.add_argument("--scale", type=float, default=15.0,
                        help="Uniform scale factor applied at OBJ export time (default: 15.0)")
    parser.add_argument("--skeleton_mode", choices=("weighted_centroids", "bone_heads", "none"),
                        default="weighted_centroids",
                        help="How to export skeleton points (default: weighted_centroids)")
    parser.add_argument("--skeleton_fallback", choices=("bone_heads", "error", "skip"),
                        default="bone_heads",
                        help="Fallback when weighted_centroids cannot be computed (default: bone_heads)")
    parser.add_argument("--min_group_weight", type=float, default=1e-6,
                        help="Minimum per-vertex group weight considered for weighted centroids (default: 1e-6)")
    parser.add_argument("--frame_range_source", choices=("armature_action", "scene"),
                        default="armature_action",
                        help="Frame range source for export (default: armature_action)")
    parser.add_argument("--preserve_uv_materials", action="store_true",
                        help="If set, export OBJ with UVs and materials (default: disabled)")
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


def pick_first_armature():
    """Pick the first armature object in scene order."""
    for obj in bpy.context.scene.objects:
        if obj.type == "ARMATURE":
            return obj, "first armature in scene order"
    return None, "no armature objects found"


def mesh_has_armature_modifier(mesh_obj):
    """Return True if the mesh has any armature modifier."""
    for mod in mesh_obj.modifiers:
        if mod.type == "ARMATURE":
            return True
    return False


def pick_largest_mesh(prefer_skinned=True):
    """
    Pick mesh by vertex count, optionally preferring skinned meshes.

    A skinned mesh is one that has an armature modifier and/or non-empty vertex groups.
    Returns (mesh_obj_or_none, reason_string).
    """
    all_meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not all_meshes:
        return None, "no mesh objects found"

    scored = []
    for mesh in all_meshes:
        vcount = len(mesh.data.vertices)
        has_vgroups = len(mesh.vertex_groups) > 0
        has_arm_mod = mesh_has_armature_modifier(mesh)
        skinned_score = 1 if (has_vgroups or has_arm_mod) else 0
        scored.append((skinned_score, vcount, mesh, has_vgroups, has_arm_mod))

    if prefer_skinned:
        skinned_only = [row for row in scored if row[0] == 1]
        if skinned_only:
            skinned_only.sort(key=lambda row: (row[1], row[2].name))
            _, vcount, mesh, has_vgroups, has_arm_mod = skinned_only[-1]
            return (
                mesh,
                f"largest skinned mesh by vertex count "
                f"(vertices={vcount}, vgroups={has_vgroups}, armature_modifier={has_arm_mod})",
            )

    scored.sort(key=lambda row: (row[1], row[2].name))
    _, vcount, mesh, has_vgroups, has_arm_mod = scored[-1]
    return (
        mesh,
        f"largest mesh by vertex count fallback "
        f"(vertices={vcount}, vgroups={has_vgroups}, armature_modifier={has_arm_mod})",
    )


def resolve_mixamo_objects(mesh_name_override, armature_name_override, need_armature=True):
    """
    Resolve mesh and armature using explicit overrides first, then Mixamo-friendly autodetection.

    Returns (mesh_obj, armature_obj_or_none).
    Exits with a readable error when required objects cannot be found.
    """
    mesh_obj = None
    armature_obj = None

    if mesh_name_override:
        mesh_obj = find_object(mesh_name_override, expected_type="MESH")
        if mesh_obj is None:
            print(f"ERROR: Mesh object override '{mesh_name_override}' not found in FBX.")
            print("Available objects:")
            for obj in bpy.data.objects:
                print(f"  {obj.name} (type={obj.type})")
            sys.exit(1)
        print(f"Resolved mesh from override: {mesh_obj.name}")
    else:
        mesh_obj, mesh_reason = pick_largest_mesh(prefer_skinned=True)
        if mesh_obj is None:
            print("ERROR: Could not auto-detect mesh object.")
            print(f"Reason: {mesh_reason}")
            print("Available objects:")
            for obj in bpy.data.objects:
                print(f"  {obj.name} (type={obj.type})")
            sys.exit(1)
        print(f"Auto-selected mesh: {mesh_obj.name} ({mesh_reason})")

    if not need_armature:
        return mesh_obj, None

    if armature_name_override:
        armature_obj = find_object(armature_name_override, expected_type="ARMATURE")
        if armature_obj is None:
            print(f"ERROR: Armature object override '{armature_name_override}' not found in FBX.")
            print("Available objects:")
            for obj in bpy.data.objects:
                print(f"  {obj.name} (type={obj.type})")
            sys.exit(1)
        print(f"Resolved armature from override: {armature_obj.name}")
    else:
        armature_obj, arm_reason = pick_first_armature()
        if armature_obj is None:
            print("ERROR: Could not auto-detect armature object.")
            print(f"Reason: {arm_reason}")
            print("Available objects:")
            for obj in bpy.data.objects:
                print(f"  {obj.name} (type={obj.type})")
            sys.exit(1)
        print(f"Auto-selected armature: {armature_obj.name} ({arm_reason})")

    return mesh_obj, armature_obj


def export_selected_obj(output_path, keep_vertex_order=False, scale=1.0, preserve_uv_materials=False):
    """Export selected objects to OBJ with Blender-version compatibility."""
    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")

    # Blender <= 3.x operator (can appear present but still fail if addon/operator isn't available).
    try:
        kwargs = {
            "filepath": output_path,
            "use_selection": True,
            "use_materials": bool(preserve_uv_materials),
            "use_normals": False,
            "use_uvs": bool(preserve_uv_materials),
            "keep_vertex_order": keep_vertex_order,
        }
        if scale != 1.0:
            kwargs["global_scale"] = scale
        bpy.ops.export_scene.obj(**kwargs)
        return
    except (AttributeError, RuntimeError, TypeError):
        pass

    # Blender 4.x operator
    if hasattr(bpy.ops.wm, "obj_export"):
        prop_keys = set(bpy.ops.wm.obj_export.get_rna_type().properties.keys())
        kwargs = {"filepath": output_path}

        # Map options defensively: only pass keys this Blender build supports.
        for key in ("export_selected_objects", "use_selection"):
            if key in prop_keys:
                kwargs[key] = True
                break
        for key in ("export_materials", "use_materials"):
            if key in prop_keys:
                kwargs[key] = bool(preserve_uv_materials)
        for key in ("export_normals", "use_normals"):
            if key in prop_keys:
                kwargs[key] = False
        for key in ("export_uv", "export_uvs", "use_uvs"):
            if key in prop_keys:
                kwargs[key] = bool(preserve_uv_materials)
                break
        if keep_vertex_order and "keep_vertex_order" in prop_keys:
            kwargs["keep_vertex_order"] = True
        if scale != 1.0:
            for key in ("global_scale", "scale"):
                if key in prop_keys:
                    kwargs[key] = scale
                    break

        bpy.ops.wm.obj_export(**kwargs)
        return

    print("ERROR: No OBJ export operator found (tried bpy.ops.export_scene.obj and bpy.ops.wm.obj_export).")
    sys.exit(1)


def export_mesh_frame(mesh_obj, output_path, scale=1.0, preserve_uv_materials=False):
    """Export a single mesh object to OBJ for the current frame."""
    # Deselect all, then select only the mesh object
    bpy.ops.object.select_all(action="DESELECT")
    mesh_obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_obj

    export_selected_obj(
        output_path,
        keep_vertex_order=True,
        scale=scale,
        preserve_uv_materials=preserve_uv_materials,
    )


def export_positions_as_icospheres(positions, output_path, radius=0.02, subdivisions=1, scale=1.0):
    """Create one icosphere per position and export as OBJ."""
    if not positions:
        print("  WARNING: No skeleton positions to export")
        return False

    created_objects = []
    for i, pos in enumerate(positions):
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

    export_selected_obj(output_path, keep_vertex_order=False, scale=scale)

    # Delete the temporary icospheres
    bpy.ops.object.delete()
    return True


def get_pose_bone_names(armature_obj):
    """Return armature pose bone names in stable armature order."""
    return [pose_bone.name for pose_bone in armature_obj.pose.bones]


def get_group_matched_bone_names(mesh_obj, armature_obj):
    """Return pose bones that have a matching mesh vertex-group name."""
    group_names = {group.name for group in mesh_obj.vertex_groups}
    return [name for name in get_pose_bone_names(armature_obj) if name in group_names]


def select_bone_head_names(mesh_obj, armature_obj):
    """
    Bone-head export bone selection.

    Prefer deform bones that also have a matching vertex group.
    If none are found, use all pose bones.
    """
    group_names = {group.name for group in mesh_obj.vertex_groups}
    deform_names = {bone.name for bone in armature_obj.data.bones if bone.use_deform}
    preferred = [
        name for name in get_pose_bone_names(armature_obj)
        if name in deform_names and name in group_names
    ]
    if preferred:
        return preferred
    return get_pose_bone_names(armature_obj)


def bone_head_positions_world(armature_obj, bone_names):
    """Get world-space head positions for named pose bones."""
    positions = []
    for name in bone_names:
        pose_bone = armature_obj.pose.bones.get(name)
        if pose_bone is None:
            continue
        positions.append(armature_obj.matrix_world @ pose_bone.head)
    return positions


def weighted_centroid_positions_world(armature_obj, mesh_obj, bone_names, min_group_weight=1e-6):
    """
    Compute one skinning-weighted centroid per bone from deformed mesh vertices.

    Returns (positions, num_zero_weight_bones).
    """
    if not bone_names:
        return [], 0

    bone_index = {name: idx for idx, name in enumerate(bone_names)}
    group_index_to_bone = {}
    for group in mesh_obj.vertex_groups:
        mapped_idx = bone_index.get(group.name)
        if mapped_idx is not None:
            group_index_to_bone[group.index] = mapped_idx

    depsgraph = bpy.context.evaluated_depsgraph_get()
    mesh_eval_obj = mesh_obj.evaluated_get(depsgraph)
    mesh_eval_data = mesh_eval_obj.to_mesh()
    base_vertices = mesh_obj.data.vertices

    if len(mesh_eval_data.vertices) != len(base_vertices):
        mesh_eval_obj.to_mesh_clear()
        raise RuntimeError(
            f"Vertex count mismatch in weighted centroid export: "
            f"evaluated={len(mesh_eval_data.vertices)} base={len(base_vertices)}"
        )

    sums = [Vector((0.0, 0.0, 0.0)) for _ in bone_names]
    weights = [0.0 for _ in bone_names]

    for idx, base_vertex in enumerate(base_vertices):
        world_pos = mesh_eval_obj.matrix_world @ mesh_eval_data.vertices[idx].co
        for elem in base_vertex.groups:
            mapped_idx = group_index_to_bone.get(elem.group)
            if mapped_idx is None:
                continue
            if elem.weight < min_group_weight:
                continue
            sums[mapped_idx] += world_pos * elem.weight
            weights[mapped_idx] += elem.weight

    mesh_eval_obj.to_mesh_clear()

    positions = []
    zero_weight_count = 0
    for idx, name in enumerate(bone_names):
        if weights[idx] > 0.0:
            positions.append(sums[idx] / weights[idx])
        else:
            pose_bone = armature_obj.pose.bones.get(name)
            if pose_bone is None:
                continue
            positions.append(armature_obj.matrix_world @ pose_bone.head)
            zero_weight_count += 1

    return positions, zero_weight_count


def resolve_export_frame_range(scene, armature_obj, frame_range_source):
    """
    Resolve export frame range as integer inclusive [start, end].

    Returns (start_frame, end_frame, resolved_source, warning_message_or_none).
    """
    warning_message = None
    if frame_range_source == "armature_action":
        action = None
        if armature_obj is not None and armature_obj.animation_data is not None:
            action = armature_obj.animation_data.action
        if action is not None:
            action_start, action_end = action.frame_range
            start_frame = int(math.floor(action_start))
            end_frame = int(math.ceil(action_end))
            if end_frame < start_frame:
                raise RuntimeError(
                    f"Invalid action frame range for '{action.name}': "
                    f"{action_start}..{action_end}"
                )
            return start_frame, end_frame, "armature_action", None

        warning_message = (
            "WARNING: --frame_range_source=armature_action was requested, but no active "
            "armature action was found. Falling back to scene frame range."
        )

    start_frame = int(scene.frame_start)
    end_frame = int(scene.frame_end)
    if end_frame < start_frame:
        raise RuntimeError(
            f"Invalid scene frame range: {scene.frame_start}..{scene.frame_end}"
        )
    return start_frame, end_frame, "scene", warning_message


def main():
    args = parse_args()

    fbx_path = os.path.abspath(args.fbx_path)
    output_dir = os.path.abspath(args.output_dir)

    if not os.path.isfile(fbx_path):
        print(f"ERROR: FBX file not found: {fbx_path}")
        sys.exit(1)

    # Create output directories
    mesh_dir = os.path.join(output_dir, "mesh")
    os.makedirs(mesh_dir, exist_ok=True)
    skeleton_mesh_dir = os.path.join(output_dir, "skeleton_mesh")
    if args.skeleton_mode != "none":
        os.makedirs(skeleton_mesh_dir, exist_ok=True)

    # Clear scene and import FBX
    clear_scene()
    import_fbx(fbx_path)

    # Find mesh and armature using optional overrides or Mixamo-oriented autodetection.
    mesh_obj, armature_obj = resolve_mixamo_objects(
        mesh_name_override=args.mesh_obj.strip(),
        armature_name_override=args.skeleton_obj.strip(),
        need_armature=(args.skeleton_mode != "none"),
    )

    runtime_skeleton_mode = args.skeleton_mode
    skeleton_bone_names = []
    used_fallback = None
    if args.skeleton_mode == "weighted_centroids":
        skeleton_bone_names = get_group_matched_bone_names(mesh_obj, armature_obj)
        if not skeleton_bone_names:
            if args.skeleton_fallback == "error":
                print(
                    "ERROR: skeleton_mode=weighted_centroids found no overlapping "
                    "bone names between mesh vertex groups and armature pose bones."
                )
                sys.exit(1)
            if args.skeleton_fallback == "skip":
                runtime_skeleton_mode = "none"
                used_fallback = "skip"
            else:
                runtime_skeleton_mode = "bone_heads"
                skeleton_bone_names = select_bone_head_names(mesh_obj, armature_obj)
                used_fallback = "bone_heads"
    elif args.skeleton_mode == "bone_heads":
        skeleton_bone_names = select_bone_head_names(mesh_obj, armature_obj)

    if runtime_skeleton_mode != "none" and not skeleton_bone_names:
        print("ERROR: No skeleton bones selected for export.")
        sys.exit(1)

    print(f"\nMesh object:     {mesh_obj.name} (type={mesh_obj.type})")
    if args.mesh_obj.strip():
        print("  Mesh source:    explicit --mesh_obj override")
    else:
        print("  Mesh source:    auto-detected")
    if armature_obj is not None:
        print(f"Armature object: {armature_obj.name} (type={armature_obj.type})")
        if args.skeleton_obj.strip():
            print("  Armature source: explicit --skeleton_obj override")
        else:
            print("  Armature source: auto-detected")
        print(f"  Bones: {len(armature_obj.pose.bones)}")
        if len(armature_obj.pose.bones) <= 120:
            for bone in armature_obj.pose.bones:
                print(f"    - {bone.name}")
    print(f"Skeleton mode:   {runtime_skeleton_mode}")
    if runtime_skeleton_mode != "none":
        print(f"  Selected bones: {len(skeleton_bone_names)}")
    if used_fallback is not None:
        print(f"  Fallback used:  {used_fallback}")

    start_frame, end_frame, resolved_frame_source, frame_warning = resolve_export_frame_range(
        bpy.context.scene,
        armature_obj,
        args.frame_range_source,
    )
    num_frames = end_frame - start_frame + 1
    print(f"\nAnimation range source: {resolved_frame_source}")
    print(f"Animation range: {start_frame} to {end_frame} ({num_frames} frames)")
    if frame_warning is not None:
        print(frame_warning)
    print(f"Output directory: {output_dir}")
    print(f"Export scale: {args.scale}x")
    print(f"  mesh/            -> {mesh_dir}")
    if runtime_skeleton_mode != "none":
        print(f"  skeleton_mesh/   -> {skeleton_mesh_dir}")
    else:
        print("  skeleton_mesh/   -> skipped")
    print()

    frame_zero_weight_total = 0
    # Export each frame
    for frame in range(start_frame, end_frame + 1):
        # 0-based index for output filenames
        idx = frame - start_frame

        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()

        # Export body mesh
        mesh_path = os.path.join(mesh_dir, f"mesh_frame_{idx:04d}.obj")
        export_mesh_frame(
            mesh_obj,
            mesh_path,
            scale=args.scale,
            preserve_uv_materials=args.preserve_uv_materials,
        )

        if runtime_skeleton_mode != "none":
            skeleton_path = os.path.join(skeleton_mesh_dir, f"skeleton_frame_{idx:04d}.obj")
            if runtime_skeleton_mode == "weighted_centroids":
                positions, zero_weight_count = weighted_centroid_positions_world(
                    armature_obj,
                    mesh_obj,
                    skeleton_bone_names,
                    min_group_weight=args.min_group_weight,
                )
                frame_zero_weight_total += zero_weight_count
            else:
                positions = bone_head_positions_world(armature_obj, skeleton_bone_names)

            export_positions_as_icospheres(
                positions,
                skeleton_path,
                radius=args.icosphere_radius,
                subdivisions=args.icosphere_subdivisions,
                scale=args.scale,
            )

        print(f"Frame {idx}/{num_frames - 1} (blender frame {frame}) -> exported")

    print(f"\nExport complete!")
    print(f"  Mesh frames:     {num_frames} OBJs in {mesh_dir}")
    if runtime_skeleton_mode != "none":
        print(f"  Skeleton frames: {num_frames} OBJs in {skeleton_mesh_dir}")
        if runtime_skeleton_mode == "weighted_centroids":
            print(f"  Zero-weight bone fallbacks (sum over all frames): {frame_zero_weight_total}")
    else:
        print("  Skeleton frames: skipped")


if __name__ == "__main__":
    main()
