"""
Retarget OBJ mesh sequence onto textured FBX mesh and export animated GLB.

This rewrite intentionally uses topology-consistent OBJ sequences (mesh_frame_*.obj)
instead of point-cloud PLY data.
"""

import argparse
import math
import os
import re
import sys
from collections import Counter

import bpy
import numpy as np


def parse_args():
    argv = sys.argv
    if "--" not in argv:
        print("ERROR: No arguments provided after '--'")
        sys.exit(1)
    argv = argv[argv.index("--") + 1 :]

    parser = argparse.ArgumentParser(description="Retarget OBJ sequence to textured FBX mesh.")
    parser.add_argument("--source_fbx", required=True, help="Path to original textured FBX.")
    parser.add_argument("--obj_seq_dir", required=True, help="Directory containing mesh_frame_*.obj.")
    parser.add_argument("--output_glb", required=True, help="Path to output GLB file.")
    parser.add_argument("--mesh_obj", default="", help="Optional mesh object name override.")
    parser.add_argument("--fps", type=int, default=30, help="Output animation FPS.")
    parser.add_argument(
        "--obj_scale",
        default="auto",
        help="OBJ coordinate divisor (number) or 'auto' (tries 1 and 15).",
    )
    parser.add_argument(
        "--coord_space",
        choices=("auto", "world", "local"),
        default="auto",
        help="Interpret OBJ vertices as world/local space before applying to mesh.",
    )
    parser.add_argument("--save_blend", action="store_true", help="Also save a .blend next to output GLB.")
    return parser.parse_args(argv)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def import_fbx(path):
    print(f"Importing FBX: {path}")
    bpy.ops.import_scene.fbx(filepath=path)


def find_mesh_object(name_override):
    if name_override:
        obj = bpy.data.objects.get(name_override)
        if obj is None:
            print(f"ERROR: Mesh object '{name_override}' not found.")
            sys.exit(1)
        if obj.type != "MESH":
            print(f"ERROR: Object '{name_override}' is type '{obj.type}', not MESH.")
            sys.exit(1)
        return obj

    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        print("ERROR: No mesh object found in source FBX.")
        sys.exit(1)
    meshes.sort(key=lambda o: (len(o.data.vertices), o.name))
    return meshes[-1]


def discover_obj_frames(obj_seq_dir):
    if not os.path.isdir(obj_seq_dir):
        print(f"ERROR: obj_seq_dir does not exist: {obj_seq_dir}")
        sys.exit(1)

    def frame_id(path):
        name = os.path.basename(path)
        m = re.search(r"mesh_frame_(\d+)", name)
        if m:
            return int(m.group(1))
        nums = re.findall(r"(\d+)", name)
        return int(nums[-1]) if nums else -1

    files = [
        os.path.join(obj_seq_dir, f)
        for f in os.listdir(obj_seq_dir)
        if f.lower().endswith(".obj") and f.lower().startswith("mesh_frame_")
    ]
    files = [f for f in files if frame_id(f) >= 0]
    files.sort(key=lambda p: (frame_id(p), p))

    if not files:
        print(f"ERROR: No mesh_frame_*.obj found in {obj_seq_dir}")
        sys.exit(1)
    return files


def parse_obj_vertices(path, expected_count=None):
    verts = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) < 4:
                    raise RuntimeError(f"Malformed vertex line in {path}")
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
    arr = np.array(verts, dtype=np.float64)
    if expected_count is not None and len(arr) != expected_count:
        raise RuntimeError(
            f"Vertex count mismatch in {path}: expected {expected_count}, got {len(arr)}"
        )
    return arr


def parse_obj_vertices_and_triangles(path):
    verts = []
    tris = []

    def parse_index(token, nverts):
        head = token.split("/")[0]
        if head == "":
            raise RuntimeError(f"Malformed face token '{token}' in {path}")
        idx = int(head)
        if idx < 0:
            idx = nverts + idx + 1
        return idx - 1

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) < 4:
                    raise RuntimeError(f"Malformed vertex line in {path}")
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("f "):
                parts = line.split()[1:]
                if len(parts) < 3:
                    continue
                face = [parse_index(tok, len(verts)) for tok in parts]
                for i in range(1, len(face) - 1):
                    tris.append((face[0], face[i], face[i + 1]))

    return np.array(verts, dtype=np.float64), tris


def mesh_triangles(mesh_obj):
    mesh = mesh_obj.data
    mesh.calc_loop_triangles()
    return [tuple(t.vertices) for t in mesh.loop_triangles]


def topology_check(mesh_tris, obj_tris):
    if len(mesh_tris) != len(obj_tris):
        return False, f"triangle count mismatch: FBX={len(mesh_tris)} OBJ={len(obj_tris)}"

    if mesh_tris == obj_tris:
        return True, "exact triangle order match"

    mesh_norm = Counter(tuple(sorted(t)) for t in mesh_tris)
    obj_norm = Counter(tuple(sorted(t)) for t in obj_tris)
    if mesh_norm == obj_norm:
        return True, "triangle set matches (order/winding differ)"

    return False, "triangle topology mismatch"


def bbox_diag_and_center(points):
    pmin = points.min(axis=0)
    pmax = points.max(axis=0)
    diag = float(np.linalg.norm(pmax - pmin))
    center = 0.5 * (pmin + pmax)
    return diag, center


def transform_world_to_local(points, mesh_obj):
    inv_m = np.array(mesh_obj.matrix_world.inverted(), dtype=np.float64)
    hom = np.ones((points.shape[0], 4), dtype=np.float64)
    hom[:, :3] = points
    out = (inv_m @ hom.T).T
    return out[:, :3]


def convert_obj_points(points_obj, mesh_obj, scale_divisor, coord_mode):
    pts = points_obj / scale_divisor
    if coord_mode == "world":
        return transform_world_to_local(pts, mesh_obj)
    return pts


def resolve_scale_and_space(first_obj_verts, mesh_base_local, mesh_obj, scale_arg, space_arg):
    if str(scale_arg).strip().lower() == "auto":
        scales = [1.0, 15.0]
    else:
        try:
            val = float(scale_arg)
        except ValueError as exc:
            raise RuntimeError(f"Invalid --obj_scale '{scale_arg}'") from exc
        if val <= 0:
            raise RuntimeError("--obj_scale must be > 0")
        scales = [val]

    if space_arg == "auto":
        spaces = ["world", "local"]
    else:
        spaces = [space_arg]

    mesh_diag, mesh_center = bbox_diag_and_center(mesh_base_local)
    eps = 1e-12

    best = None
    rows = []
    for s in scales:
        for mode in spaces:
            cand = convert_obj_points(first_obj_verts, mesh_obj, s, mode)
            diag, center = bbox_diag_and_center(cand)
            diag_score = abs(np.log((diag + eps) / (mesh_diag + eps)))
            center_score = float(np.linalg.norm(center - mesh_center) / (mesh_diag + eps))
            score = diag_score + 0.25 * center_score
            rows.append((mode, s, score, diag_score, center_score))
            if best is None or score < best[2]:
                best = (mode, s, score)

    for mode, s, score, dsc, csc in rows:
        print(
            f"  candidate mode={mode:5s} scale={s:8.4f} score={score:8.5f} "
            f"(diag={dsc:8.5f}, center={csc:8.5f})"
        )

    if best is None:
        raise RuntimeError("Could not resolve OBJ scale/space.")

    return best[0], float(best[1])


def remove_shape_keys(mesh_obj):
    if mesh_obj.data.shape_keys is None:
        return
    bpy.ops.object.select_all(action="DESELECT")
    mesh_obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_obj
    while mesh_obj.data.shape_keys and len(mesh_obj.data.shape_keys.key_blocks) > 0:
        mesh_obj.active_shape_key_index = len(mesh_obj.data.shape_keys.key_blocks) - 1
        bpy.ops.object.shape_key_remove(all=False)


def build_shape_key_action(sk_data, key_info, num_frames):
    ad = sk_data.animation_data_create()
    if ad.action is not None:
        ad.action = None
    while ad.nla_tracks:
        ad.nla_tracks.remove(ad.nla_tracks[0])

    action = bpy.data.actions.new(name="RetargetShapeKeys")
    ad.action = action

    for key_name, active_frame in key_info:
        path = f'key_blocks["{key_name}"].value'
        fcu = action.fcurves.new(data_path=path)
        fcu.keyframe_points.add(num_frames)
        for idx in range(num_frames):
            frame = idx + 1
            val = 1.0 if frame == active_frame else 0.0
            fcu.keyframe_points[idx].co = (frame, val)
        for kp in fcu.keyframe_points:
            kp.interpolation = "LINEAR"

    track = ad.nla_tracks.new()
    track.name = "RetargetShapeKeysTrack"
    strip = track.strips.new(action.name, start=1, action=action)
    strip.action_frame_start = 1
    strip.action_frame_end = num_frames

    return action


def export_glb(output_glb, mesh_obj):
    out_dir = os.path.dirname(output_glb)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    bpy.ops.object.select_all(action="DESELECT")
    mesh_obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_obj

    op = bpy.ops.export_scene.gltf
    prop_keys = set(op.get_rna_type().properties.keys())
    kwargs = {"filepath": output_glb}

    if "export_format" in prop_keys:
        kwargs["export_format"] = "GLB"
    if "use_selection" in prop_keys:
        kwargs["use_selection"] = True
    if "export_animations" in prop_keys:
        kwargs["export_animations"] = True
    if "export_animation_mode" in prop_keys:
        kwargs["export_animation_mode"] = "NLA_TRACKS"
    if "export_frame_range" in prop_keys:
        kwargs["export_frame_range"] = True
    if "export_texcoords" in prop_keys:
        kwargs["export_texcoords"] = True
    if "export_normals" in prop_keys:
        kwargs["export_normals"] = True
    if "export_materials" in prop_keys:
        kwargs["export_materials"] = "EXPORT"
    if "export_morph" in prop_keys:
        kwargs["export_morph"] = True
    if "export_morph_animation" in prop_keys:
        kwargs["export_morph_animation"] = True
    if "export_skins" in prop_keys:
        kwargs["export_skins"] = False

    op(**kwargs)


def rotate_for_export_x90(mesh_obj):
    prev_mode = mesh_obj.rotation_mode
    prev_rot = tuple(mesh_obj.rotation_euler)
    mesh_obj.rotation_mode = "XYZ"
    mesh_obj.rotation_euler = (
        prev_rot[0] + (math.pi * 0.5),
        prev_rot[1],
        prev_rot[2],
    )
    bpy.context.view_layer.update()
    return prev_mode, prev_rot


def restore_rotation(mesh_obj, prev_mode, prev_rot):
    mesh_obj.rotation_mode = "XYZ"
    mesh_obj.rotation_euler = (prev_rot[0], prev_rot[1], prev_rot[2])
    mesh_obj.rotation_mode = prev_mode
    bpy.context.view_layer.update()


def save_debug_blend(output_glb):
    blend_path = os.path.splitext(output_glb)[0] + ".blend"
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    print(f"Saved debug blend: {blend_path}")


def main():
    args = parse_args()

    if args.fps <= 0:
        print("ERROR: --fps must be > 0")
        sys.exit(1)
    if not os.path.isfile(args.source_fbx):
        print(f"ERROR: source_fbx not found: {args.source_fbx}")
        sys.exit(1)

    obj_frames = discover_obj_frames(args.obj_seq_dir)
    print(f"Discovered {len(obj_frames)} OBJ frames in {args.obj_seq_dir}")

    clear_scene()
    import_fbx(args.source_fbx)
    mesh_obj = find_mesh_object(args.mesh_obj.strip())
    print(f"Target mesh: {mesh_obj.name} (verts={len(mesh_obj.data.vertices)})")

    mesh_base_local = np.array([[v.co.x, v.co.y, v.co.z] for v in mesh_obj.data.vertices], dtype=np.float64)

    obj0_verts, obj0_tris = parse_obj_vertices_and_triangles(obj_frames[0])
    if len(obj0_verts) != len(mesh_base_local):
        print(
            f"ERROR: Vertex count mismatch between FBX mesh ({len(mesh_base_local)}) "
            f"and first OBJ frame ({len(obj0_verts)})."
        )
        sys.exit(1)

    fbx_tris = mesh_triangles(mesh_obj)
    ok, reason = topology_check(fbx_tris, obj0_tris)
    print(f"Topology check: {reason}")
    if not ok:
        print("ERROR: OBJ sequence topology is incompatible with source FBX mesh.")
        sys.exit(1)

    try:
        resolved_space, resolved_scale = resolve_scale_and_space(
            obj0_verts,
            mesh_base_local,
            mesh_obj,
            args.obj_scale,
            args.coord_space,
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    print(f"Using obj_scale divisor: {resolved_scale:.6f} (arg={args.obj_scale})")
    print(f"Using coord_space: {resolved_space} (arg={args.coord_space})")

    # Set base mesh vertices to first frame to avoid T-pose baseline artifacts.
    frame0_local = convert_obj_points(obj0_verts, mesh_obj, resolved_scale, resolved_space)
    for i, co in enumerate(frame0_local):
        mesh_obj.data.vertices[i].co = (float(co[0]), float(co[1]), float(co[2]))
    mesh_obj.data.update()

    print("Preparing shape keys...")
    remove_shape_keys(mesh_obj)
    bpy.ops.object.select_all(action="DESELECT")
    mesh_obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_obj
    _ = mesh_obj.shape_key_add(name="Basis", from_mix=False)
    sk_data = mesh_obj.data.shape_keys
    sk_data.use_relative = True

    key_info = []
    for frame_idx, path in enumerate(obj_frames):
        if frame_idx == 0:
            continue
        verts = parse_obj_vertices(path, expected_count=len(mesh_base_local))
        local = convert_obj_points(verts, mesh_obj, resolved_scale, resolved_space)

        key = mesh_obj.shape_key_add(name=f"Frame_{frame_idx:04d}", from_mix=False)
        for i, co in enumerate(local):
            key.data[i].co = (float(co[0]), float(co[1]), float(co[2]))
        key.value = 0.0
        key.mute = False
        key_info.append((key.name, frame_idx + 1))

        if frame_idx % 10 == 0 or frame_idx == len(obj_frames) - 1:
            print(f"Created shape key {frame_idx + 1}/{len(obj_frames)}")

    scene = bpy.context.scene
    scene.render.fps = args.fps
    scene.frame_start = 1
    scene.frame_end = len(obj_frames)

    if key_info:
        action = build_shape_key_action(sk_data, key_info, len(obj_frames))
        fcurve_count = len(sk_data.animation_data.action.fcurves) if sk_data.animation_data and sk_data.animation_data.action else 0
        print(f"Shape-key fcurves: {fcurve_count} (action={action.name})")
    else:
        print("Single-frame sequence: no animation curves created.")

    print(f"Exporting GLB: {args.output_glb}")
    scene.frame_set(scene.frame_start)
    prev_mode, prev_rot = rotate_for_export_x90(mesh_obj)
    try:
        export_glb(args.output_glb, mesh_obj)
    finally:
        restore_rotation(mesh_obj, prev_mode, prev_rot)

    if args.save_blend:
        save_debug_blend(args.output_glb)

    print("Done.")


if __name__ == "__main__":
    main()
