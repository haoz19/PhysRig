import argparse
import os
from glob import glob

import numpy as np
import open3d as o3d
import trimesh


def collect_frame_paths(skeleton_mesh_dir):
    pattern = os.path.join(skeleton_mesh_dir, "skeleton_frame_*.obj")
    return sorted(glob(pattern))


def extract_centroids(mesh_path):
    mesh = trimesh.load(mesh_path, process=False)
    parts = mesh.split(only_watertight=False)
    centroids = []
    for part in parts:
        if part.vertices.shape[0] == 0:
            continue
        centroids.append(part.vertices.mean(axis=0))
    return np.asarray(centroids, dtype=np.float32)


def write_ply(points, output_path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(output_path, pcd)


def main():
    parser = argparse.ArgumentParser(description="Generate skeleton gt_*.ply from skeleton_mesh components.")
    parser.add_argument("--input_folder", required=True, help="Dataset folder containing skeleton_mesh/")
    parser.add_argument("--output_folder", required=True, help="Output folder for skeleton/gt_*.ply")
    parser.add_argument("--num_frames", type=int, default=None, help="Number of frames to generate")
    args = parser.parse_args()

    skeleton_mesh_dir = os.path.join(args.input_folder, "skeleton_mesh")
    if not os.path.isdir(skeleton_mesh_dir):
        raise FileNotFoundError(f"Missing skeleton_mesh directory: {skeleton_mesh_dir}")

    frame_paths = collect_frame_paths(skeleton_mesh_dir)
    if not frame_paths:
        raise FileNotFoundError(f"No skeleton_frame_*.obj found in {skeleton_mesh_dir}")

    if args.num_frames is None:
        num_frames = len(frame_paths)
    else:
        num_frames = args.num_frames
        if num_frames > len(frame_paths):
            raise ValueError(
                f"Requested {num_frames} frames but only found {len(frame_paths)} skeleton meshes."
            )

    os.makedirs(args.output_folder, exist_ok=True)

    for idx in range(num_frames):
        mesh_path = frame_paths[idx]
        centroids = extract_centroids(mesh_path)
        output_path = os.path.join(args.output_folder, f"gt_{idx}.ply")
        write_ply(centroids, output_path)
        print(f"Frame {idx}: {centroids.shape[0]} centroids -> {output_path}")


if __name__ == "__main__":
    main()
