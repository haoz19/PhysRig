#!/usr/bin/env python3
"""
Generate cuboid sphere meshes from skeleton point clouds.
Each skeleton point becomes a sphere mesh centered at that point.
"""

import argparse
import os
import numpy as np
import open3d as o3d
import trimesh
from glob import glob


def create_sphere_mesh(center, radius, subdivisions=2):
    """
    Create a sphere mesh using trimesh.
    
    Args:
        center: [x, y, z] center position
        radius: sphere radius
        subdivisions: number of subdivisions (higher = smoother sphere)
    
    Returns:
        trimesh.Trimesh object
    """
    sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    sphere.apply_translation(center)
    return sphere


def process_skeleton_frame(skeleton_ply_path, cuboid_sizes, output_obj_path, subdivisions=2):
    """
    Process a single skeleton frame and generate sphere meshes.
    
    Args:
        skeleton_ply_path: path to skeleton PLY file
        cuboid_sizes: numpy array of shape (N, 3) with cuboid sizes
        output_obj_path: output OBJ file path
        subdivisions: sphere subdivision level
    """
    # Load skeleton points
    pcd = o3d.io.read_point_cloud(skeleton_ply_path)
    points = np.asarray(pcd.points)
    
    if len(points) != len(cuboid_sizes):
        raise ValueError(
            f"Mismatch: {len(points)} skeleton points but {len(cuboid_sizes)} cuboid sizes"
        )
    
    # Create sphere mesh for each point
    meshes = []
    for i, (point, size) in enumerate(zip(points, cuboid_sizes)):
        # Size is [rx, ry, rz], we use the average or first dimension as radius
        radius = size[0]  # Assuming spherical cuboids (all dimensions equal)
        sphere = create_sphere_mesh(point, radius, subdivisions=subdivisions)
        meshes.append(sphere)
    
    # Combine all spheres into one mesh
    combined_mesh = trimesh.util.concatenate(meshes)
    
    # Save as OBJ
    combined_mesh.export(output_obj_path)
    print(f"Saved {len(points)} spheres to {output_obj_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate cuboid sphere meshes from skeleton point clouds"
    )
    parser.add_argument(
        "--skeleton_dir",
        required=True,
        help="Directory containing skeleton PLY files (e.g., output_step_XXXX_frame_YYYY_part_0.ply)",
    )
    parser.add_argument(
        "--cuboid_sizes_file",
        required=True,
        help="NPY file containing cuboid sizes array of shape (N, 3)",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for cuboid OBJ files",
    )
    parser.add_argument(
        "--subdivisions",
        type=int,
        default=2,
        help="Sphere subdivision level (0-4, higher = smoother but larger files)",
    )
    
    args = parser.parse_args()
    
    # Load cuboid sizes
    if not os.path.exists(args.cuboid_sizes_file):
        raise FileNotFoundError(f"Cuboid sizes file not found: {args.cuboid_sizes_file}")
    
    cuboid_sizes = np.load(args.cuboid_sizes_file)
    print(f"Loaded {len(cuboid_sizes)} cuboid sizes from {args.cuboid_sizes_file}")
    print(f"Cuboid size range: min={cuboid_sizes.min():.4f}, max={cuboid_sizes.max():.4f}")
    
    # Find all skeleton PLY files
    skeleton_pattern = os.path.join(args.skeleton_dir, "*.ply")
    skeleton_files = sorted(glob(skeleton_pattern))
    
    if not skeleton_files:
        raise FileNotFoundError(f"No skeleton PLY files found in {args.skeleton_dir}")
    
    print(f"Found {len(skeleton_files)} skeleton frames")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each frame
    for skeleton_file in skeleton_files:
        # Extract frame identifier from filename
        # e.g., output_step_0049_frame_0001_part_0.ply -> cuboid_step_0049_frame_0001.obj
        basename = os.path.basename(skeleton_file)
        base_name_no_ext = os.path.splitext(basename)[0]
        
        # Replace "output" with "cuboid" and change extension to .obj
        obj_name = base_name_no_ext.replace("output", "cuboid") + ".obj"
        output_obj_path = os.path.join(args.output_dir, obj_name)
        
        try:
            process_skeleton_frame(
                skeleton_file,
                cuboid_sizes,
                output_obj_path,
                subdivisions=args.subdivisions
            )
        except Exception as e:
            print(f"Error processing {skeleton_file}: {e}")
            continue
    
    print(f"\n✅ Generated {len(skeleton_files)} cuboid meshes in {args.output_dir}")


if __name__ == "__main__":
    main()
