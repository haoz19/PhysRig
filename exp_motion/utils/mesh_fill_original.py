import os
import argparse
import trimesh
import numpy as np
import open3d as o3d

np.random.seed(42)

'''
mesh如果保存normal和uv会导致点分裂，导致有重复点，而trimesh或者open3d会滤掉，导致gt和mesh点不匹配
'''

# 解析输入参数
parser = argparse.ArgumentParser(description="Mesh infilling script")
parser.add_argument("--input_folder", required=True, help="Path to input mesh folder")
parser.add_argument("--output_folder_infilled", required=True, help="Path to output infilled folder")
parser.add_argument("--output_folder_gt", required=True, help="Path to output GT folder")
parser.add_argument("--num_frames", type=int, required=True, help="Number of frames to process")
parser.add_argument("--num_surface_samples", type=int, required=True, help="Number of surface samples")
parser.add_argument("--resolution", type=int, required=True, help="Voxel grid resolution")
parser.add_argument("--samples_per_voxel", type=int, required=True, help="Samples per voxel")

args = parser.parse_args()

os.makedirs(args.output_folder_infilled, exist_ok=True)
os.makedirs(args.output_folder_gt, exist_ok=True)

def get_adaptive_voxel_size(mesh, resolution):
    bbox_min, bbox_max = mesh.bounds
    bbox_extent = bbox_max - bbox_min
    return bbox_extent / resolution

def sample_inside_voxel(voxel_center, voxel_size, num_samples):
    half_size = voxel_size / 2.0
    samples = np.random.uniform(-half_size, half_size, (num_samples, 3))
    return samples + voxel_center

def fill_mesh_volume_adaptive(mesh, resolution, samples_per_voxel):
    voxel_size = get_adaptive_voxel_size(mesh, resolution)
    bbox_min, bbox_max = mesh.bounds

    grid_x, grid_y, grid_z = np.mgrid[
        bbox_min[0]:bbox_max[0]:voxel_size[0],
        bbox_min[1]:bbox_max[1]:voxel_size[1],
        bbox_min[2]:bbox_max[2]:voxel_size[2]
    ]

    grid_points = np.vstack((grid_x.flatten(), grid_y.flatten(), grid_z.flatten())).T
    inside = mesh.contains(grid_points)
    internal_points = grid_points[inside]

    filled_points = []
    for point in internal_points:
        sampled_points = sample_inside_voxel(point, voxel_size, samples_per_voxel)
        inside_samples = mesh.contains(sampled_points + 1e-5)
        filled_points.extend(sampled_points[inside_samples])

    return np.array(filled_points)

print("Processing meshes for infilling and GT extraction...")

for i in range(args.num_frames):
    mesh_path = os.path.join(args.input_folder, f"mesh_frame_{i+1:04d}.obj")
    mesh = trimesh.load(mesh_path, process=False)

    vertices = mesh.vertices
    
    surface_points = trimesh.sample.sample_surface(mesh, args.num_surface_samples)[0]

    if i == 0:
        volume_points = fill_mesh_volume_adaptive(mesh, args.resolution, args.samples_per_voxel)
        infilled_particles = np.vstack((surface_points, volume_points, vertices))
        
        print(f"Frame {i+1}: Total infilled particles: {infilled_particles.shape[0]}")
        pcd_infilled = o3d.geometry.PointCloud()
        pcd_infilled.points = o3d.utility.Vector3dVector(infilled_particles)
        output_file_infilled = os.path.join(args.output_folder_infilled, f"infilled_{i}.ply")
        o3d.io.write_point_cloud(output_file_infilled, pcd_infilled)
        # print(f"Infilled point cloud saved to: {output_file_infilled}")

    print(f"Frame {i+1}: Saving GT point cloud with {len(vertices)} vertices")
    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(vertices)
    output_file_gt = os.path.join(args.output_folder_gt, f"gt_{i+1}.ply")
    o3d.io.write_point_cloud(output_file_gt, pcd_gt)
    # print(f"GT point cloud saved to: {output_file_gt}")

print("All frames processed and saved.")