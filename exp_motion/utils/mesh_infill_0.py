import os
import sys
sys.path.append("/taiga/illinois/eng/ece/n-ahuja/haozhang/tjx/PHYSDREAMER/PhysDreamer")
import argparse
import trimesh
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from exp_motion.train.new_utils import calculate_bounding_box

# 设置随机种子，保证采样结果可复现
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


def uniform_surface_sampling(mesh, num_samples, min_distance, scale, shift):
    """
    在mesh表面进行最远点采样，返回:
    faces: 每个采样点对应的三角面索引
    bary_coords: 每个采样点对应的barycentric坐标 (u, v, w)
    
    Args:
        mesh: trimesh对象
        num_samples: 目标采样点数量
        min_distance: 最小距离约束，确保采样点间距不小于此值
        scale: 坐标变换的scale参数
        shift: 坐标变换的shift参数
    """
   
    # 1. 先进行密集采样
    dense_samples, face_indices = trimesh.sample.sample_surface(mesh, num_samples * 10)
    
    # 2. 在变换后的坐标系中进行FPS采样
    # 将密集采样点变换到目标坐标系
    transformed_samples = (dense_samples + shift) / scale
    
    first_idx = np.random.randint(len(transformed_samples))
    selected_indices = [first_idx]
    remaining_indices = list(range(len(transformed_samples)))
    remaining_indices.remove(first_idx)
    
    for _ in range(num_samples - 1):
        if not remaining_indices:
            break
            
        selected_transformed = transformed_samples[selected_indices]
        remaining_transformed = transformed_samples[remaining_indices]
        
        # 在变换后的坐标系中计算距离
        kdtree = cKDTree(selected_transformed)
        distances, _ = kdtree.query(remaining_transformed)
        
        # 应用距离约束
        valid_mask = distances >= min_distance
        if not valid_mask.any():
            print(f"警告：无法找到满足最小距离约束 {min_distance:.6f} 的点，停止采样")
            print(f"当前最近距离: {np.max(distances):.6f}")
            print(f"已选择的点数量: {len(selected_indices)}")
            break
        valid_indices = np.where(valid_mask)[0]
        distances = distances[valid_indices]
        remaining_indices = [remaining_indices[i] for i in valid_indices]
        remaining_transformed = remaining_transformed[valid_indices]
        
        if len(remaining_indices) == 0:
            break
            
        farthest_idx = np.argmax(distances)
        selected_indices.append(remaining_indices[farthest_idx])
        remaining_indices.pop(farthest_idx)
    
    # 获取选中的原始采样点和对应的面索引
    sampled_points = dense_samples[selected_indices]
    sampled_faces = face_indices[selected_indices]
    
    # 3. 计算 barycentric 坐标
    bary_coords = []
    for i, face_idx in enumerate(sampled_faces):
        v0, v1, v2 = mesh.faces[face_idx]
        tri_verts = mesh.vertices[[v0, v1, v2]]
        p = sampled_points[i]
        v0v1 = tri_verts[1] - tri_verts[0]
        v0v2 = tri_verts[2] - tri_verts[0]
        v0p = p - tri_verts[0]
        d00 = np.dot(v0v1, v0v1)
        d01 = np.dot(v0v1, v0v2)
        d11 = np.dot(v0v2, v0v2)
        d20 = np.dot(v0p, v0v1)
        d21 = np.dot(v0p, v0v2)
        denom = d00 * d11 - d01 * d01
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        bary_coords.append([u, v, w])
    
    return np.array(sampled_faces, dtype=np.int32), np.array(bary_coords, dtype=np.float32)


def reconstruct_points(mesh, faces, bary_coords):
    """
    根据 barycentric 信息重建采样点
    """
    points = []
    for face_idx, (u, v, w) in zip(faces, bary_coords):
        v0, v1, v2 = mesh.faces[face_idx]
        tri_verts = mesh.vertices[[v0, v1, v2]]
        p = u * tri_verts[0] + v * tri_verts[1] + w * tri_verts[2]
        points.append(p)
    return np.array(points)


print("Processing meshes for infilling and GT extraction...")



# 计算grid_dx（完全模拟baseline_gt_tjx_new.py中的计算逻辑）
def calculate_grid_dx_from_infilled(infilled_particles, grid_size=150):
    """
    从infilled点云计算grid_dx，完全模拟baseline_gt_tjx_new.py中的计算逻辑
    返回grid_dx, scale, shift
    """
    # 模拟baseline中的scale和shift处理
    pos_max = np.max(infilled_particles)
    pos_min = np.min(infilled_particles)
    print(f"Infilled particles range: min={pos_min:.3f}, max={pos_max:.3f}")
    
    scale = (pos_max - pos_min) * 1.8
    shift = -pos_min + (pos_max - pos_min) * 0.25
    print("scale, shift", scale, shift)
    
    # 应用scale和shift变换
    sim_xyzs = (infilled_particles + shift) / scale
    
    # 使用导入的calculate_bounding_box函数
    bbox = calculate_bounding_box(sim_xyzs)
    
    # 获取最大绝对值
    max_abs_value = max(abs(bbox['x_min']), abs(bbox['x_max']), 
                        abs(bbox['y_min']), abs(bbox['y_max']), 
                        abs(bbox['z_min']), abs(bbox['z_max']))
    
    # Haolan：grid_size和grid_lim对于模拟很重要，因为它能决定grid的大小
    x_range = max(abs(bbox['x_min']), abs(bbox['x_max'])) * 2.0  
    y_range = max(abs(bbox['y_min']), abs(bbox['y_max'])) * 2.0  # y轴保持1.5倍
    z_range = max(abs(bbox['z_min']), abs(bbox['z_max'])) * 2.0  # z轴保持1.5倍
    grid_lim = max(x_range, y_range, z_range)  # 取最大值作为网格边界
    grid_dx = grid_lim / grid_size
    print("grid_dx:",grid_dx)
    print(f"X range: {x_range:.3f}m, Y range: {y_range:.3f}m, Z range: {z_range:.3f}m")
    
    return grid_dx, scale, shift


# 处理所有帧：都需要体积填充
for i in range(0,1):
    print(f"Processing frame {i} with volume filling...")
    mesh_path = os.path.join(args.input_folder, "mesh", f"mesh_frame_{i:04d}.obj")
    mesh = trimesh.load(mesh_path, process=False)
    vertices = mesh.vertices
    # 目标表面点数量：原逻辑为顶点数的 3 倍
    target_surface = len(vertices)
    
    # 使用 trimesh.sample.sample_surface_even 进行均匀表面采样
    surface_points, _ = trimesh.sample.sample_surface_even(mesh, target_surface)
    
    print(f"原始顶点数: {len(vertices)}, 表面采样点数: {len(surface_points)}")
    
    volume_points = fill_mesh_volume_adaptive(mesh, args.resolution, args.samples_per_voxel)
    infilled_particles = np.vstack((surface_points, volume_points))
    print(f"Total infilled particles for frame {i}: {infilled_particles.shape[0]}")
    pcd_infilled = o3d.geometry.PointCloud()
    pcd_infilled.points = o3d.utility.Vector3dVector(infilled_particles)
    output_file_infilled = os.path.join(args.output_folder_infilled, f"infilled_{i}.ply")
    o3d.io.write_point_cloud(output_file_infilled, pcd_infilled)
    print(f"Infilled point cloud saved to: {output_file_infilled}")

# 使用infilled点云计算grid_dx，同时获取scale和shift
grid_dx, scale, shift = calculate_grid_dx_from_infilled(infilled_particles)

# FPS采样 + 获取barycentric信息
# 使用grid_dx计算最小距离约束
min_distance_threshold = grid_dx * 3.0
print(f"设置骨骼点采样最小距离约束: {min_distance_threshold:.6f} (grid_dx * 3.0)")

# 第一帧：采样并保存 barycentric 信息
first_frame = 0
skeleton_mesh_path = os.path.join(args.input_folder, "skeleton_mesh", f"skeleton_frame_{first_frame:04d}.obj")
skeleton_mesh = trimesh.load(skeleton_mesh_path, process=False)
num_gt_samples = len(skeleton_mesh.vertices)#调整cuboid的数量
print("cuboid数量:",num_gt_samples)

faces, bary_coords = uniform_surface_sampling(skeleton_mesh, num_gt_samples, 
                                            min_distance=min_distance_threshold,
                                            scale=scale, shift=shift)

# 保存 barycentric 信息
np.savez(os.path.join(args.output_folder_gt, 'barycentric_info.npz'), faces=faces, bary=bary_coords)
print(f"Frame {first_frame}: FPS采样完成，保存 barycentric 信息")


# 后续帧：加载 barycentric 信息，重建点云
for i in range(0, args.num_frames):
    skeleton_mesh_path = os.path.join(args.input_folder, "skeleton_mesh", f"skeleton_frame_{i:04d}.obj")
    skeleton_mesh = trimesh.load(skeleton_mesh_path, process=False)
    
    # 读取 barycentric 信息
    data = np.load(os.path.join(args.output_folder_gt, 'barycentric_info.npz'))
    faces = data['faces']
    bary_coords = data['bary']
    
    # 重建采样点
    gt_surface_points = reconstruct_points(skeleton_mesh, faces, bary_coords)
    
    # 保存点云
    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(gt_surface_points)
    output_file_gt = os.path.join(args.output_folder_gt, f"gt_{i}.ply")
    o3d.io.write_point_cloud(output_file_gt, pcd_gt)
    
    print(f"Frame {i}: 通过 barycentric 信息重建完成")

# ===== 第一帧骨骼点中心调整，并将位移应用到所有帧 =====
print("\n=== 第一帧骨骼点中心调整 ===")

# 读取第一帧的骨骼点
pcd_gt_0 = o3d.io.read_point_cloud(os.path.join(args.output_folder_gt, "gt_0.ply"))
gt_points_0_original = np.asarray(pcd_gt_0.points)

# 读取第一帧的 infilled_particles
infilled_path_0 = os.path.join(args.output_folder_infilled, f"infilled_0.ply")
pcd_infilled_0 = o3d.io.read_point_cloud(infilled_path_0)
infilled_particles_0 = np.asarray(pcd_infilled_0.points)

# 构建第一帧 infilled_particles 的 KDTree（在变换后的坐标系中）
transformed_infilled_0 = (infilled_particles_0 + shift) / scale
infilled_tree_0 = cKDTree(transformed_infilled_0)

# 设置调整半径（min_distance_threshold 的一半）
adjustment_radius = min_distance_threshold * 0.5
print(f"调整半径: {adjustment_radius:.6f} (min_distance_threshold * 0.5)")

# 第一帧骨骼点变换到目标坐标系
transformed_skeleton_0 = (gt_points_0_original + shift) / scale

# 调整到 infilled_particles 中心
adjusted_skeleton_0 = []
for j, sk_point in enumerate(transformed_skeleton_0):
    indices = infilled_tree_0.query_ball_point(sk_point, adjustment_radius)
    if len(indices) > 0:
        nearby_infilled = transformed_infilled_0[indices]
        center = np.mean(nearby_infilled, axis=0)
        adjusted_skeleton_0.append(center)
    else:
        adjusted_skeleton_0.append(sk_point)

adjusted_skeleton_0 = np.array(adjusted_skeleton_0)

# 记录每个点的位移向量（方向和距离）
displacement_vectors = adjusted_skeleton_0 - transformed_skeleton_0  # (N, 3)
displacement_distances = np.linalg.norm(displacement_vectors, axis=1)  # (N,)
print(f"位移向量计算完成，平均位移距离: {np.mean(displacement_distances):.6f}")

# 对所有帧应用相同的位移向量
print("\n=== 应用位移向量到所有帧 ===")
for i in range(0, args.num_frames):
    # 读取已保存的骨骼点
    gt_path = os.path.join(args.output_folder_gt, f"gt_{i}.ply")
    pcd_gt = o3d.io.read_point_cloud(gt_path)
    gt_points_original = np.asarray(pcd_gt.points)
    
    # 变换到目标坐标系
    transformed_skeleton = (gt_points_original + shift) / scale
    
    # 应用第一帧计算的位移向量（相同方向、相同距离）
    adjusted_skeleton = transformed_skeleton + displacement_vectors
    
    # 反变换回原始坐标系
    gt_surface_points = adjusted_skeleton * scale - shift
    
    # 重新保存点云
    pcd_gt.points = o3d.utility.Vector3dVector(gt_surface_points)
    o3d.io.write_point_cloud(gt_path, pcd_gt)
    
    print(f"Frame {i}: 应用位移向量完成")

# 通过读取保存的gt_0.ply文件来检测点之间的最小距离（在变换后的坐标系中）
print("\n=== 距离检测 ===")
gt_0_path = os.path.join(args.output_folder_gt, "gt_0.ply")
if os.path.exists(gt_0_path):
    pcd_gt_0 = o3d.io.read_point_cloud(gt_0_path)
    gt_vertices = np.asarray(pcd_gt_0.points)
    
    if len(gt_vertices) > 1:
        # 应用与 baseline_gt_tjx_new.py 相同的预处理变换
        preprocessed_vertices = (gt_vertices + shift) / scale

        # ✅ 计算每个点到最近邻点的距离
        from scipy.spatial import cKDTree
        tree = cKDTree(preprocessed_vertices)
        dists, _ = tree.query(preprocessed_vertices, k=2)
        nearest_dists = dists[:, 1]  # 每个点的最近邻距离

        # ✅ 输出所有点的最近邻距离（逐行）
        print("\n每个点到最近邻点的距离列表:")
        for i, dist in enumerate(nearest_dists):
            print(f"点 {i:4d}: {dist:.6f}")

        # 可选：打印 grid_dx（如果定义过）
        if 'grid_dx' in locals():
            print(f"grid_dx: {grid_dx:.6f}")
    else:
        print("警告：gt_0.ply中的点云数量不足，无法计算最小距离")
else:
    print(f"错误：找不到文件 {gt_0_path}")

print("All frames processed and saved.")



