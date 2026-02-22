import os
import numpy as np
import open3d as o3d
import trimesh
import torch
import pdb
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import random
from typing import List
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree, ConvexHull
from scipy.spatial.distance import cdist


def shrink_adjacent_cuboids(cuboid_centers, cuboid_sizes, cuboid_types, vertices, grid_dx, scale_factor=0.9):

    min_size = 1.2 * grid_dx  # 最小尺寸限制

    for i, (center, size, ctype) in enumerate(zip(cuboid_centers, cuboid_sizes, cuboid_types)):
        if ctype != 'adjacent':
            continue  # 只缩放相邻 cuboid

        half_size = size / 2.0

        # 计算上下边界
        lower_bound = center - half_size
        upper_bound = center + half_size

        # 检查是否有点在 cuboid 内（只要有一个点就缩小）
        in_cuboid_mask = np.any(np.all((vertices >= lower_bound) & (vertices <= upper_bound), axis=1))

        # 如果 Cuboid 内有点，则逐步缩小
        while in_cuboid_mask:
            # 单独缩放 XYZ 三个维度
            for dim in range(3):  # dim = 0 (X), 1 (Y), 2 (Z)
                if size[dim] > min_size:
                    size[dim] *= scale_factor  # 缩小当前维度

            # 更新边界
            half_size = size / 2.0
            lower_bound = center - half_size
            upper_bound = center + half_size

            # 再次检测是否还有点在 Cuboid 内
            in_cuboid_mask = np.any(np.all((vertices >= lower_bound) & (vertices <= upper_bound), axis=1))

            # 如果 XYZ 三个维度都已经达到最小尺寸，则停止缩小
            if np.all(size <= min_size):
                size = np.maximum(size, min_size)
                break

        # 更新 Cuboid 尺寸
        cuboid_sizes[i] = size

    return cuboid_sizes

    
# 一定要确保cuboid顺序一样！！！！！！
# tjx
def cuboid_finding(vertices, grid_dx, vertices_assignment=None, mesh=None):
    # 新增：只输入点云时，直接返回点云的包围盒cuboid
    if (vertices_assignment is None or len(vertices_assignment) == 0) and mesh is None:
        cuboid_centers = []
        cuboid_sizes = []
        cuboid_types = []
       
        # 直接使用传入的点的位置，不进行缩放
        vertices_original = vertices

        dists = cdist(vertices_original, vertices_original)
        
        # 计算全局最小距离（排除对角线）
        min_dist = np.min(dists[np.nonzero(dists)])  # 非零最小距离
        global_radius = (min_dist / 2.0)*0.7  # 全局最小距离的一半

        # 特殊索引使用该点到最近点的距离，其他使用全局最小距离的一半
        special_indices = {16, 31}
        
        for i in range(vertices_original.shape[0]):
            center = vertices_original[i]
            cuboid_centers.append(center)
            # 根据索引选择不同的半径计算方式
            if i in special_indices:
                radius = min_dist / 2.0  # 直接使用距离作为半径
            else:
                # 非特殊索引：使用全局最小距离的一半
                radius = global_radius
            # 将标量radius转换为3维向量
            cuboid_sizes.append(np.array([radius, radius, radius]))
            cuboid_types.append('point_sphere')
            
        return cuboid_centers, cuboid_sizes, cuboid_types
        
    vertices_assignment = np.array(vertices_assignment, dtype=str)
    unique_parts = np.unique(vertices_assignment)

    cuboid_centers = []
    cuboid_sizes = []
    cuboid_types = []
    boundary_points_list = []
    
    remove_index = None
    
    # ===============================
    # 通过 mesh 拓扑构建 part 之间的邻接关系
    # ===============================
    adjacency_map = {}
    for face in mesh.faces:
        num_vertices = len(face)
        for i in range(num_vertices):
            v1 = face[i]
            v2 = face[(i + 1) % num_vertices]  # 环状连接
            part1 = str(vertices_assignment[v1])
            part2 = str(vertices_assignment[v2])
            if part1 != part2:
                adjacency_map.setdefault(part1, set()).add(part2)
                adjacency_map.setdefault(part2, set()).add(part1)

    # ===============================
    # 利用从 mesh 拓扑获得的相邻关系，
    # 对于每个相邻的 part 对，使用点云数据来构建 cuboid
    # ===============================
    cuboid_threshold = 2.0 * grid_dx  # 筛选边界点的距离阈值
    processed_pairs = set()
    # 对 unique_parts 进行排序，保证顺序一致
    for part_a in sorted(unique_parts):
        # 将邻接集合转换为排序后的 list
        neighbors = sorted(list(adjacency_map.get(part_a, set())))
        for part_b in neighbors:
            # 避免重复处理，保证 part_a < part_b
            if part_a >= part_b:
                continue
            pair_key = (part_a, part_b)
            if pair_key in processed_pairs:
                continue
            processed_pairs.add(pair_key)

            # 取出属于这两个 part 的点云索引
            idx_a = np.where(vertices_assignment == part_a)[0]
            idx_b = np.where(vertices_assignment == part_b)[0]
            if len(idx_a) < 2 or len(idx_b) < 2:
                continue
            v_a = vertices[idx_a]
            v_b = vertices[idx_b]

            # 根据距离阈值筛选边界点
            dist_ab = cdist(v_a, v_b)
            boundary_points_a = v_a[np.any(dist_ab <= cuboid_threshold, axis=1)]
            boundary_points_b = v_b[np.any(dist_ab <= cuboid_threshold, axis=0)]
            boundary_points = np.vstack((boundary_points_a, boundary_points_b))
            if boundary_points.shape[0] < 2:
                continue

            # 根据边界点计算 cuboid 中心和尺寸
            boundary_center = np.mean(boundary_points, axis=0)
            size_x = np.max(boundary_points[:, 0]) - np.min(boundary_points[:, 0])
            size_y = np.max(boundary_points[:, 1]) - np.min(boundary_points[:, 1])
            size_z = np.max(boundary_points[:, 2]) - np.min(boundary_points[:, 2])
            boundary_size = np.array([size_x, size_y, size_z])
            boundary_size = np.maximum(boundary_size, 1.2 * grid_dx)

            cuboid_centers.append(boundary_center)
            cuboid_sizes.append(boundary_size)
            cuboid_types.append('adjacent')
            boundary_points_list.append(boundary_points)
                
    cuboid_sizes = shrink_adjacent_cuboids(cuboid_centers, cuboid_sizes, cuboid_types, vertices, grid_dx, scale_factor=0.9)
    
    if remove_index is not None and 0 <= remove_index < len(cuboid_centers):
        del cuboid_centers[remove_index]
        del cuboid_sizes[remove_index]
        del cuboid_types[remove_index]
        del boundary_points_list[remove_index]
        print(f"Removed cuboid at index {remove_index}")
    
    if len(cuboid_centers) == 0 and len(unique_parts) == 1:
        # 只有一个 part，直接包围整个 mesh
        print("进入单part兜底逻辑")
        print("vertices.shape:", vertices.shape)
        print("vertices示例:", vertices[:5] if vertices.shape[0] > 5 else vertices)
        boundary_center = np.mean(vertices, axis=0)
        print("boundary_center:", boundary_center)
        size_x = np.max(vertices[:, 0]) - np.min(vertices[:, 0])
        size_y = np.max(vertices[:, 1]) - np.min(vertices[:, 1])
        size_z = np.max(vertices[:, 2]) - np.min(vertices[:, 2])
        boundary_size = np.array([size_x, size_y, size_z])
        boundary_size = np.maximum(boundary_size, 1.2 * grid_dx)
        cuboid_centers.append(boundary_center)
        cuboid_sizes.append(boundary_size)
        cuboid_types.append('single')
        boundary_points_list.append(vertices)
        print("兜底后 cuboid_centers:", len(cuboid_centers))
    
    return cuboid_centers, cuboid_sizes, cuboid_types, boundary_points_list

def assign_cuboid_velocity(cuboid_centers, cuboid_sizes, cuboid_types, total_frames, gt_meshes, vertices, delta_time, device, output_dir=None):
    """
    简化版本：直接基于cuboid中心位置计算速度
    
    参数:
    - output_dir: 如果提供，将保存速度信息到txt文件
    """
    frame_start = 0
    frame_end = total_frames - 1
    num_frames_velocity = frame_end - frame_start  # 即 total_frames - 1

    num_cuboids = len(cuboid_centers)
    
    # 构造一个列表，长度为 num_frames_velocity，每个元素先作为空列表，用来存放当前帧各个 cuboid 的速度
    velocity_per_frame = [[] for _ in range(num_frames_velocity)]
    
    # 对于每个 cuboid，计算其在每一帧内的速度
    # 注意：cuboid_centers[i] 对应 vertices[i]，所以 cuboid_idx 就是 vertex_idx
    for frame_idx in range(frame_start, frame_end):
        current_frame_pos = gt_meshes[frame_idx].to(device)
        next_frame_pos = gt_meshes[frame_idx + 1].to(device)
        
        for cuboid_idx in range(num_cuboids):
            p0 = current_frame_pos[cuboid_idx]
            p1 = next_frame_pos[cuboid_idx]
            vel = (p1 - p0) / delta_time
            velocity_per_frame[frame_idx - frame_start].append(vel)
    
    # 将每一帧对应的所有 cuboid 速度列表 stack 成 tensor，并包装为 nn.Parameter
    velocity_list_param = []
    for frame_velocities in velocity_per_frame:
        if len(frame_velocities) != num_cuboids:
            raise ValueError("cuboid 数量不匹配！")
        # 得到形状 [num_cuboids, 3]
        frame_tensor = torch.stack(frame_velocities, dim=0)
        frame_param = torch.nn.Parameter(frame_tensor, requires_grad=True)
        velocity_list_param.append(frame_param)
    
    return velocity_list_param