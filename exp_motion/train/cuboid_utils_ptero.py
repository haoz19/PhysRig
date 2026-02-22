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

def cuboid_finding(vertices, vertices_assignment, grid_dx, mesh): 
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
    cuboid_threshold = 1.2 * grid_dx  # 筛选边界点的距离阈值
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
    
    return cuboid_centers, cuboid_sizes, cuboid_types, boundary_points_list


def assign_cuboid_velocity(cuboid_centers, cuboid_sizes, cuboid_types, boundary_points_list,
                           total_frames, gt_meshes, vertices, vertices_assignment, grid_dx, delta_time, 
                           device,):
    """
    为每一帧的所有 Cuboid 分配速度，返回的是一个列表，每个元素为当前帧所有 cuboid 的速度 tensor，
    其形状为 [num_cuboids, 3]，并封装为 nn.Parameter，便于后续训练时独立更新。
    
    对于：
    - endpoint 类型的 cuboid：利用其内部点的平均位移计算速度；
    - adjacent 类型的 cuboid：利用边界点云的平均位移计算速度。
    """
    # 定义帧索引范围：我们计算的是相邻帧之间的速度，因此有效帧数为 total_frames - 1
    frame_start = 0
    frame_end = total_frames - 1
    num_frames_velocity = frame_end - frame_start  # 即 total_frames - 1

    # 将 vertices 转为 tensor，方便后续比较
    vertices_tensor = torch.tensor(vertices, dtype=torch.float32, device=device)
    num_cuboids = len(cuboid_centers)
    
    # 构造一个列表，长度为 num_frames_velocity，每个元素先作为空列表，用来存放当前帧各个 cuboid 的速度
    velocity_per_frame = [[] for _ in range(num_frames_velocity)]
    
    # 对于每个 cuboid，计算其在每一帧内的速度，并添加到对应帧的列表中
    for cuboid_idx, (cuboid_center, cuboid_size, cuboid_type) in enumerate(zip(cuboid_centers, cuboid_sizes, cuboid_types)):
        if cuboid_type == 'adjacent':
            # 对于相邻 Cuboid：使用 boundary_points_list 给定的边界点云计算速度
            boundary_points = torch.tensor(boundary_points_list[cuboid_idx], dtype=torch.float32, device=device)
            for frame_idx in range(frame_start, frame_end):
                current_frame_pos = gt_meshes[frame_idx].to(device)
                next_frame_pos = gt_meshes[frame_idx + 1].to(device)
                
                # 这里采用严格比较（注意浮点数比较可能需要容忍误差，但此处保持原样）
                part_indices = torch.where((vertices_tensor.unsqueeze(1) == boundary_points).all(dim=2))[0]
                
                if part_indices.numel() == 0:
                    disp = torch.zeros((1, 3), device=device)
                else:
                    part_vertices_current = current_frame_pos[part_indices]
                    part_vertices_next = next_frame_pos[part_indices]
                    disp = (part_vertices_next.mean(dim=0) - part_vertices_current.mean(dim=0)).unsqueeze(0)
                vel = (disp / delta_time).squeeze(0)
                velocity_per_frame[frame_idx - frame_start].append(vel)
        else:
            raise ValueError(f"未知 Cuboid 类型: {cuboid_type}")
    
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