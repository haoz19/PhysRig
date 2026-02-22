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

def scale_endpoint_cuboids(cuboid_centers, cuboid_sizes, cuboid_types, scale_endpoint):

    adjusted_sizes = []  # 存储调整后的尺寸

    for i, (center, size, ctype) in enumerate(zip(cuboid_centers, cuboid_sizes, cuboid_types)):
        # 只缩放 endpoint 类型，adjacent 保持不变
        if ctype == 'endpoint':
            new_size = size * scale_endpoint  # 缩放端点 Cuboid
        else:
            new_size = size  # 其他 Cuboid 不缩放

        # 重置重叠标志
        is_overlapping = False

        # 检查是否与其他 Cuboid 重叠
        for j, (other_center, other_size) in enumerate(zip(cuboid_centers, cuboid_sizes)):
            if i == j:
                continue  # 跳过自身

            # 默认假设没有重叠
            is_overlapping = True

            # 检查每个维度是否重叠
            for d in range(3):  # x, y, z 维度
                distance = abs(center[d] - other_center[d])  # 中心点距离
                threshold = (new_size[d] + other_size[d]) / 2  # 尺寸边界之和的一半

                if distance > threshold:  # 某个维度上不重叠
                    is_overlapping = False
                    break

            if is_overlapping:  # 如果在所有维度上都重叠，停止检查
                break

        # 如果发生重叠，不更新尺寸
        if is_overlapping:
            # print(f"⚠️ Cuboid {i} 缩放导致重叠，保持原尺寸")
            adjusted_sizes.append(size)
        else:
            # print(f"✅ Cuboid {i} 缩放成功: {size} -> {new_size}")
            adjusted_sizes.append(new_size)

    return adjusted_sizes

def pca_split_parts(vertices, vertices_assignment, part_ids):
    """
    使用 PCA 对指定的 part 进行分割。
    """
    for part in part_ids:
        if str(part) not in np.unique(vertices_assignment):
            continue
        
        part_indices = np.where(vertices_assignment == str(part))[0]
        part_vertices = vertices[part_indices]
        if part_vertices.shape[0] < 2:
            continue

        center = np.mean(part_vertices, axis=0)
        centered_vertices = part_vertices - center
        cov_mat = np.cov(centered_vertices, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov_mat)
        eigvecs = eigvecs[:, np.argsort(eigvals)[::-1]]

        primary_axis = eigvecs[:, 0]
        projections = centered_vertices @ primary_axis
        median_projection = np.median(projections)

        mask_part1 = projections <= median_projection
        mask_part2 = projections > median_projection

        sub_parts = [part_vertices[mask_part1], part_vertices[mask_part2]]
        for i, sp in enumerate(sub_parts):
            if sp.shape[0] > 1:
                new_part_label = f"{part}_{i}"
                vertices_assignment[part_indices[mask_part1 if i == 0 else mask_part2]] = new_part_label

def place_endpoint_cuboids(vertices, vertices_assignment, adjacency_map, cuboid_centers, cuboid_sizes, cuboid_types, grid_dx):
    """
    为只有一个邻接部分的 part 放置端点 cuboid。
    """
    single_adjacent_parts = {p: ns for p, ns in adjacency_map.items() if len(ns) == 1}
    for part_a, neighbors in single_adjacent_parts.items():
        idx_a = np.where(vertices_assignment == part_a)[0]
        v_a = vertices[idx_a]
        if v_a.shape[0] < 2:
            continue

        part_b = neighbors[0]
        idx_b = np.where(vertices_assignment == part_b)[0]
        v_b = vertices[idx_b]
        if v_b.shape[0] < 2:
            continue

        dist_ab = cdist(v_a, v_b)
        threshold = 3 * grid_dx
        connection_mask_a = np.any(dist_ab <= threshold, axis=1)
        connection_points_a = v_a[connection_mask_a]

        if connection_points_a.shape[0] < 1:
            continue

        center_b = np.mean(v_b, axis=0)
        distances_to_center_b = np.linalg.norm(v_a - center_b, axis=1)
        endpoint = v_a[np.argmax(distances_to_center_b)]

        endpoint_size = np.array([1.2 * grid_dx] * 3)
        endpoint_size = np.maximum(endpoint_size, 1.2 * grid_dx)
        cuboid_centers.append(endpoint)
        cuboid_sizes.append(endpoint_size)
        cuboid_types.append('endpoint')
    
    scale_endpoint = 1.0
    scaled_sizes = scale_endpoint_cuboids(cuboid_centers, cuboid_sizes, cuboid_types, scale_endpoint)

    # 更新 cuboid_sizes 为调整后的尺寸
    cuboid_sizes[:] = scaled_sizes
    

def cuboid_finding(vertices, vertices_assignment, grid_dx, pca_parts=None, include_endpoints=None):
    vertices_assignment = np.array(vertices_assignment, dtype=str)
    unique_parts = np.unique(vertices_assignment)

    cuboid_centers = []
    cuboid_sizes = []
    cuboid_types = []
    boundary_points_list = []
    processed_pairs = set()
    adjacency_map = {}

    # 1️⃣ PCA 分割指定的 parts
    if pca_parts:
        pca_split_parts(vertices, vertices_assignment, pca_parts)

    updated_unique_parts = np.unique(vertices_assignment)
    for i, part_a in enumerate(updated_unique_parts):
        for j, part_b in enumerate(updated_unique_parts):
            
            if part_a >= part_b:
                continue

            idx_a, idx_b = np.where(vertices_assignment == part_a)[0], np.where(vertices_assignment == part_b)[0]
            v_a, v_b = vertices[idx_a], vertices[idx_b]
            
            if v_a.shape[0] < 2 or v_b.shape[0] < 2:
                continue
            
            dist_ab = cdist(v_a, v_b)
            
            threshold = 2 * grid_dx
            
            if np.min(cdist(v_a, v_b)) < threshold:
                
                adjacency_map.setdefault(part_a, []).append(part_b)
                adjacency_map.setdefault(part_b, []).append(part_a)

                # ✅ 在构建邻接关系时直接计算并保存边界点云
                boundary_points_a = v_a[np.any(dist_ab <= threshold, axis=1)]
                boundary_points_b = v_b[np.any(dist_ab <= threshold, axis=0)]
                boundary_points = np.vstack((boundary_points_a, boundary_points_b))

                if boundary_points.shape[0] < 2:
                    continue

                # ✅ 直接保存 boundary_points，避免重复计算
                boundary_points_list.append(boundary_points)

                # ✅ 同时在这里直接创建相邻 Cuboid
                boundary_center = np.mean(boundary_points, axis=0)

                size_x = np.max(boundary_points[:, 0]) - np.min(boundary_points[:, 0])
                size_y = np.max(boundary_points[:, 1]) - np.min(boundary_points[:, 1])
                size_z = np.max(boundary_points[:, 2]) - np.min(boundary_points[:, 2])
                boundary_size = np.array([size_x, size_y, size_z])
                boundary_size = np.maximum(boundary_size, 1.2 * grid_dx)

                cuboid_centers.append(boundary_center)
                cuboid_sizes.append(boundary_size)
                cuboid_types.append('adjacent')

                processed_pairs.add(tuple(sorted([part_a, part_b])))
                
    cuboid_sizes = shrink_adjacent_cuboids(cuboid_centers, cuboid_sizes, cuboid_types, vertices, grid_dx, scale_factor=0.9)
    
    # 3️⃣ 处理端点 cuboid
    if include_endpoints:
        place_endpoint_cuboids(vertices, vertices_assignment, adjacency_map, cuboid_centers, cuboid_sizes, cuboid_types, grid_dx)

    return cuboid_centers, cuboid_sizes, cuboid_types, boundary_points_list


def assign_cuboid_velocity(cuboid_centers, cuboid_sizes, cuboid_types, boundary_points_list, total_frames, gt_meshes, vertices, vertices_assignment, grid_dx, device, delta_time=1/30):
    """
    为每个 Cuboid 分配速度：
    - endpoint：基于 Cuboid 内部点的平均速度。
    - adjacent：基于相邻点云（boundary_points）的平均速度。
    """
    total_frames = total_frames
    frame_start = 0
    frame_end = total_frames - 1

    cuboid_velocity = torch.nn.ParameterList()
    vertices_tensor = torch.tensor(vertices, dtype=torch.float32, device=device)

    for cuboid_idx, (cuboid_center, cuboid_size, cuboid_type) in enumerate(zip(cuboid_centers, cuboid_sizes, cuboid_types)):
        velocity_tensor = torch.zeros((frame_end - frame_start, 1, 3), dtype=torch.float32, device=device)

        if cuboid_type == 'endpoint':
            # ✅ 端点 Cuboid：直接使用内部点
            condition = (
                (vertices_tensor[:, 0] >= cuboid_center[0] - cuboid_size[0] / 2) &
                (vertices_tensor[:, 0] <= cuboid_center[0] + cuboid_size[0] / 2) &
                (vertices_tensor[:, 1] >= cuboid_center[1] - cuboid_size[1] / 2) &
                (vertices_tensor[:, 1] <= cuboid_center[1] + cuboid_size[1] / 2) &
                (vertices_tensor[:, 2] >= cuboid_center[2] - cuboid_size[2] / 2) &
                (vertices_tensor[:, 2] <= cuboid_center[2] + cuboid_size[2] / 2)
            )

            part_indices = torch.where(condition)[0]

            for frame_idx in range(frame_start, frame_end):
                current_frame_pos = gt_meshes[frame_idx].to(device)
                next_frame_pos = gt_meshes[frame_idx + 1].to(device)

                if part_indices.numel() == 0:
                    disp = torch.zeros((1, 3), device=device)
                else:
                    part_vertices_current = current_frame_pos[part_indices]
                    part_vertices_next = next_frame_pos[part_indices]
                    disp = (part_vertices_next.mean(dim=0) - part_vertices_current.mean(dim=0)).unsqueeze(0)

                velocity_tensor[frame_idx - frame_start, 0] = disp / delta_time

        elif cuboid_type == 'adjacent':
            # ✅ 相邻 Cuboid：直接使用 cuboid_finding 返回的边界点云
            boundary_points = torch.tensor(boundary_points_list[cuboid_idx], dtype=torch.float32, device=device)

            for frame_idx in range(frame_start, frame_end):
                current_frame_pos = gt_meshes[frame_idx].to(device)
                next_frame_pos = gt_meshes[frame_idx + 1].to(device)

                part_indices = torch.where((vertices_tensor.unsqueeze(1) == boundary_points).all(dim=2))[0]

                if part_indices.numel() == 0:
                    disp = torch.zeros((1, 3), device=device)
                else:
                    part_vertices_current = current_frame_pos[part_indices]
                    part_vertices_next = next_frame_pos[part_indices]
                    disp = (part_vertices_next.mean(dim=0) - part_vertices_current.mean(dim=0)).unsqueeze(0)

                velocity_tensor[frame_idx - frame_start, 0] = disp / delta_time

        else:
            raise ValueError(f"未知 Cuboid 类型: {cuboid_type}")

        cuboid_velocity.append(torch.nn.Parameter(velocity_tensor, requires_grad=True))

    return cuboid_velocity
