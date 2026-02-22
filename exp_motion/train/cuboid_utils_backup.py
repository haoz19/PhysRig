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
from collections import defaultdict


def find_bone_connections(skeleton_points, skeleton_mesh_path, max_bone_length=None):
    """
    Find bone connections between skeleton points using mesh topology.
    Returns list of (point_idx_a, point_idx_b) pairs that form bones.
    
    Uses a more conservative approach: only connect nearby points that share
    the same mesh component (bone segment).
    """
    skel_mesh = trimesh.load(skeleton_mesh_path, process=False)
    
    # Map each skeleton point to nearest mesh vertex
    mesh_tree = cKDTree(skel_mesh.vertices)
    dists_to_mesh, closest_verts = mesh_tree.query(skeleton_points)
    
    # Get connected components of the mesh (each bone is a separate component)
    try:
        components = trimesh.graph.connected_components(skel_mesh.edges_unique)
    except:
        # Fallback: use face adjacency
        components = trimesh.graph.connected_components(skel_mesh.face_adjacency)
    
    # Map each vertex to its component
    vertex_to_component = {}
    for comp_idx, vertices in enumerate(components):
        for v in vertices:
            vertex_to_component[v] = comp_idx
    
    # Map each skeleton point to its mesh component
    point_to_component = {}
    for i, closest_v in enumerate(closest_verts):
        point_to_component[i] = vertex_to_component.get(closest_v, -1)
    
    # Calculate max bone length if not provided (use 3x average nearest neighbor distance)
    if max_bone_length is None:
        pt_dists = cdist(skeleton_points, skeleton_points)
        np.fill_diagonal(pt_dists, np.inf)
        avg_nn_dist = np.min(pt_dists, axis=1).mean()
        max_bone_length = avg_nn_dist * 3.0
    
    # Find bone connections: skeleton points in SAME component AND close enough
    bones = []
    for i in range(len(skeleton_points)):
        for j in range(i+1, len(skeleton_points)):
            # Must be in same mesh component
            if point_to_component[i] != point_to_component[j]:
                continue
            if point_to_component[i] == -1:  # Unknown component
                continue
            
            # Must be within max bone length
            bone_length = np.linalg.norm(skeleton_points[i] - skeleton_points[j])
            if bone_length <= max_bone_length:
                bones.append((i, j))
    
    return bones


def cuboid_finding_capsule(vertices, grid_dx, skeleton_mesh_path, spheres_per_bone=3):
    """
    Approximate cylinders using chains of spheres along bones.
    This uses existing sphere infrastructure while providing cylinder-like coverage.
    
    Args:
        vertices: Skeleton point positions
        grid_dx: Grid spacing
        skeleton_mesh_path: Path to skeleton mesh for finding bone connections
        spheres_per_bone: Number of spheres to place along each bone (including endpoints)
    
    Returns:
        cuboid_centers: All sphere centers (including interpolated ones)
        cuboid_sizes: Radii for each sphere
        cuboid_types: Type labels
        point_to_original_idx: Maps each cuboid to original skeleton point (-1 for interpolated)
    """
    vertices_original = vertices
    
    # Find bone connections using mesh topology
    bones = find_bone_connections(vertices_original, skeleton_mesh_path)
    
    print(f"Found {len(bones)} bone connections from mesh topology")
    
    # Track which original points are in bones
    in_bone = set()
    for b in bones:
        in_bone.add(b[0])
        in_bone.add(b[1])
    
    # Build cuboid list
    cuboid_centers = []
    cuboid_sizes = []
    cuboid_types = []
    point_to_original_idx = []  # Maps each cuboid to original skeleton point index
    
    # Calculate a base radius from point distances - same as original
    dists = cdist(vertices_original, vertices_original)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    base_radius = (min_dist / 2.0) * 0.7  # Same 0.7 factor as original
    
    # For bones: place spheres along the bone
    for bone_idx, (i, j) in enumerate(bones):
        start = vertices_original[i]
        end = vertices_original[j]
        bone_length = np.linalg.norm(end - start)
        
        # Radius: use base_radius
        radius = base_radius
        
        # Place spheres along bone
        for k in range(spheres_per_bone):
            t = k / max(1, spheres_per_bone - 1)  # 0 to 1
            center = start + t * (end - start)
            
            cuboid_centers.append(center)
            cuboid_sizes.append(np.array([radius, radius, radius]))
            cuboid_types.append(f'bone_{bone_idx}')
            
            # Track original index: endpoints map to original, interpolated = -1
            if k == 0:
                point_to_original_idx.append(i)
            elif k == spheres_per_bone - 1:
                point_to_original_idx.append(j)
            else:
                point_to_original_idx.append(-1)  # Interpolated point
    
    # For isolated points (not in any bone): use same base radius
    isolated_indices = [idx for idx in range(len(vertices_original)) if idx not in in_bone]
    
    for idx in isolated_indices:
        center = vertices_original[idx]
        cuboid_centers.append(center)
        cuboid_sizes.append(np.array([base_radius, base_radius, base_radius]))
        cuboid_types.append('isolated')
        point_to_original_idx.append(idx)
    
    print(f"Capsule cuboids: {len(bones) * spheres_per_bone} bone spheres + {len(isolated_indices)} isolated = {len(cuboid_centers)} total")
    print(f"Radius: {base_radius:.4f}")
    
    return cuboid_centers, cuboid_sizes, cuboid_types, point_to_original_idx, bones


def shrink_adjacent_cuboids(cuboid_centers, cuboid_sizes, cuboid_types, vertices, grid_dx, scale_factor=0.9):#0.9

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
    # Also handle case where vertices_assignment is None (even if mesh is provided)
    if vertices_assignment is None:
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
        special_indices = {}
        
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
        # Return empty boundary_points_list for compatibility
        boundary_points_list = []
        return cuboid_centers, cuboid_sizes, cuboid_types, boundary_points_list
        
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


def assign_cuboid_velocity(cuboid_centers, cuboid_sizes, cuboid_types, total_frames, gt_meshes, vertices, delta_time, device, output_dir=None, num_intermediate_frames: int = 0):
    """
    简化版本：直接基于cuboid中心位置计算速度
    
    参数:
    - output_dir: 如果提供，将保存速度信息到txt文件
    - gt_meshes: 如果为 None，则返回所有速度为零的列表
    """
    num_cuboids = len(cuboid_centers)
    frame_start = 0
    frame_end = total_frames - 1
    num_frames_velocity = frame_end - frame_start

    # 如果 gt_meshes 为 None，返回所有速度为零的列表
    # 注意：当 gt_meshes 为 None 时，num_intermediate_frames 也会为 0，所以不需要处理插入帧逻辑
    if gt_meshes is None:
        velocity_list_param = []
        for _ in range(num_frames_velocity):
            zero_velocity = torch.zeros(num_cuboids, 3, dtype=torch.float32, device=device)
            frame_param = torch.nn.Parameter(zero_velocity, requires_grad=True)
            velocity_list_param.append(frame_param)
        return velocity_list_param

    # 允许0->1帧插入子步：改为动态追加，每个"步"一条记录
    velocity_per_frame = []   # List[List[Tensor]]: 每个步里的所有 cuboid 速度
    position_per_frame = []   # List[List[Tensor]]: 每个步开始时各 cuboid 的位置（用于日志）
    
    # 将原始顶点转换为 tensor，并为每个 cuboid 预计算对应的顶点索引
    # Note: vertex_indices are computed from vertices, but we'll find closest vertex in GT mesh for each frame
    vertices_tensor = torch.tensor(vertices, dtype=torch.float32, device=device)
    cuboid_centers_tensor = torch.tensor(cuboid_centers, dtype=torch.float32, device=device)

    # 按帧段生成步（0->1 支持插入 num_intermediate_frames 个子步，总步数=插入数+1）
    for frame_idx in range(frame_start, frame_end):
        current_frame_pos = gt_meshes[frame_idx].to(device)
        next_frame_pos = gt_meshes[frame_idx + 1].to(device)
        
        # Find closest vertex in current GT mesh for each cuboid center
        vertex_indices = []
        for cuboid_center in cuboid_centers_tensor:
            distances = torch.norm(current_frame_pos - cuboid_center, dim=1)
            vertex_idx = torch.argmin(distances)
            vertex_indices.append(int(vertex_idx))

        steps_this_segment = (num_intermediate_frames + 1) if (frame_idx == 0 and num_intermediate_frames > 0) else 1

        # 对于该段的每个步，记录开始位置与对应的等分速度
        for sub_step in range(steps_this_segment):
            step_vels = []
            step_positions = []
            for cuboid_idx, vertex_idx in enumerate(vertex_indices):
                # Ensure vertex_idx is within bounds
                if vertex_idx >= len(current_frame_pos):
                    vertex_idx = len(current_frame_pos) - 1
                if vertex_idx >= len(next_frame_pos):
                    vertex_idx = len(next_frame_pos) - 1
                p0 = current_frame_pos[vertex_idx]
                p1 = next_frame_pos[vertex_idx]
                vel_full = (p1 - p0) / delta_time
                vel_step = vel_full / steps_this_segment if steps_this_segment > 1 else vel_full
                # 子步开始位置 = p0 + vel_step * delta_time * sub_step
                pos_start = p0 + vel_step * delta_time * sub_step if steps_this_segment > 1 else p0
                step_vels.append(vel_step)
                step_positions.append(pos_start)
            velocity_per_frame.append(step_vels)
            position_per_frame.append(step_positions)
    
    velocity_list_param = []
    for frame_velocities in velocity_per_frame:
        frame_tensor = torch.stack(frame_velocities, dim=0)
        frame_param = torch.nn.Parameter(frame_tensor, requires_grad=True)
        velocity_list_param.append(frame_param)
    
    # 保存速度信息到txt文件（直接从velocity_list_param取）
    if output_dir is not None:
        import os
        os.makedirs(output_dir, exist_ok=True)
        velocity_log_path = os.path.join(output_dir, "cuboid_velocities.txt")
        
        with open(velocity_log_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"Cuboid Velocity Log\n")
            f.write(f"Total Cuboids: {num_cuboids}\n")
            f.write(f"Total Frames: {num_frames_velocity}\n")
            time_scale = 1 + max(0, int(num_intermediate_frames))
            f.write(f"Insert on 0->1: {num_intermediate_frames} subframes | Steps total: {len(velocity_list_param)} (base={num_frames_velocity})\n")
            f.write("=" * 80 + "\n\n")
            
            # 逐帧记录每个步的速度（直接从velocity_list_param取）
            for frame_idx in range(len(velocity_list_param)):
                f.write(f"\n{'='*60}\n")
                f.write(f"Step {frame_idx} (uniform-substepped where applicable)\n")
                f.write(f"{'='*60}\n")
                
                frame_param = velocity_list_param[frame_idx]  # 直接取Parameter
                frame_positions = position_per_frame[frame_idx]  # 获取该帧的实际位置
                for cuboid_idx in range(num_cuboids):
                    vel = frame_param[cuboid_idx]
                    vel_np = vel.detach().cpu().numpy()
                    vel_magnitude = np.linalg.norm(vel_np)
                    
                    # 使用该帧的实际位置，而不是初始位置
                    actual_pos = frame_positions[cuboid_idx].detach().cpu().numpy()
                    
                    f.write(f"\nCuboid {cuboid_idx}:\n")
                    f.write(f"  Type: {cuboid_types[cuboid_idx]}\n")
                    f.write(f"  Center: [{actual_pos[0]:.6f}, {actual_pos[1]:.6f}, {actual_pos[2]:.6f}]\n")
                    f.write(f"  Velocity: [{vel_np[0]:.6f}, {vel_np[1]:.6f}, {vel_np[2]:.6f}]\n")
                    f.write(f"  Magnitude: {vel_magnitude:.6f}\n")
            
            f.write(f"\n{'='*80}\n")
            f.write("End of Velocity Log\n")
            f.write(f"{'='*80}\n")
        
        print(f"✅ Cuboid velocities saved to: {velocity_log_path}")
    
    return velocity_list_param


def assign_capsule_velocity(cuboid_centers, cuboid_sizes, cuboid_types, point_to_original_idx, bones, 
                            total_frames, gt_meshes, original_vertices, delta_time, device, 
                            spheres_per_bone=3, output_dir=None, num_intermediate_frames: int = 0):
    """
    Assign velocities for capsule-based cuboids.
    
    Endpoint spheres get velocity from their skeleton point.
    Interpolated spheres get linearly interpolated velocity from bone endpoints.
    """
    num_cuboids = len(cuboid_centers)
    frame_start = 0
    frame_end = total_frames - 1
    num_frames_velocity = frame_end - frame_start
    
    if gt_meshes is None:
        velocity_list_param = []
        for _ in range(num_frames_velocity):
            zero_velocity = torch.zeros(num_cuboids, 3, dtype=torch.float32, device=device)
            frame_param = torch.nn.Parameter(zero_velocity, requires_grad=True)
            velocity_list_param.append(frame_param)
        return velocity_list_param
    
    # Build mapping: for each bone, track which cuboid indices belong to it
    cuboid_to_bone_info = {}  # cuboid_idx -> (bone_idx, t, start_orig, end_orig)
    
    cuboid_idx = 0
    for bone_idx, (start_orig, end_orig) in enumerate(bones):
        for k in range(spheres_per_bone):
            t = k / max(1, spheres_per_bone - 1)
            cuboid_to_bone_info[cuboid_idx] = (bone_idx, t, start_orig, end_orig)
            cuboid_idx += 1
    
    velocity_per_frame = []
    
    for frame_idx in range(frame_start, frame_end):
        current_frame_pos = gt_meshes[frame_idx].to(device)
        next_frame_pos = gt_meshes[frame_idx + 1].to(device)
        
        steps_this_segment = (num_intermediate_frames + 1) if (frame_idx == 0 and num_intermediate_frames > 0) else 1
        
        for sub_step in range(steps_this_segment):
            frame_vels = []
            
            for c_idx in range(num_cuboids):
                if c_idx in cuboid_to_bone_info:
                    # This is a bone sphere - interpolate velocity from endpoints
                    bone_idx, t, start_orig, end_orig = cuboid_to_bone_info[c_idx]
                    
                    # Clamp indices
                    start_orig = min(start_orig, len(current_frame_pos) - 1)
                    end_orig = min(end_orig, len(current_frame_pos) - 1)
                    
                    # Get endpoint velocities
                    p0_start = current_frame_pos[start_orig]
                    p1_start = next_frame_pos[start_orig]
                    vel_start = (p1_start - p0_start) / delta_time
                    
                    p0_end = current_frame_pos[end_orig]
                    p1_end = next_frame_pos[end_orig]
                    vel_end = (p1_end - p0_end) / delta_time
                    
                    # Interpolate velocity
                    vel = vel_start * (1 - t) + vel_end * t
                    
                else:
                    # Isolated point - use its original index
                    orig_idx = point_to_original_idx[c_idx]
                    orig_idx = min(orig_idx, len(current_frame_pos) - 1)
                    
                    p0 = current_frame_pos[orig_idx]
                    p1 = next_frame_pos[orig_idx]
                    vel = (p1 - p0) / delta_time
                
                # Apply substep scaling if needed
                if steps_this_segment > 1:
                    vel = vel / steps_this_segment
                
                frame_vels.append(vel)
            
            velocity_per_frame.append(frame_vels)
    
    velocity_list_param = []
    for frame_velocities in velocity_per_frame:
        frame_tensor = torch.stack(frame_velocities, dim=0)
        frame_param = torch.nn.Parameter(frame_tensor, requires_grad=True)
        velocity_list_param.append(frame_param)
    
    print(f"✅ Assigned capsule velocities: {num_cuboids} cuboids, {len(velocity_list_param)} frames")
    
    return velocity_list_param