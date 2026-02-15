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

#test
def shrink_adjacent_cuboids(cuboid_centers, cuboid_sizes, cuboid_types, vertices, grid_dx, scale_factor=0.9):
    min_size = 1.2 * grid_dx

    for i, (center, size, ctype) in enumerate(zip(cuboid_centers, cuboid_sizes, cuboid_types)):
        if ctype != 'adjacent':
            continue

        half_size = size / 2.0
        lower_bound = center - half_size
        upper_bound = center + half_size
        in_cuboid_mask = np.any(np.all((vertices >= lower_bound) & (vertices <= upper_bound), axis=1))

        while in_cuboid_mask:
            for dim in range(3):
                if size[dim] > min_size:
                    size[dim] *= scale_factor

            half_size = size / 2.0
            lower_bound = center - half_size
            upper_bound = center + half_size
            in_cuboid_mask = np.any(np.all((vertices >= lower_bound) & (vertices <= upper_bound), axis=1))

            if np.all(size <= min_size):
                size = np.maximum(size, min_size)
                break

        cuboid_sizes[i] = size

    return cuboid_sizes

    
def cuboid_finding(vertices, grid_dx, vertices_assignment=None, mesh=None,
                   sizing_mode="fixed", sizing_coeff=0.7, knn_k=20,
                   full_pointcloud=None):
    if vertices_assignment is None:
        cuboid_centers = []
        cuboid_sizes = []
        cuboid_types = []

        vertices_original = vertices
        num_vertices = vertices_original.shape[0]

        print(f"\n=== Cuboid sizing: mode={sizing_mode}, coeff={sizing_coeff}, knn_k={knn_k} ===")

        # =====================================================================
        # Sizing strategy selection (exclusive if-elif-else, only one runs)
        # =====================================================================

        # 1. Fixed Size (original behavior)
        if sizing_mode == "fixed":
            dists = cdist(vertices_original, vertices_original)
            min_dist = np.min(dists[np.nonzero(dists)])
            fixed_radius = (min_dist / 2.0) * sizing_coeff
            print(f"  Fixed: min_dist={min_dist:.6f}, radius={fixed_radius:.6f}")

            for i in range(num_vertices):
                cuboid_centers.append(vertices_original[i])
                cuboid_sizes.append(np.array([fixed_radius, fixed_radius, fixed_radius]))
                cuboid_types.append('point_sphere')

        # 2. Adaptive Neighbor Distance (per-cuboid nearest neighbor)
        elif sizing_mode == "adaptive":
            tree = cKDTree(vertices_original)
            # k=2 because the first neighbor is the point itself (distance 0)
            dists_nn, _ = tree.query(vertices_original, k=2)
            nearest_dists = dists_nn[:, 1]  # distance to closest other cuboid center

            for i in range(num_vertices):
                radius_i = (nearest_dists[i] / 2.0) * sizing_coeff
                cuboid_centers.append(vertices_original[i])
                cuboid_sizes.append(np.array([radius_i, radius_i, radius_i]))
                cuboid_types.append('point_sphere')
                print(f"  Cuboid {i}: nearest_dist={nearest_dists[i]:.6f}, radius={radius_i:.6f}")

        # 3. KNN (K-Nearest Points from dense point cloud)
        #    HARDCODED: coefficient is 1.0, ignoring sizing_coeff parameter
        elif sizing_mode == "knn":
            if full_pointcloud is None:
                raise ValueError("sizing_mode='knn' requires full_pointcloud (dense point cloud) to be provided.")
            pc_tree = cKDTree(full_pointcloud)
            # Query knn_k nearest dense points for each cuboid center
            dists_knn, _ = pc_tree.query(vertices_original, k=knn_k)
            # Distance to the K-th nearest point (last column)
            kth_dists = dists_knn[:, -1]

            for i in range(num_vertices):
                radius_i = kth_dists[i] * 1.0  # HARDCODED coefficient = 1.0
                cuboid_centers.append(vertices_original[i])
                cuboid_sizes.append(np.array([radius_i, radius_i, radius_i]))
                cuboid_types.append('point_sphere')
                print(f"  Cuboid {i}: kth_dist={kth_dists[i]:.6f}, radius={radius_i:.6f}")

        # 4. Hybrid (KNN + Fixed Radius constraint)
        #    KNN sets the base radius (coeff=1.0); sizing_coeff controls the fixed constraint
        elif sizing_mode == "hybrid":
            if full_pointcloud is None:
                raise ValueError("sizing_mode='hybrid' requires full_pointcloud (dense point cloud) to be provided.")
            # KNN distances from dense point cloud
            pc_tree = cKDTree(full_pointcloud)
            dists_knn, _ = pc_tree.query(vertices_original, k=knn_k)
            kth_dists = dists_knn[:, -1]
            
            # Fixed radius constraint (controlled by sizing_coeff)
            fixed_radius = grid_dx * sizing_coeff

            for i in range(num_vertices):
                knn_radius = kth_dists[i] * 1.0  # HARDCODED coefficient = 1.0 for KNN
                # Take minimum of KNN radius and fixed constraint
                radius_i = min(knn_radius, fixed_radius)
                cuboid_centers.append(vertices_original[i])
                cuboid_sizes.append(np.array([radius_i, radius_i, radius_i]))
                cuboid_types.append('point_sphere')
                print(f"  Cuboid {i}: knn_r={knn_radius:.6f}, fixed_r={fixed_radius:.6f}, final={radius_i:.6f}")

        # 5. Ray Casting / Sphere Casting (density-based boundary detection)
        elif sizing_mode == "raycast":
            if full_pointcloud is None:
                raise ValueError("sizing_mode='raycast' requires full_pointcloud (dense point cloud) to be provided.")
            pc_tree = cKDTree(full_pointcloud)

            # Generate uniformly distributed ray directions (26 directions:
            # 6 axis-aligned + 12 edge diagonals + 8 corner diagonals)
            ray_dirs = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        d = np.array([dx, dy, dz], dtype=np.float64)
                        ray_dirs.append(d / np.linalg.norm(d))
            ray_dirs = np.array(ray_dirs)  # (26, 3)

            # Cone half-angle for ray search (in radians, ~30 degrees)
            cone_half_angle = np.pi / 6.0
            cos_threshold = np.cos(cone_half_angle)

            # Number of radial bins for density detection
            num_bins = 20

            for i in range(num_vertices):
                center = vertices_original[i]

                # Find all points within a generous search radius
                # Use knn_k to set a baseline search radius
                knn_dists_i, _ = pc_tree.query(center, k=min(knn_k * 10, len(full_pointcloud)))
                search_radius = knn_dists_i[-1] * 2.0

                # Get all points within search radius
                nearby_indices = pc_tree.query_ball_point(center, search_radius)
                if len(nearby_indices) == 0:
                    # Fallback: use a small default radius
                    radius_i = grid_dx
                    cuboid_centers.append(center)
                    cuboid_sizes.append(np.array([radius_i, radius_i, radius_i]))
                    cuboid_types.append('point_sphere')
                    print(f"  Cuboid {i}: raycast fallback, radius={radius_i:.6f}")
                    continue

                nearby_points = full_pointcloud[nearby_indices]
                offsets = nearby_points - center
                dists_from_center = np.linalg.norm(offsets, axis=1)
                # Avoid division by zero
                valid_mask = dists_from_center > 1e-10
                offsets_valid = offsets[valid_mask]
                dists_valid = dists_from_center[valid_mask]

                if len(dists_valid) == 0:
                    radius_i = grid_dx
                    cuboid_centers.append(center)
                    cuboid_sizes.append(np.array([radius_i, radius_i, radius_i]))
                    cuboid_types.append('point_sphere')
                    print(f"  Cuboid {i}: raycast fallback (no valid points), radius={radius_i:.6f}")
                    continue

                # Normalize offsets to get directions
                directions = offsets_valid / dists_valid[:, np.newaxis]

                # For each ray direction, find boundary distance
                ray_boundary_dists = []
                for ray_dir in ray_dirs:
                    # Points within the cone of this ray direction
                    cos_angles = directions @ ray_dir
                    in_cone = cos_angles >= cos_threshold
                    cone_dists = dists_valid[in_cone]

                    if len(cone_dists) < 2:
                        continue  # Skip rays with insufficient points

                    # Bin by distance and detect density drop-off
                    max_cone_dist = cone_dists.max()
                    bin_edges = np.linspace(0, max_cone_dist, num_bins + 1)

                    # Find the last bin that contains points
                    boundary_dist = max_cone_dist
                    for b in range(num_bins):
                        bin_mask = (cone_dists >= bin_edges[b]) & (cone_dists < bin_edges[b + 1])
                        if bin_mask.sum() == 0:
                            # Empty bin found: boundary is at this bin's start
                            boundary_dist = bin_edges[b]
                            break

                    ray_boundary_dists.append(boundary_dist)

                if len(ray_boundary_dists) > 0:
                    # Use minimum across ray directions (conservative)
                    radius_i = np.min(ray_boundary_dists) * sizing_coeff
                else:
                    radius_i = grid_dx

                # Ensure a minimum radius
                radius_i = max(radius_i, grid_dx * 0.1)

                cuboid_centers.append(center)
                cuboid_sizes.append(np.array([radius_i, radius_i, radius_i]))
                cuboid_types.append('point_sphere')
                print(f"  Cuboid {i}: raycast radius={radius_i:.6f} (from {len(ray_boundary_dists)} rays)")

        else:
            raise ValueError(
                f"Unknown sizing_mode: '{sizing_mode}'. "
                f"Must be one of: 'fixed', 'adaptive', 'knn', 'hybrid', 'raycast'"
            )

        boundary_points_list = []
        return cuboid_centers, cuboid_sizes, cuboid_types, boundary_points_list
        
    vertices_assignment = np.array(vertices_assignment, dtype=str)
    unique_parts = np.unique(vertices_assignment)

    cuboid_centers = []
    cuboid_sizes = []
    cuboid_types = []
    boundary_points_list = []
    
    remove_index = None
    
    adjacency_map = {}
    for face in mesh.faces:
        num_vertices = len(face)
        for i in range(num_vertices):
            v1 = face[i]
            v2 = face[(i + 1) % num_vertices]
            part1 = str(vertices_assignment[v1])
            part2 = str(vertices_assignment[v2])
            if part1 != part2:
                adjacency_map.setdefault(part1, set()).add(part2)
                adjacency_map.setdefault(part2, set()).add(part1)

    cuboid_threshold = 2.0 * grid_dx
    processed_pairs = set()
    
    for part_a in sorted(unique_parts):
        neighbors = sorted(list(adjacency_map.get(part_a, set())))
        for part_b in neighbors:
            if part_a >= part_b:
                continue
            pair_key = (part_a, part_b)
            if pair_key in processed_pairs:
                continue
            processed_pairs.add(pair_key)

            idx_a = np.where(vertices_assignment == part_a)[0]
            idx_b = np.where(vertices_assignment == part_b)[0]
            if len(idx_a) < 2 or len(idx_b) < 2:
                continue
            v_a = vertices[idx_a]
            v_b = vertices[idx_b]

            dist_ab = cdist(v_a, v_b)
            boundary_points_a = v_a[np.any(dist_ab <= cuboid_threshold, axis=1)]
            boundary_points_b = v_b[np.any(dist_ab <= cuboid_threshold, axis=0)]
            boundary_points = np.vstack((boundary_points_a, boundary_points_b))
            if boundary_points.shape[0] < 2:
                continue

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


def assign_cuboid_velocity(cuboid_centers, cuboid_sizes, cuboid_types, total_frames, gt_meshes, vertices, delta_time, device, output_dir=None, num_intermediate_frames: int = 0, cuboid_update_mode: str = "both", position_method: str = "mean"):
    """
    TASK 2: New velocity/location updating routine based on tracked points from frame 0
    
    参数:
    - output_dir: 如果提供，将保存速度信息到txt文件
    - gt_meshes: 如果为 None，则返回所有速度为零的列表
    - cuboid_update_mode: "velocity_only", "location_only", or "both"
        - "velocity_only": Update cuboid velocity from tracked points, keep location fixed
        - "location_only": Update cuboid location from tracked points, keep velocity traditional
        - "both": Update both velocity and location from tracked points
    - position_method: method for computing cuboid center from tracked points
        - "mean": Simple average of all tracked points (default, original behavior)
        - "median": Median position of tracked points
        - "weighted": Inverse distance weighted from initial cuboid center
        - "bbox": Center of bounding box (min/max)
        - "adaptive": Trimmed mean with outlier removal + density weighting
        - "pca": Principal Component Analysis center (covariance-based)
        - "optimized": LBFGS optimization to minimize L2 deviation (EXPENSIVE!)
    
    NOTE: Only the active method is computed (not all 7), for efficiency and numerical stability.
    """
    VALID_METHODS = ("mean", "median", "weighted", "bbox", "adaptive", "pca", "optimized")
    if position_method not in VALID_METHODS:
        raise ValueError(f"Unknown position_method: {position_method}. Must be one of: {VALID_METHODS}")
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
        return velocity_list_param, [], cuboid_update_mode

    # TASK 2: Track points from frame 0
    frame_0_mesh = gt_meshes[0].cpu().numpy()
    cuboid_centers_np = np.array(cuboid_centers)
    cuboid_sizes_np = np.array(cuboid_sizes)
    
    cuboid_point_indices = []
    # Only compute weights when "weighted" method is selected to avoid unnecessary
    # computation that could affect numerical behavior of other methods.
    need_weights = (position_method == "weighted")
    cuboid_weights = [] if need_weights else None
    
    print(f"\n=== Frame 0: Identifying points covered by each cuboid (position_method={position_method}) ===")
    for cuboid_idx, (center, size) in enumerate(zip(cuboid_centers_np, cuboid_sizes_np)):
        half_size = size / 2.0
        lower_bound = center - half_size
        upper_bound = center + half_size
        
        in_cuboid_mask = np.all((frame_0_mesh >= lower_bound) & (frame_0_mesh <= upper_bound), axis=1)
        point_indices = np.where(in_cuboid_mask)[0]
        
        # Compute inverse-distance weights only for "weighted" method
        if need_weights:
            if len(point_indices) > 0:
                points_in_cuboid = frame_0_mesh[point_indices]
                distances = np.linalg.norm(points_in_cuboid - center, axis=1)
                weights = 1.0 / (distances + 1e-6)
                weights = weights / weights.sum()
                weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
            else:
                weights_tensor = torch.tensor([], dtype=torch.float32, device=device)
            cuboid_weights.append(weights_tensor)
        
        cuboid_point_indices.append(point_indices.tolist())
        print(f"  Cuboid {cuboid_idx} ({cuboid_types[cuboid_idx]}): covers {len(point_indices)} points")
    
    total_covered = sum(len(idx) for idx in cuboid_point_indices)
    print(f"\n  Total points covered: {total_covered}")
    input("Press Enter to continue...")
    
    velocity_per_frame = []
    position_per_frame = []
    
    cuboid_centers_tensor = torch.tensor(cuboid_centers, dtype=torch.float32, device=device)

    for frame_idx in range(frame_start, frame_end):
        current_frame_pos = gt_meshes[frame_idx].to(device)
        next_frame_pos = gt_meshes[frame_idx + 1].to(device)
        
        steps_this_segment = (num_intermediate_frames + 1) if (frame_idx == 0 and num_intermediate_frames > 0) else 1

        for sub_step in range(steps_this_segment):
            step_vels = []
            step_positions = []
            
            for cuboid_idx in range(num_cuboids):
                point_indices = cuboid_point_indices[cuboid_idx]
                
                if len(point_indices) == 0:
                    vel_step = torch.zeros(3, dtype=torch.float32, device=device)
                    pos_start = cuboid_centers_tensor[cuboid_idx]
                else:
                    current_points = current_frame_pos[point_indices]
                    next_points = next_frame_pos[point_indices]
                    
                    # ========================================
                    # Position method selection
                    # Only the active method is computed for efficiency.
                    # ========================================
                    
                    # 1. Mean (default - original behavior)
                    if position_method == "mean":
                        p0 = current_points.mean(dim=0)
                        p1 = next_points.mean(dim=0)
                    
                    # 2. Median
                    elif position_method == "median":
                        p0 = current_points.median(dim=0).values
                        p1 = next_points.median(dim=0).values
                    
                    # 3. Weighted average (inverse distance weighting)
                    elif position_method == "weighted":
                        weights = cuboid_weights[cuboid_idx]
                        p0 = (current_points * weights.unsqueeze(1)).sum(dim=0)
                        p1 = (next_points * weights.unsqueeze(1)).sum(dim=0)
                    
                    # 4. Bounding box center
                    elif position_method == "bbox":
                        p0_bbox_min = current_points.min(dim=0).values
                        p0_bbox_max = current_points.max(dim=0).values
                        p0 = (p0_bbox_min + p0_bbox_max) / 2.0
                        
                        p1_bbox_min = next_points.min(dim=0).values
                        p1_bbox_max = next_points.max(dim=0).values
                        p1 = (p1_bbox_min + p1_bbox_max) / 2.0
                    
                    # 5. Adaptive (trimmed mean with outlier removal + density weighting)
                    elif position_method == "adaptive":
                        def adaptive_center(points):
                            if len(points) <= 3:
                                return points.mean(dim=0)
                            
                            point_median = points.median(dim=0).values
                            distances_from_median = torch.norm(points - point_median, dim=1)
                            
                            q1 = torch.quantile(distances_from_median, 0.25)
                            q3 = torch.quantile(distances_from_median, 0.75)
                            iqr = q3 - q1
                            outlier_threshold = q3 + 1.5 * iqr
                            
                            inlier_mask = distances_from_median <= outlier_threshold
                            inliers = points[inlier_mask]
                            
                            if len(inliers) == 0:
                                return points.mean(dim=0)
                            
                            inlier_distances = torch.norm(inliers - point_median, dim=1)
                            density_weights = 1.0 / (inlier_distances + 1e-6)
                            density_weights = density_weights / density_weights.sum()
                            
                            return (inliers * density_weights.unsqueeze(1)).sum(dim=0)
                        
                        p0 = adaptive_center(current_points)
                        p1 = adaptive_center(next_points)
                    
                    # 6. PCA-based center (using covariance matrix and principal axes)
                    elif position_method == "pca":
                        def pca_center(points):
                            center_avg = points.mean(dim=0)
                            centered = points - center_avg
                            cov = (centered.T @ centered) / len(points)
                            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
                            sizes_along_axes = 2.0 * torch.sqrt(eigenvalues)
                            return center_avg
                        
                        p0 = pca_center(current_points)
                        p1 = pca_center(next_points)
                    
                    # 7. Optimization-based (Minimize Deviation using LBFGS)
                    elif position_method == "optimized":
                        def optimize_position(points, initial_center, max_iters=10):
                            pos = initial_center.clone().requires_grad_(True)
                            optimizer = torch.optim.LBFGS([pos], max_iter=max_iters, line_search_fn='strong_wolfe')
                            
                            def closure():
                                optimizer.zero_grad()
                                distances = torch.norm(points - pos, dim=1)
                                loss = distances.pow(2).sum()
                                loss.backward()
                                return loss
                            
                            optimizer.step(closure)
                            return pos.detach()
                        
                        p0 = optimize_position(current_points, cuboid_centers_tensor[cuboid_idx])
                        p1 = optimize_position(next_points, cuboid_centers_tensor[cuboid_idx])
                    
                    vel_full = (p1 - p0) / delta_time
                    vel_step = vel_full / steps_this_segment if steps_this_segment > 1 else vel_full
                    
                    if cuboid_update_mode == "velocity_only":
                        pos_start = cuboid_centers_tensor[cuboid_idx]
                    elif cuboid_update_mode == "location_only":
                        distances = torch.norm(current_frame_pos - cuboid_centers_tensor[cuboid_idx], dim=1)
                        closest_idx = torch.argmin(distances)
                        trad_p0 = current_frame_pos[closest_idx]
                        trad_p1 = next_frame_pos[closest_idx]
                        vel_full = (trad_p1 - trad_p0) / delta_time
                        vel_step = vel_full / steps_this_segment if steps_this_segment > 1 else vel_full
                        pos_start = p0 + vel_step * delta_time * sub_step if steps_this_segment > 1 else p0
                    else:  # "both"
                        pos_start = p0 + vel_step * delta_time * sub_step if steps_this_segment > 1 else p0
                
                step_vels.append(vel_step)
                step_positions.append(pos_start)
            
            velocity_per_frame.append(step_vels)
            position_per_frame.append(step_positions)
    
    velocity_list_param = []
    position_list_tensors = []
    
    for frame_velocities in velocity_per_frame:
        frame_tensor = torch.stack(frame_velocities, dim=0)
        frame_param = torch.nn.Parameter(frame_tensor, requires_grad=True)
        velocity_list_param.append(frame_param)
    
    for frame_positions in position_per_frame:
        position_tensor = torch.stack(frame_positions, dim=0)
        position_list_tensors.append(position_tensor)
    
    if output_dir is not None:
        import os
        os.makedirs(output_dir, exist_ok=True)
        velocity_log_path = os.path.join(output_dir, "cuboid_velocities.txt")
        
        with open(velocity_log_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"Cuboid Velocity Log\n")
            f.write(f"Update Mode: {cuboid_update_mode}\n")
            f.write(f"Total Cuboids: {num_cuboids}\n")
            f.write(f"Total Frames: {num_frames_velocity}\n")
            f.write(f"Insert on 0->1: {num_intermediate_frames} subframes | Steps total: {len(velocity_list_param)} (base={num_frames_velocity})\n")
            
            f.write("\nTracked Points per Cuboid (from Frame 0):\n")
            for cuboid_idx in range(num_cuboids):
                num_tracked = len(cuboid_point_indices[cuboid_idx])
                f.write(f"  Cuboid {cuboid_idx}: {num_tracked} points\n")
            
            f.write("=" * 80 + "\n\n")
            
            for frame_idx in range(len(velocity_list_param)):
                f.write(f"\n{'='*60}\n")
                f.write(f"Step {frame_idx} (uniform-substepped where applicable)\n")
                f.write(f"{'='*60}\n")
                
                frame_param = velocity_list_param[frame_idx]
                frame_positions = position_per_frame[frame_idx]
                for cuboid_idx in range(num_cuboids):
                    vel = frame_param[cuboid_idx]
                    vel_np = vel.detach().cpu().numpy()
                    vel_magnitude = np.linalg.norm(vel_np)
                    
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
    
    return velocity_list_param, position_list_tensors, cuboid_update_mode
