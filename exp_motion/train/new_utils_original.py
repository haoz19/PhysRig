import os
import numpy as np
import pandas as pd
import open3d as o3d
import trimesh
import math
import torch
import glob
import pdb
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import random
from typing import List
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree, ConvexHull, KDTree
from scipy.spatial.distance import cdist
import point_cloud_utils as pcu

loss_history = {
    "frame_losses": [],  # 改为列表
}

def set_random_seed(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def init_logit_for_youngs(target_youngs, E_min, E_max):
    # 先算 alpha = (target_youngs - E_min) / (E_max - E_min)
    alpha = (target_youngs - E_min) / (E_max - E_min)
    # 夹一下，避免 alpha=0 or 1 时 log 爆掉
    alpha = max(min(alpha, 0.9999), 0.0001)
    # logit 反函数
    unbounded = math.log(alpha / (1.0 - alpha))
    return unbounded

def load_mesh_sequences(mesh_dir):
    mesh_files = sorted(
        glob.glob(os.path.join(mesh_dir, "gt_*.ply")),
        key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0])  # 先去掉路径，正确提取帧序号
    )

    # 将每一帧的 mesh 数据加载为 torch tensor
    gt_meshes = [torch.tensor(pcu.load_mesh_v(mesh_file), dtype=torch.float32).to("cuda") for mesh_file in mesh_files]
    return gt_meshes
    
def vertice_assignment(vertices, weights):
    if vertices.shape[0] != weights.shape[0]:
        print(vertices.shape[0],weights.shape[0])
        raise ValueError("顶点数和权重矩阵的顶点数不匹配。")
    
    # 根据最大权重直接分配顶点到骨骼部分
    part_assignments = np.argmax(weights, axis=1)
    
    return part_assignments

def match_point(sim_path, sim_mask_path, threshold=1e-4, device="cuda"):
    
    # 加载点云数据并转换为 float32
    sim_points = torch.tensor(pcu.load_mesh_v(sim_path), dtype=torch.float32).to(device)
    mask_points = torch.tensor(pcu.load_mesh_v(sim_mask_path), dtype=torch.float32).to(device)

    # print(f"sim_points: {sim_points.shape[0]}")
    # print(f"mask_points: {mask_points.shape[0]}")

    # 转换为 numpy 数组以便进行最近邻搜索
    sim_points_np = sim_points.cpu().numpy()
    mask_points_np = mask_points.cpu().numpy()

    # 使用 KD-Tree 进行最近邻搜索
    kdtree = cKDTree(sim_points_np)
    distances, indices = kdtree.query(mask_points_np)

    # 过滤满足距离阈值的点，确保能匹配到所有 mask_points
    matched_indices_mask = distances < threshold
    sim_mask_indices = indices[matched_indices_mask]

    sim_mask_indices = torch.tensor(sim_mask_indices, dtype=torch.long, device=device)
    
    return sim_mask_indices


def calculate_bounding_box(meshes):
    # 如果 meshes 是一个 NumPy 数组，先检查是否为空，并转换为 torch.Tensor
    if isinstance(meshes, np.ndarray):
        if meshes.size == 0:
            raise ValueError("meshes 数组为空，无法计算 bounding box。")
        # 转换为 torch.Tensor（默认在 CPU 上）
        meshes = torch.tensor(meshes, dtype=torch.float32)
    # 如果 meshes 是一个列表，检查列表长度
    elif isinstance(meshes, list):
        if len(meshes) == 0:
            raise ValueError("meshes 列表为空，无法计算 bounding box。")
    
    # 如果 meshes 是 torch.Tensor 且为二维（例如形状为 [N, 3]），则认为是单个 mesh
    if isinstance(meshes, torch.Tensor) and meshes.ndim == 2:
        device = meshes.device  # 获取设备
        global_min = meshes.min(dim=0).values
        global_max = meshes.max(dim=0).values
    else:
        # 如果 meshes 是一个列表（每个元素是一个 mesh），先设定设备为列表中第一个元素的设备
        device = meshes[0].device if hasattr(meshes[0], 'device') else torch.device('cpu')
        global_min = torch.tensor([float('inf'), float('inf'), float('inf')], device=device)
        global_max = torch.tensor([float('-inf'), float('-inf'), float('-inf')], device=device)
        for mesh in meshes:
            # 如果 mesh 是 NumPy 数组，则先转换为 torch.Tensor
            if isinstance(mesh, np.ndarray):
                mesh = torch.tensor(mesh, dtype=torch.float32, device=device)
            if mesh.numel() == 0:
                continue  # 跳过空的 mesh
            min_vals = mesh.min(dim=0).values
            max_vals = mesh.max(dim=0).values
            global_min = torch.minimum(global_min, min_vals)
            global_max = torch.maximum(global_max, max_vals)
    
    bbox = {
        'x_min': global_min[0].item(),
        'x_max': global_max[0].item(),
        'y_min': global_min[1].item(),
        'y_max': global_max[1].item(),
        'z_min': global_min[2].item(),
        'z_max': global_max[2].item()
    }
    
    return bbox

def check_gradients(*params):
    for name, param in params:
        if param.grad is None:
            print(f"Gradient of {name} is None.")
            continue
        
        grad = param.grad.clone().detach().cpu().numpy()  # 确保不会影响计算图

            # 计算梯度的统计信息
        grad_abs = np.abs(grad)
        grad_mean = np.mean(grad)
        grad_abs_min, grad_abs_max= np.min(grad_abs), np.max(grad_abs)
        non_zero_count = np.count_nonzero(grad)
        very_small_1e6 = np.sum(grad_abs < 1e-6)
        very_small_1e9 = np.sum(grad_abs < 1e-9)
        very_small_1e12 = np.sum(grad_abs < 1e-12)

        print(f"  Shape: {grad.shape}")
        print(f" Abs Min: {grad_abs_min:.5e}, Abs Max: {grad_abs_max:.5e}, Mean: {grad_mean:.5e}")
        print(f"  Non-zero elements: {non_zero_count}/{grad.size} ")
        print(f"  Very small values (<1e-6): {very_small_1e6}/{grad.size}")
        print(f"  Very small values (<1e-9): {very_small_1e9}/{grad.size}")
        print(f"  Very small values (<1e-12): {very_small_1e12}/{grad.size}")

        print("-" * 50)


def save_ply(points_list: List[np.ndarray], step: int, dataset_dir: str, frame_idx: int, for_gt: bool = False):
    """
    保存多组点云数据为多个 PLY 文件至指定目录.
    :param points_list: List of (N, 3) numpy arrays, each representing a point cloud.
    :param step: 当前训练的 step，用于生成唯一的文件名.
    :param dataset_dir: 数据集的根目录，将用于保存 output 文件夹.
    :param frame_idx: 当前帧的索引，用于生成唯一的文件名.
    """
    # 设置输出路径，将 PLY 文件保存到 dataset_dir 下的 output 文件夹
    output_dir = os.path.join(dataset_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍历每一组点云数据（模拟的和 GT 的）
    for idx, points in enumerate(points_list):
        # 确保 points 是 (N, 3) 的形状
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Expected points to have shape (N, 3), but got shape {points.shape}")
        
        # 指定保存文件路径（包含当前 step、帧索引和序号）
        if for_gt:
            filename = os.path.join(output_dir, f"gt_{frame_idx}.ply")
        else:
            filename = os.path.join(output_dir, f"output_step_{step:04d}_frame_{frame_idx:04d}_part_{idx}.ply")
        
        # 转换点数据格式
        vertices = [tuple(point) for point in points]
        vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
        vertices = np.array(vertices, dtype=vertex_dtype)
        ply_data = PlyData([PlyElement.describe(vertices, 'vertex')], text=True)
        ply_data.write(filename)
    
    print(f"Results have been saved to {output_dir}")

# 可视化函数

def generate_colors(num_colors):
    """简单随机生成若干颜色"""
    colors = []
    for _ in range(num_colors):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        colors.append(f'rgb({r},{g},{b})')
    return colors


def log_losses(frame_loss=None, iteration=None):
    """
    记录每步训练的损失值到loss_history中。
    """
    # 确保 loss_history 结构正确
    if "frame_losses" not in loss_history:
        loss_history["frame_losses"] = []

    # 记录 frame_loss
    if frame_loss is not None:
        if iteration is None:
            # 如果没有传入 iteration，则直接 append
            loss_history["frame_losses"].append(frame_loss)
        else:
            while len(loss_history["frame_losses"]) <= iteration:
                loss_history["frame_losses"].append(None)
            loss_history["frame_losses"][iteration] = frame_loss


def plot_losses(output_dir, num_iterations):
    """
    绘制 frame_loss（按iteration）。
    """

    # -------------------- 绘制 frame_loss --------------------
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(loss_history["frame_losses"])), loss_history["frame_losses"], marker='o', label="Frame loss")
    plt.xlabel("Iterations")
    plt.ylabel("frame loss")
    plt.title("Frame loss over Iterations")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "frame_loss_plot.png"))
    plt.close()

    # 保存 Loss 到 txt 文件
    l2_loss_path = os.path.join(output_dir, "frame_losses.txt")
    with open(l2_loss_path, "w") as f:
        f.write("Iteration\tFrame_loss\n")
        for iter_idx in range(len(loss_history["frame_losses"])):
            f.write(f"{iter_idx}\t{loss_history['frame_losses'][iter_idx]:.6f}\n")
    
    print(f"Frame Loss plot saved to {os.path.join(output_dir, 'frame_loss_plot.png')}")


        
def plot_youngs_modulus(output_dir, youngs_mean_list, youngs_max_list, youngs_min_list):
    # 写入数据到txt文件
    with open(f"{output_dir}/youngs_modulus_values.txt", "w") as file:
        file.write("Iteration\tYoungs Mean\tYoungs Max\tYoungs Min\n")
        for i, (mean, max_, min_) in enumerate(zip(youngs_mean_list, youngs_max_list, youngs_min_list)):
            file.write(f"{i}\t{mean:.6f}\t{max_:.6f}\t{min_:.6f}\n")
    
    # 绘制折线图
    plt.figure(figsize=(12, 6))
    plt.plot(youngs_mean_list, label='Youngs Mean', marker='o', linestyle='-', markersize=5)
    plt.plot(youngs_max_list, label='Youngs Max', marker='x', linestyle='--', markersize=5)
    plt.plot(youngs_min_list, label='Youngs Min', marker='s', linestyle='-.', markersize=5)

    plt.title('Youngs Modulus Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Youngs Modulus')
    plt.legend()
    plt.grid(axis='y')  # 仅显示水平网格线
    plt.xticks(
        np.arange(0, len(youngs_mean_list), step=max(len(youngs_mean_list) // 10, 1))
    )  # 调整横坐标刻度，避免重叠
    plt.tight_layout()  # 调整布局，避免标签重叠

    # 保存图像
    plot_file_path = os.path.join(output_dir, "youngs_modulus_plot.png")
    plt.savefig(plot_file_path)
    plt.close()
    
    print(f"Youngs Modulus plot saved to {plot_file_path}")

def visualize_cuboids(selected_vertices, assignment, cuboid_centers, cuboid_sizes, output_html_path):
    """
    参数:
    - selected_vertices: np.ndarray, 点云顶点坐标 (N, 3)
    - assignment: np.ndarray, 顶点分配的 part_id (N,)
    - cuboid_centers: np.ndarray, Cuboid 中心点坐标 (num_cuboids, 3)
    - cuboid_sizes: np.ndarray, Cuboid 尺寸 (num_cuboids, 3)
    - output_html_path: str, 输出 HTML 文件路径
    """
    import plotly.graph_objects as go
    import numpy as np
    import random

    # 如果 assignment 为空，则所有点用同一种颜色；否则分配不同颜色
    if assignment is None:
        vertex_colors = 'blue'
    else:
        num_parts = assignment.max() + 1
        colors = generate_colors(num_parts)  # 你自己实现的颜色生成函数
        vertex_colors = [colors[idx] for idx in assignment]

    fig = go.Figure()

    # (1) 绘制点云
    fig.add_trace(
        go.Scatter3d(
            x=selected_vertices[:, 0],
            y=selected_vertices[:, 1],
            z=selected_vertices[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=vertex_colors,
                opacity=0.8
            ),
            name='Vertices'
        )
    )

    # (2) 可视化 cuboid (每个 Cuboid 作为一个完整的对象)
    for idx, (center, size) in enumerate(zip(cuboid_centers, cuboid_sizes)):
        color = f'rgb({random.randint(0,255)},{random.randint(0,255)},{random.randint(0,255)})'
        half_size = size / 2.0
        corners = np.array([
            [1, 1, 1],
            [1, 1, -1],
            [1, -1, 1],
            [1, -1, -1],
            [-1, 1, 1],
            [-1, 1, -1],
            [-1, -1, 1],
            [-1, -1, -1]
        ]) * half_size + center

        edges = [
            [0, 1], [1, 3], [3, 2], [2, 0],
            [4, 5], [5, 7], [7, 6], [6, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]

        # 将所有边合并成一个 trace
        cuboid_x, cuboid_y, cuboid_z = [], [], []
        for edge in edges:
            cuboid_x.extend([corners[edge[0], 0], corners[edge[1], 0], None])
            cuboid_y.extend([corners[edge[0], 1], corners[edge[1], 1], None])
            cuboid_z.extend([corners[edge[0], 2], corners[edge[1], 2], None])

        fig.add_trace(
            go.Scatter3d(
                x=cuboid_x,
                y=cuboid_y,
                z=cuboid_z,
                mode='lines',
                line=dict(color=color, width=2),
                name=f'Cuboid_{idx}',
                showlegend=True
            )
        )

    # (3) 布局美化
    fig.update_layout(
        scene=dict(
            aspectmode='data',
            xaxis=dict(backgroundcolor="white", gridcolor="lightgrey"),
            yaxis=dict(backgroundcolor="white", gridcolor="lightgrey"),
            zaxis=dict(backgroundcolor="white", gridcolor="lightgrey")
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=800,
        width=800,
        title="Vertices + Cuboids"
    )

    fig.write_html(output_html_path)
    # print(f"可视化结果已保存到 {output_html_path}")
    
    
def visualize_parts(ply_path, masks, output_dir):
    """
    将 ply 按不同 part_id 进行可视化，每个 part 使用不同颜色，并保存为可视化点云和 HTML。

    参数:
    - ply_path: str, 点云文件路径 (.ply)
    - masks: torch.Tensor, mask 张量 (num_parts, num_vertices)
    - output_dir: str, 保存输出文件的目录
    """
    os.makedirs(output_dir, exist_ok=True)
    output_html_path = os.path.join(output_dir, "part_colored_pointcloud.html")
    
    # 加载点云
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    num_vertices = points.shape[0]
    
    assert masks.shape[1] == num_vertices, "masks 和点云顶点数量不匹配"
    
    # 转换 masks 为 numpy 数组
    masks_np = masks.cpu().numpy()
    num_parts = masks_np.shape[0]
    
    # 生成颜色
    colors = generate_colors(num_parts)
    
    fig = go.Figure()
    
    # (1) 绘制点云
    for part_id in range(num_parts):
        part_mask = masks_np[part_id]
        selected_points = points[part_mask]

        # 不需要给每个点都传 color list，直接传一个颜色字符串即可
        color_str = colors[part_id]

        fig.add_trace(
            go.Scatter3d(
                x=selected_points[:, 0],
                y=selected_points[:, 1],
                z=selected_points[:, 2],
                mode='markers',
                marker=dict(
                    size=3,
                    color=color_str,  # 这里直接给统一的颜色字符串
                    opacity=0.8
                ),
                name=f'Part {part_id}'
            )
        )
    
    # (2) 图例美化
    legend_html = "<h2>Point Cloud Part Visualization</h2><ul>"
    for part_id in range(num_parts):
        legend_html += (
            f'<li><span style="background-color: rgb({colors[part_id][0]}, '
            f'{colors[part_id][1]}, {colors[part_id][2]}); '
            f'width: 20px; height: 20px; display: inline-block;"></span> '
            f'Part {part_id}</li>'
        )
    legend_html += "</ul>"
    
    # (3) 布局美化
    fig.update_layout(
        scene=dict(
            aspectmode='data', 
            xaxis=dict(backgroundcolor="white", gridcolor="lightgrey"),
            yaxis=dict(backgroundcolor="white", gridcolor="lightgrey"),
            zaxis=dict(backgroundcolor="white", gridcolor="lightgrey")
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=800,
        width=1000,
        title="Parts Visualization"
    )
    
    # 保存 HTML 文件
    fig.write_html(output_html_path, include_plotlyjs='cdn', full_html=True)
    
    # print(f"可视化结果已保存到 {output_html_path}")
    
    
def visualize_youngs_modulus(
    positions: torch.Tensor,           
    young_values: torch.Tensor,        
    ply_file_path: str,
    html_file_path: str,
    csv_file_path: str,  # 新增 CSV 保存路径
    colormap_name: str = 'viridis'
):
    """
    将粒子坐标 + Young 值同时保存为:
      (1) 带颜色的 .ply 文件
      (2) 可交互的 .html (Plotly)
      (3) 保存 Young's modulus 到 CSV 文件
    """
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import plotly.graph_objects as go
    
    os.makedirs(os.path.dirname(ply_file_path), exist_ok=True)
    os.makedirs(os.path.dirname(html_file_path), exist_ok=True)
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

    # --- 准备数据到 CPU ---
    positions_np = positions.clone().detach().cpu().numpy()
    young_np = young_values.clone().detach().cpu().numpy()
    N = len(positions_np)
    
    # --- 归一化 Young 值 ---
    y_min, y_max = young_np.min(), young_np.max()
    y_range = max(y_max - y_min, 1e-8)
    normalized_y = (young_np - y_min) / y_range
    
    # --- 颜色映射 (matplotlib) ---
    cmap = cm.get_cmap(colormap_name)  
    rgba_colors = cmap(normalized_y)   
    rgb_colors = (rgba_colors[:, :3] * 255).astype(np.uint8)  
    
    # ========== (1) 写 PLY 文件 ==========
    with open(ply_file_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for i in range(N):
            x, y, z = positions_np[i]
            r, g, b = rgb_colors[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
    # print(f"[Info] Color-coded .ply saved to: {ply_file_path}")

    # ========== (2) 用 Plotly 保存 HTML ==========
    colors_str = [
        f"rgb({r},{g},{b})" for (r, g, b) in rgb_colors
    ]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=positions_np[:, 0],
            y=positions_np[:, 1],
            z=positions_np[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color=colors_str,
                opacity=0.8
            ),
            name='Particles'
        )
    )
    fig.update_layout(
        scene=dict(
            aspectmode='data', 
            xaxis=dict(backgroundcolor="white", gridcolor="lightgrey"),
            yaxis=dict(backgroundcolor="white", gridcolor="lightgrey"),
            zaxis=dict(backgroundcolor="white", gridcolor="lightgrey")
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=800,
        width=1000,
        title="Young's Modulus Visualization"
    )
    fig.write_html(html_file_path, include_plotlyjs='cdn', full_html=True)
    # print(f"[Info] Interactive .html saved to: {html_file_path}")

    # ========== (3) 保存 CSV 文件 ==========
    df = pd.DataFrame({
        'X': positions_np[:, 0],
        'Y': positions_np[:, 1],
        'Z': positions_np[:, 2],
        'Youngs_Modulus': young_np
    })
    df.to_csv(csv_file_path, index=False)
    # print(f"[Info] Young's modulus values saved to: {csv_file_path}")