import os
import numpy as np
import pandas as pd
import open3d as o3d
import trimesh
import torch
import glob
import pdb
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import random
from typing import List
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree, ConvexHull
from scipy.spatial.distance import cdist
import point_cloud_utils as pcu

loss_history = {
    "frame_losses": [],  # 改为列表
    "l2_losses": [],  # 改为列表
}

def smooth_clip_gradients(param, min_val=1e-11, max_val=1e-5):
    if param.grad is not None:
        grad = param.grad
        grad_abs = grad.abs()

        # 1️⃣ 软上界裁剪（防止梯度过大）
        grad_scaled = max_val * torch.tanh(grad / max_val)

        # 2️⃣ 软下界裁剪（防止梯度过小）
        grad_adjusted = torch.where(grad_abs < min_val, min_val * grad.sign(), grad_scaled)

        # 3️⃣ 更新梯度
        param.grad.data = grad_adjusted
        
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
        raise ValueError("顶点数和权重矩阵的顶点数不匹配。")
    
    # 根据最大权重直接分配顶点到骨骼部分
    part_assignments = np.argmax(weights, axis=1)
    
    return part_assignments

def match_point(sim_path, sim_mask_path, threshold=1e-4, device="cuda"):
    
    # 加载点云数据并转换为 float32
    sim_points = torch.tensor(pcu.load_mesh_v(sim_path), dtype=torch.float32).to(device)
    mask_points = torch.tensor(pcu.load_mesh_v(sim_mask_path), dtype=torch.float32).to(device)

    print(f"sim_points: {sim_points.shape[0]}")
    print(f"mask_points: {mask_points.shape[0]}")

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
    """
    计算多个 mesh 的整体 bounding box。
    
    参数:
        meshes (list of torch.Tensor): 每个元素是一个 mesh (N, 3)，表示 N 个点的坐标 (x, y, z)。
        
    返回:
        bbox (dict): 包含 x, y, z 方向的最小值和最大值。
    """
    if not meshes:
        raise ValueError("meshes 列表为空，无法计算 bounding box。")
    
    # 确保 global_min 和 global_max 在与 mesh 相同的设备上
    device = meshes[0].device if meshes else torch.device('cpu')
    global_min = torch.tensor([float('inf'), float('inf'), float('inf')], device=device)
    global_max = torch.tensor([float('-inf'), float('-inf'), float('-inf')], device=device)
    
    # 遍历所有 mesh
    for mesh in meshes:
        if mesh.numel() == 0:
            continue  # 跳过空的 mesh
        
        # 计算当前 mesh 的最小值和最大值
        min_vals = mesh.min(dim=0).values
        max_vals = mesh.max(dim=0).values
        
        # 更新全局最小值和最大值
        global_min = torch.minimum(global_min, min_vals)
        global_max = torch.maximum(global_max, max_vals)
    
    # 返回 bounding box
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
        
# def check_gradients(frame_idx, save_dir, *params):
    
#     os.makedirs(save_dir, exist_ok=True)  # 确保保存路径存在

#     for name, param in params:
#         if param.grad is None:
#             print(f"Gradient of {name} is None.")
#             continue

#         # 获取梯度数据
#         grad = param.grad.clone().detach().cpu().numpy()
#         grad_abs = np.abs(grad.flatten())  # 计算绝对值

#         # 计算统计指标
#         grad_mean = np.mean(grad_abs)
#         grad_median = np.median(grad_abs)
#         grad_min, grad_max = np.min(grad_abs), np.max(grad_abs)
#         non_zero_count = np.count_nonzero(grad_abs)

#         # 计算 90% / 80% / 20% / 10% 的分位数范围
#         percentile_90 = np.percentile(grad_abs, 90)
#         percentile_80 = np.percentile(grad_abs, 80)
#         percentile_20 = np.percentile(grad_abs, 20)
#         percentile_10 = np.percentile(grad_abs, 10)

#         # 统计直方图数据（100个区间）
#         hist, bin_edges = np.histogram(grad_abs, bins=100, density=True)

#         # 打印梯度统计信息
#         print(f"[Frame {frame_idx}] Gradient of {name}:")
#         print(f"  Shape: {grad.shape}")
#         print(f"  Abs Min: {grad_min:.5e}, Abs Max: {grad_max:.5e}, Mean: {grad_mean:.5e}, Median: {grad_median:.5e}")
#         print(f"  90% of gradients are below: {percentile_90:.5e}")
#         print(f"  80% of gradients are below: {percentile_80:.5e}")
#         print(f"  20% of gradients are below: {percentile_20:.5e}")
#         print(f"  10% of gradients are below: {percentile_10:.5e}")
#         print(f"  Non-zero elements: {non_zero_count}/{grad.size}")
#         print("-" * 50)

#         # **保存分析结果**
#         analysis_filename = os.path.join(save_dir, f"grad_analysis_frame_{frame_idx}.csv")
#         df = pd.DataFrame({
#             "frame": [frame_idx],
#             "param": [name],
#             "grad_mean": [grad_mean],
#             "grad_median": [grad_median],
#             "grad_min": [grad_min],
#             "grad_max": [grad_max],
#             "percentile_90": [percentile_90],
#             "percentile_80": [percentile_80],
#             "percentile_20": [percentile_20],
#             "percentile_10": [percentile_10],
#             "non_zero_count": [non_zero_count],
#             "total_count": [grad.size]
#         })
#         df.to_csv(analysis_filename, index=False)

#         print(f"Saved gradient analysis for {name} at {analysis_filename}")

#         # **优化可视化：使用折线图，类似正态分布**
#         plt.figure(figsize=(7, 5))
#         sns.set(style="whitegrid")
        
#         # 计算每个 bin 的中心点
#         bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
#         # 画折线图
#         plt.plot(bin_centers, hist, linestyle="-", marker="o", markersize=2, color="b", alpha=0.7, label="Gradient Distribution")
        
#         # 画分位数
#         plt.axvline(percentile_90, color='r', linestyle="--", label="90%")
#         plt.axvline(percentile_80, color='g', linestyle="--", label="80%")
#         plt.axvline(percentile_20, color='purple', linestyle="--", label="20%")
#         plt.axvline(percentile_10, color='orange', linestyle="--", label="10%")

#         # 轴设置
#         plt.xlabel("Gradient Absolute Value")
#         plt.ylabel("Density")
#         plt.title(f"Gradient Distribution - {name} (Frame {frame_idx})")
#         plt.legend()
#         plt.xscale("log")  # 使用对数坐标
#         plt.grid(True, linestyle="--", alpha=0.6)

#         # 保存折线图
#         plot_filename = os.path.join(save_dir, f"grad_curve_frame_{frame_idx}.png")
#         plt.savefig(plot_filename)
#         plt.close()

#         print(f"Saved gradient curve for {name} at {plot_filename}")
    

def get_gradient_percentiles(param, percentile_high=90, percentile_low=10):
    """
    计算给定参数梯度的高分位数（默认 90%）和低分位数（默认 10%）

    Args:
        param (torch.nn.Parameter): 需要计算梯度的 PyTorch 参数
        percentile_high (int, optional): 高分位数，默认 90%
        percentile_low (int, optional): 低分位数，默认 10%

    Returns:
        tuple: (percentile_high_value, percentile_low_value)
    """
    if param.grad is None:
        print("Gradient is None.")
        return None, None

    # 获取梯度数据
    grad = param.grad.clone().detach().cpu().numpy()
    grad_abs = np.abs(grad.flatten())  # 计算绝对值

    # 计算分位数
    percentile_high_value = np.percentile(grad_abs, percentile_high)
    percentile_low_value = np.percentile(grad_abs, percentile_low)

    return percentile_high_value, percentile_low_value

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
        # 扩展列表长度以适应当前 iteration
        while len(loss_history["frame_losses"]) <= iteration:
            loss_history["frame_losses"].append(None)
        
        loss_history["frame_losses"][iteration] = frame_loss


def plot_losses(output_dir, num_iterations):
    """
    绘制 frame_loss（按iteration）。
    """
    # 创建保存loss图像和txt文件的目录
    plot_dir = os.path.join(output_dir, "loss_plots")
    txt_dir = os.path.join(output_dir, "loss_txt")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(txt_dir, exist_ok=True)

    # -------------------- 绘制 frame_loss --------------------
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(loss_history["frame_losses"])), loss_history["frame_losses"], marker='o', label="Frame loss")
    plt.xlabel("Iterations")
    plt.ylabel("frame loss")
    plt.title("Frame loss over Iterations")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "frame_loss_plot.png"))
    plt.close()

    # 保存 l2 Loss 到 txt 文件
    l2_loss_path = os.path.join(txt_dir, "frame_losses.txt")
    with open(l2_loss_path, "w") as f:
        f.write("Iteration\tFrame_loss\n")
        for iter_idx in range(len(loss_history["frame_losses"])):
            f.write(f"{iter_idx}\t{loss_history['frame_losses'][iter_idx]:.6f}\n")
    
    print(f"Frame Loss plot saved to {os.path.join(plot_dir, 'frame_loss_plot.png')}")


        
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
    print(f"可视化结果已保存到 {output_html_path}")
    
    
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
    
    print(f"可视化结果已保存到 {output_html_path}")
    
    
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
    print(f"[Info] Color-coded .ply saved to: {ply_file_path}")

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
    print(f"[Info] Interactive .html saved to: {html_file_path}")

    # ========== (3) 保存 CSV 文件 ==========
    df = pd.DataFrame({
        'X': positions_np[:, 0],
        'Y': positions_np[:, 1],
        'Z': positions_np[:, 2],
        'Youngs_Modulus': young_np
    })
    df.to_csv(csv_file_path, index=False)
    print(f"[Info] Young's modulus values saved to: {csv_file_path}")