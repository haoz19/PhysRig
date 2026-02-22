import os
import numpy as np
import torch
import pdb
import plotly.graph_objects as go
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_youngs_weights(points: torch.Tensor, sample_indices: torch.Tensor, output_dir: str, sigma: float, small_thresh: float) -> torch.Tensor:

    sample_positions = points[sample_indices]

    distances = torch.cdist(points, sample_positions, p=2)

    sigma = sigma # 高斯核衰减程度

    raw_weights = torch.exp(- (distances ** 2) / (2 * sigma ** 2))

    small_thresh = small_thresh
    
    valid_mask = (raw_weights >= small_thresh).float()
    masked_weights = raw_weights * valid_mask  

    weight_sum = masked_weights.sum(dim=1, keepdim=True)

    weight_sum = torch.where(weight_sum == 0, torch.ones_like(weight_sum), weight_sum)

    normalized_weights = masked_weights / weight_sum    

    non_zero_weights_count = (normalized_weights > 0).sum(dim=1)

    max_count = torch.max(non_zero_weights_count)
    min_count = torch.min(non_zero_weights_count)

    # print(f"最大值: {max_count.item()}, 最小值: {min_count.item()}")
    
    # os.makedirs(output_dir, exist_ok=True)

    # # (a) 保存 normalized_weights[0]
    # point0_normalized_file = os.path.join(output_dir, "point0_normalized_weights.txt")
    # np.savetxt(
    #     point0_normalized_file,
    #     normalized_weights[0].detach().cpu().numpy(),
    #     fmt="%.6f"
    # )
    # print(f"[Info] 第 0 号粒子的 normalized_weights 已保存到 {point0_normalized_file}")

    # # (b) 保存 masked_weights[0]
    # point0_masked_file = os.path.join(output_dir, "point0_masked_weights.txt")
    # np.savetxt(
    #     point0_masked_file,
    #     masked_weights[0].detach().cpu().numpy(),
    #     fmt="%.6f"
    # )
    # print(f"[Info] 第 0 号粒子的 masked_weights 已保存到 {point0_masked_file}")

    
    return normalized_weights

def farthest_point_sampling(points: torch.Tensor, k: int) -> torch.Tensor:
    """
    points: [N, 3] tensor, 点云
    k: 要采样的点的数量
    返回: 采样点的索引，形状 [k]
    """
    N, _ = points.shape
    device = points.device

    # 初始化采样索引数组
    sampled_indices = torch.zeros(k, dtype=torch.long, device=device)
    # 初始化每个点到采样集的最小距离，大值初始化
    distances = torch.full((N,), float("inf"), device=device)
    
    # 随机选择第一个点（为了确定性，可以固定种子）
    initial_index = torch.randint(0, N, (1,), device=device)

    sampled_indices[0] = initial_index
    centroid = points[initial_index].squeeze(0)
    
    # 更新所有点到采样点集合的距离
    dist = torch.norm(points - centroid, dim=1)
    distances = torch.min(distances, dist)
    
    for i in range(1, k):
        # 选择距离当前采样点集合最远的点
        farthest_index = torch.argmax(distances)
        sampled_indices[i] = farthest_index
        
        # 更新距离，每个点到采样集的最小距离更新为当前与新加入点的距离的最小值
        new_point = points[farthest_index]
        dist = torch.norm(points - new_point, dim=1)
        distances = torch.min(distances, dist)
    
    return sampled_indices

def plot_youngs_distribution(csv_file, output_dir, youngs_column='Youngs_Modulus', bins=50):
    """
    从csv_file中读取Young's Modulus值，并绘制直方图+KDE曲线。
    
    参数：
        csv_file (str): CSV 文件路径
        output_dir (str): 输出文件目录
        youngs_column (str): CSV 中存储 Young's 值的列名
        bins (int): 直方图 bin 的数量
    """
    # 1. 读取CSV
    df = pd.read_csv(csv_file)
    
    # 2. 提取 Young's Modulus 数据
    if youngs_column not in df.columns:
        raise ValueError(f"列名 '{youngs_column}' 在 CSV 文件中不存在，请检查文件列名。")
    
    youngs_data = df[youngs_column].values

    # 3. 生成 PNG 文件路径
    output_png = os.path.join(output_dir, "youngs_distribution.png")
    
    # 4. 绘制直方图和 KDE 曲线
    plt.figure(figsize=(8, 5))
    sns.histplot(youngs_data, bins=bins, kde=True, color='blue', edgecolor='black')
    
    # 5. 画面设置
    plt.title("Distribution of Young's Modulus")
    plt.xlabel("Young's Modulus")
    plt.ylabel("Frequency")
    
    # 6. 保存到文件
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    # print(f"[Info] Young's modulus distribution saved to {output_png}")
    plt.close()

    
def visualize_sampled_points(
    positions: torch.Tensor,       # 所有粒子的位置，shape: [N, 3]
    sample_indices: torch.Tensor,  # 采样点的索引，shape: [M]
    output_dir: str                # 输出目录，例如 self.output_dir
):
    """
    将所有粒子点保存为带颜色的 .ply 文件和交互式 .html 文件：
      - 采样的点固定为红色 (rgb: 255,0,0)
      - 其他点固定为淡蓝色 (rgb: 173,216,230)
    """
    
    if not isinstance(positions, torch.Tensor):
        positions = torch.tensor(positions, dtype=torch.float32)
        
        
    # 构造输出文件路径
    ply_file_path = os.path.join(output_dir, "sampled_points.ply")
    html_file_path = os.path.join(output_dir, "sampled_points.html")
    
    os.makedirs(os.path.dirname(ply_file_path), exist_ok=True)
    os.makedirs(os.path.dirname(html_file_path), exist_ok=True)
    
    # 将数据搬到 CPU 上，并转换为 numpy 数组
    positions_np = positions.clone().detach().cpu().numpy()
    N = positions_np.shape[0]
    
    # 构建颜色数组：所有点先赋为淡蓝色，采样点赋为红色
    # 淡蓝色 RGB: (173, 216, 230)
    # 红色 RGB: (255, 0, 0)
    colors = np.tile(np.array([173, 216, 230], dtype=np.uint8), (N, 1))
    
    # 将采样点的颜色修改为红色
    sample_indices_np = sample_indices.clone().detach().cpu().numpy()
    colors[sample_indices_np] = np.array([255, 0, 0], dtype=np.uint8)
    
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
            r, g, b = colors[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
    # print(f"[Info] Color-coded .ply saved to: {ply_file_path}")

    # ========== (2) 用 Plotly 保存 HTML ==========
    # 为了分层显示，先画出所有点（淡蓝色），再叠加采样点（红色）
    # 生成所有点的颜色字符串
    color_all = "rgb(173,216,230)"
    color_sample = "rgb(255,0,0)"
    
    # 提取采样点的坐标
    sampled_positions = positions_np[sample_indices_np]
    
    fig = go.Figure()
    # 添加所有点（淡蓝色）
    fig.add_trace(
        go.Scatter3d(
            x=positions_np[:, 0],
            y=positions_np[:, 1],
            z=positions_np[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=color_all,
                opacity=0.6
            ),
            name='All Points'
        )
    )
    # 添加采样点（红色）
    fig.add_trace(
        go.Scatter3d(
            x=sampled_positions[:, 0],
            y=sampled_positions[:, 1],
            z=sampled_positions[:, 2],
            mode='markers',
            marker=dict(
                size=6,
                color=color_sample,
                opacity=1.0
            ),
            name='Sampled Points'
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
        title="Visualization of Sampled Points"
    )
    fig.write_html(html_file_path, include_plotlyjs='cdn', full_html=True)
    # print(f"[Info] Interactive .html saved to: {html_file_path}")
