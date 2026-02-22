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
from scipy.spatial import cKDTree, ConvexHull, KDTree
from scipy.spatial.distance import cdist
import point_cloud_utils as pcu


def get_points_in_cuboid_sphere(sim_xyzs: torch.Tensor, 
                                cuboid_centers: torch.Tensor, 
                                cuboid_sizes: torch.Tensor) -> list:
    """
    Args:
        sim_xyzs: [N, 3] 模拟点坐标
        cuboid_centers: [M, 3] 每个 cuboid 的中心点
        cuboid_sizes: [M, 3] 每个 cuboid 的尺寸 (长、宽、高)
    Returns:
        indices_list: 长度为 M 的列表，每个元素是一个 tensor，包含在对应 cuboid 包裹的球内的 sim_xyzs 点的索引
    """
    indices_list = []
    # 遍历每个 cuboid
    for i in range(cuboid_centers.shape[0]):
        center = cuboid_centers[i]  # [3]
        size = cuboid_sizes[i]      # [3]
        # 计算球形半径：cuboid 对角线的一半
        radius = 0.5 * torch.sqrt(torch.sum(size ** 2))
        # 计算所有点到该中心的欧氏距离
        distances = torch.norm(sim_xyzs - center, dim=1)
        # 找出距离小于等于半径的点的索引
        indices = torch.nonzero(distances <= radius, as_tuple=False).squeeze()
        indices_list.append(indices)
        
    return indices_list

def get_knn_neighbors(positions, k=6):
    """
    获取每个点的 k 近邻索引
    positions: Tensor (N, 3)，粒子位置
    k: int，邻居数量
    返回:
    neighbor_indices: List[List[int]]，每个粒子的邻居索引
    """
    positions_np = positions.cpu().numpy()  # 转换为 NumPy 进行 KDTree 计算
    tree = KDTree(positions_np)
    _, indices = tree.query(positions_np, k=k+1)  # 查询 k+1 近邻（包括自己）
    
    # 去掉自身索引
    neighbor_indices = [inds[1:].tolist() for inds in indices]
    
    return neighbor_indices

def compute_smoothness_loss(youngs_modulus, neighbor_indices, positions):
    """
    计算基于 Laplacian 的 Young's modulus 平滑损失
    youngs_modulus: Tensor (N,) 每个粒子的 Young’s modulus
    neighbor_indices: List[List[int]] 每个粒子的相邻粒子索引
    """
    smooth_loss = 0.0
    num_valid_particles = 0

    for i, neighbors in enumerate(neighbor_indices):
        if len(neighbors) > 0:
            neighbor_values = youngs_modulus[neighbors]
            
            # 计算权重（基于欧几里得距离）
            distances = torch.norm(positions[neighbors] - positions[i], dim=1)
            weights = torch.exp(-distances)  # 权重越大，距离越小
            
            weighted_mean = torch.sum(weights * neighbor_values) / torch.sum(weights)
            
            smooth_loss += torch.norm(youngs_modulus[i] - weighted_mean)  # L2 平滑
            num_valid_particles += 1

    if num_valid_particles > 0:
        smooth_loss = smooth_loss / num_valid_particles

    return smooth_loss

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

