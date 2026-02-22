
'''
Haolan：

单个youngs值能够有效学习，每个点的youngs值无法有效学习

基于随机选取的点插值全局youngs

只测试速度训练

以sp为最终版

'''
     
import argparse
import time
import os
import torch
from tqdm import tqdm

from torch import Tensor
from jaxtyping import Float, Int, Shaped
from typing import List

import point_cloud_utils as pcu

import open3d as o3d
import numpy as np
import logging
import argparse
import shutil
import wandb
import pdb
import trimesh
import math

import sys
sys.path.append("/taiga/illinois/eng/ece/n-ahuja/haozhang/tjx/PHYSDREAMER/PhysDreamer")

from motionrep.utils.config import create_config
from motionrep.utils.optimizer import get_linear_schedule_with_warmup
from omegaconf import OmegaConf
from PIL import Image
import imageio
from chamferdist import ChamferDistance # Haolan:chamferdist lib

# from motionrep.utils.torch_utils import get_sync_time
from einops import rearrange, repeat



from typing import NamedTuple
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import gradcheck

from thirdparty_code.warp_mpm.mpm_data_structure import (
    MPMStateStruct,
    MPMModelStruct,
    get_float_array_product,
)
from thirdparty_code.warp_mpm.mpm_solver_diff_cuboid import MPMWARPDiff
from thirdparty_code.warp_mpm.warp_utils import from_torch_safe
import warp as wp
import random

from exp_motion.train.local_utils import (
    get_volume,
    create_spatial_fields,
)

# Haolan：整合新加入的函数
from exp_motion.train.new_utils import (
    set_random_seed,
    load_mesh_sequences,
    vertice_assignment,
    match_point,
    calculate_bounding_box,
    check_gradients,
    save_ply,
    log_losses, 
    plot_losses,
    plot_youngs_modulus,
    visualize_cuboids,
    visualize_parts,
    visualize_youngs_modulus,
)

from exp_motion.train.new2_utils import (
    smooth_clip_gradients,
    get_knn_neighbors,
    compute_smoothness_loss,
    get_gradient_percentiles,
    get_points_in_cuboid_sphere,
)

from exp_motion.train.new3_utils import (
    get_youngs_weights,
    farthest_point_sampling,
    visualize_sampled_points,
    plot_youngs_distribution,
)

from exp_motion.train.cuboid_utils import (
    cuboid_finding,
    assign_cuboid_velocity,
)


from exp_motion.train.interface import (
    MPMDifferentiableSimulationRig,
)

class Trainer:
    def __init__(self, args):
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.args = args
        
        self.num_frames = int(args.num_frames)
        
        self.window_size = self.num_frames
        
        set_random_seed(42)
        
        # output
        args.wandb_name += (
            "vf_{}_SP_{}_youngs_{}_lr_{}_substep_{}_iters_{}_sw_{}".format(
                args.velo_factor,
                args.sample_particles,
                args.youngs,
                args.lr,
                args.substep,
                args.train_iters,
                self.window_size,
            )
        )
        
        # path
        self.output_dir = os.path.join(args.output_dir, args.wandb_name)
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        dataset_dir = args.dataset_dir
        
        self.dataset_dir = dataset_dir
        
        gtmesh_dir = os.path.join(dataset_dir, "output")
        
        
        self.sim_path = os.path.join(dataset_dir, "infilled/infilled_0.ply")
        
        # gt and sim
        self.gt_meshes = load_mesh_sequences(gtmesh_dir)
        
        
        self.xyzs = pcu.load_mesh_v(self.sim_path)
        
        self.num_particles = torch.tensor(self.xyzs, dtype=torch.float32, device=self.device).shape[0]

        # setup simulation
        
        self.delta_time = 1/25
        
        self.sample_particles = args.sample_particles
        
        # 不再需要skin_weights，因为不使用mask
        
        self.velo_factor = args.velo_factor
        
        E_nu_list = self.init_trainable_params()
        
        self.E_nu_list = E_nu_list
        
        self.setup_simulation(dataset_dir, grid_size=args.grid_size)
        
        self.youngs_modulus_min = []
        self.youngs_modulus_mean = []
        self.youngs_modulus_max = []
        
        
        visualize_sampled_points(self.xyzs, self.sample_indices, self.output_dir)
        
        # setup training
        
        self.train_iters = args.train_iters
        
        self.trainable_params = self.cuboid_velocity
        
        self.velocity_optimizer = torch.optim.SGD(
            self.cuboid_velocity,
            lr=args.lr, 
            momentum=0.0,  # 添加动量项，提高收敛速度
            weight_decay=0.0,  # 权重衰减
        )
        
        # self.velocity_optimizer = torch.optim.AdamW(
        #     self.cuboid_velocity,
        #     lr=args.lr,
        #     weight_decay=0.0,
        # )
        
        self.velocity_scheduler = get_linear_schedule_with_warmup(
            optimizer=self.velocity_optimizer,
            num_warmup_steps=args.warmup_step,
            num_training_steps=args.train_iters,
        )

        self.step = 0
        
        self.loss_array = np.full((self.train_iters, self.num_frames - 1), np.nan)
        
         #保存每帧的最终结果
        self.final_particle_positions = []  # 每帧的最终粒子位置
        self.final_cuboid_positions = []    # 每帧的最终cuboid位置
        self.final_losses = []              # 每帧的最终loss

        self.iter_material = args.iter_material

        self.max_grad_norm = args.max_grad_norm
        
        # 可视化：重叠判断放大倍数
        self.overlap_scale = getattr(args, 'overlap_scale', 5.5)
  
        
    def init_trainable_params(self,):

        sim_points = torch.tensor(self.xyzs, dtype=torch.float32, device=self.device)
        
        self.sample_indices = farthest_point_sampling(sim_points, self.sample_particles)
        
        youngs_modulus = nn.Parameter(
            torch.ones(self.sample_particles, device=self.device, dtype=torch.float32) * self.args.youngs, 
            requires_grad=False
        )
        
        # 简化Young模量初始化，不使用mask
        print(f"初始化 {self.sample_particles} 个采样点的Young模量，统一值为 {self.args.youngs}")
        
        poisson_ratio = torch.ones(
            self.num_particles, device=self.device, dtype=torch.float32) * args.nu

        trainable_params = [youngs_modulus, poisson_ratio]

        print(
            "init young modulus: ",
            args.youngs,
            "poisson ratio: ",
            args.nu,
        )
        return trainable_params

    
    def setup_simulation(self, dataset_dir, grid_size=80):
        
        device = self.device
 
        sim_xyzs = torch.tensor(self.xyzs, dtype=torch.float32, device=device)
        
        sim_cov = torch.eye(3).expand(len(sim_xyzs), 3, 3).to(device)
        
        # scale, and shift
        pos_max = sim_xyzs.max()
        pos_min = sim_xyzs.min()
        scale = (pos_max - pos_min) * 1.8
        shift = -pos_min + (pos_max - pos_min) * 0.25
        self.scale, self.shift = scale, shift
        print("scale, shift", scale, shift)
        
        sim_xyzs = (sim_xyzs + shift) / scale
        self.gt_meshes= [
            (mesh + shift) / scale for mesh in self.gt_meshes]
        
        # 保存归一化后的GT用于逐帧训练
        self.gt_meshes_original = self.gt_meshes
        
        bbox = calculate_bounding_box(sim_xyzs)

        # 获取最大绝对值
        max_abs_value = max(abs(bbox['x_min']), abs(bbox['x_max']), 
                            abs(bbox['y_min']), abs(bbox['y_max']), 
                            abs(bbox['z_min']), abs(bbox['z_max']))
        
        # 计算points_volume
        points_volume = get_volume(sim_xyzs.detach().cpu().numpy())

        wp.init()
        wp.config.mode = "debug"
        wp.config.verify_cuda = True
    
        
        # Haolan：grid_size和grid_lim对于模拟很重要，因为它能决定grid的大小
        grid_size = 150
        x_range = max(abs(bbox['x_min']), abs(bbox['x_max'])) * 2.0  
        y_range = max(abs(bbox['y_min']), abs(bbox['y_max'])) * 2.0  # y轴保持1.5倍
        z_range = max(abs(bbox['z_min']), abs(bbox['z_max'])) * 2.0  # z轴保持1.5倍
        grid_lim = max(x_range, y_range, z_range)  # 取最大值作为网格边界
        grid_dx = grid_lim / grid_size
        print(f"X range: {x_range:.3f}m, Y range: {y_range:.3f}m, Z range: {z_range:.3f}m")
        
         # Haolan：创建gt_mask
        gt_mesh_initial = os.path.join(dataset_dir, "skeleton/gt_0.ply")
        pcd = o3d.io.read_point_cloud(gt_mesh_initial)
        gt_vertices = np.asarray(pcd.points)
        # 保证 shift/scale 是 numpy 数组
        shift_np = shift.cpu().numpy() if hasattr(shift, 'cpu') else np.array(shift)
        scale_np = scale.cpu().numpy() if hasattr(scale, 'cpu') else np.array(scale)
        preprocessed_vertices = (gt_vertices + shift_np) / scale_np
        
        
        
        cuboid_centers, cuboid_sizes, cuboid_types = cuboid_finding(preprocessed_vertices, grid_dx, vertices_assignment=None, mesh=None)
        
    
        # 检查 cuboid_finding 是否成功找到立方体
        if len(cuboid_centers) == 0 or len(cuboid_sizes) == 0:
            raise ValueError("cuboid_finding 未能找到有效的 cuboid，请检查输入顶点。")

            
        cuboid_centers_np = np.asarray(cuboid_centers, dtype=np.float32)
        cuboid_sizes_np = np.asarray(cuboid_sizes, dtype=np.float32)
        self.cuboid_point = torch.from_numpy(cuboid_centers_np).to(device)
        self.cuboid_size = torch.from_numpy(cuboid_sizes_np).to(device)
        
        # print("========== Cuboid DEBUG INFO ==========")
        # for i in range(len(cuboid_centers)):
        #     print(f"Cuboid {i}:")
        #     print("  Center:", cuboid_centers[i])
        #     print("  Size  :", cuboid_sizes[i])
        #     bp = boundary_points_list[i]
        #     print("  Boundary points shape:", bp.shape)
        #     print("  Some boundary points (first 5):", bp[:5])
        # print("=========================================")
        
      
        
        
        # 删除part可视化，因为不再使用mask
        
        # pdb.set_trace()

        # cuboid可视化
        output_html_path = os.path.join(self.output_dir, "cuboid_visualization.html")

        # 直接使用已有的grid信息
        grid_info = {
            'n_grid': grid_size,
            'grid_lim': grid_lim,
            'dx': grid_dx,
            'grid_dim_x': grid_size,
            'grid_dim_y': grid_size,
            'grid_dim_z': grid_size
        }
        

        # 调用函数（包含grid可视化）
        visualize_cuboids(
            selected_vertices=preprocessed_vertices,
            assignment=None,  # 由于不再使用mask，设为None
            cuboid_centers=cuboid_centers,
            cuboid_sizes=cuboid_sizes,
            output_html_path=output_html_path,
            grid_info=grid_info,  # 新增grid参数
        )

        
        
        self.cuboid_velocity = assign_cuboid_velocity(
            cuboid_centers=cuboid_centers,
            cuboid_sizes=cuboid_sizes,
            cuboid_types=cuboid_types,
            total_frames = self.num_frames,
            gt_meshes=self.gt_meshes,
            vertices=preprocessed_vertices,
            delta_time=self.delta_time,
            device=device,
        )
        
        for i in range(len(self.cuboid_velocity)):
            self.cuboid_velocity[i].data *= self.velo_factor


        mpm_state = MPMStateStruct()
        mpm_state.init(self.num_particles, device=device, requires_grad=True)
        self.particle_init_position = sim_xyzs.clone()
        
        mpm_state.from_torch(
            self.particle_init_position.clone(),
            torch.from_numpy(points_volume).float().to(device).clone(),
            sim_cov,
            device=device,
            requires_grad=True,
            n_grid=grid_size, # xyz,分成多少个grid
            grid_lim=grid_lim,
        )
        mpm_model = MPMModelStruct()
        mpm_model.init(self.num_particles, device=device, requires_grad=True)
        mpm_model.init_other_params(n_grid=grid_size, grid_lim=grid_lim, device=device)

        material_params = {
            "material": "jelly",  # "jelly", "metal", "sand", "foam", "snow", "plasticine", "neo-hookean"
            "g": [0.0, 0.0, 0.0],
            "density": 800,  # kg / m^3
            "grid_v_damping_scale": 0.999,  # 0.999,
        }
        
        
        self.v_damping = material_params["grid_v_damping_scale"]
        
        self.material_name = material_params["material"]
        
        mpm_solver = MPMWARPDiff(
            self.num_particles, n_grid=grid_size, grid_lim=grid_lim, device=device
        )
        
        mpm_solver.set_parameters_dict(mpm_model, mpm_state, material_params)

        self.mpm_state, self.mpm_model, self.mpm_solver = (
            mpm_state,
            mpm_model,
            mpm_solver,
        )

        # density, youngs, poisson_ratio tensor
        density = (
            torch.ones_like(self.particle_init_position[..., 0])
            * material_params["density"]
        )

        self.density = density
        
        
        mpm_state.reset_density(
            density.clone(),
            torch.ones_like(density).type(torch.int),
            device,
            update_mass=True,
        )
        
        adaptive_sigma = 0.05 * max_abs_value
        
        small_thresh = 0.05
        
        self.normalized_weights = get_youngs_weights(self.particle_init_position, self.sample_indices, self.output_dir, adaptive_sigma, small_thresh)
        
        sample_youngs = self.E_nu_list[0].clone() 
        
        youngs_modulus = (self.normalized_weights * sample_youngs.unsqueeze(0)).sum(dim=1)
        
        mpm_solver.set_E_nu_from_torch(
            mpm_model, youngs_modulus, self.E_nu_list[1].clone(), device
        )
        
        mpm_solver.prepare_mu_lam(mpm_model, mpm_state, device)
        
        


    def get_simulation_input(self, device):
        """
        Outs: All padded
            density: [N]
            youngs_modulus: [N]
            poisson_ratio: [N]
            velocity: [N, 3]
        """

        density, youngs_modulus, poisson = self.get_material_params(device)
        
        initial_position = self.particle_init_position.clone()
    
        # 使用保存的初始速度（如果有），否则从0开始
        if hasattr(self, 'particle_init_velocity') and self.particle_init_velocity is not None:
            velocity = self.particle_init_velocity.clone()
        else:
            # 第一帧或没有保存的速度时，从0开始
            velocity = torch.zeros_like(initial_position, device=device)

        # 使用保存的初始F和C（如果有），否则初始化
        if hasattr(self, 'particle_init_F') and self.particle_init_F is not None:
            particle_F = self.particle_init_F.clone()
        else:
            # 第一帧或没有保存的F时，初始化为单位矩阵
            I_mat = torch.eye(3, dtype=torch.float32).to(device)
            particle_F = torch.repeat_interleave(
                I_mat[None, ...], initial_position.shape[0], dim=0
            )
        
        if hasattr(self, 'particle_init_C') and self.particle_init_C is not None:
            particle_C = self.particle_init_C.clone()
        else:
            # 第一帧或没有保存的C时，初始化为0
            particle_C = torch.zeros_like(particle_F)
        
        return (
            density,
            youngs_modulus,
            poisson,
            velocity,
            particle_F,
            particle_C,
        )

    def get_material_params(self, device):
        
        sample_youngs = self.E_nu_list[0]
        
        youngs_modulus = (self.normalized_weights * sample_youngs.unsqueeze(0)).sum(dim=1)

        density = self.density

        poisson = self.E_nu_list[1]

        return density, youngs_modulus, poisson
    
        
    def train_one_step(self):
        
        device = self.device
        
        iteration_start_time = time.time() # 用于记录时间
        
        # scheduler, start from 0
        window_size = self.window_size - 1
        
        print(f"Window size: {self.window_size}")    
        
        log_loss_dict = {
            "Frame_loss": [],
        }
            
        particle_pos = self.particle_init_position.clone()
        
        # clean grid, stress, F, C and rest initial position
        self.mpm_state.reset_state(
            particle_pos.clone(),
            None,
            None,  # .clone(),
            device=device,
            requires_grad=True,
        )
        self.mpm_state.set_require_grad(True)

        (
            density,
            youngs_modulus,
            poisson,
            particle_velo,
            particle_F,
            particle_C,
        ) = self.get_simulation_input(device)

        delta_time = self.delta_time 
        substep_size = delta_time / self.args.substep
        num_substeps = int(delta_time / substep_size)

        # 逐帧训练：每帧开始时重置cuboid_point到初始位置，确保单帧训练效果与原版一致
        if self.step == 0:
            self.initial_cuboid_point = self.cuboid_point.clone()
        self.cuboid_point = self.initial_cuboid_point.detach()
        
        frame_time_offset = 0.0
        
        frame_times = []
        
        # 逐帧训练时 window_size=1，循环只执行一次
        for start_time_idx in range(0, window_size):
            
            frame_start_time = time.time()
            
            end_time_idx = window_size  # 直接使用 window_size
            
            num_step_with_grad = num_substeps * (end_time_idx - start_time_idx)
            
            gt_frame = self.gt_meshes[start_time_idx + 1] # mesh sequences作为监督，下一帧作为上一帧模拟结果的监督
  
            if start_time_idx != 0:
                density, youngs_modulus, poisson = self.get_material_params(device)
            
                
            print(f"Processing frames from {start_time_idx} to {end_time_idx}")
            
            current_cuboid_velocity = self.cuboid_velocity[start_time_idx]
            
                
            # Haolan：一次调用中执行所有模拟步骤
            particle_pos, particle_velo, particle_F, particle_C, particle_cov = (
                MPMDifferentiableSimulationRig.apply( 
                    # MPMDifferentiableSimulationClean
                    self.mpm_solver,
                    self.mpm_state,
                    self.mpm_model,
                    substep_size,
                    num_step_with_grad,
                    particle_pos,
                    particle_velo,
                    particle_F,
                    particle_C,
                    youngs_modulus,
                    poisson,
                    current_cuboid_velocity,  # Haolan：传入当前frame的cuboid_velocity
                    self.cuboid_point, # Haolan：传入当前frame的point位置
                    self.cuboid_size, # Haolan：size是固定值，通过size_scale来优化
                    frame_time_offset, # 全局时间累积
                    density,
                    device,
                    True,
                )
            )
            
            # Haolan：更新cuboid_point位置，detach()和计算图断开
            self.cuboid_point = (self.cuboid_point + current_cuboid_velocity * delta_time).detach()
            
            # Haolan：累积全局时间
            frame_time_offset += delta_time
            
            
            # Haolan:loss 
            chamfer_dist = ChamferDistance()
            
            predicted_points = particle_pos.clone()  
            target_points = gt_frame
            
            predicted_points_cd = particle_pos.unsqueeze(0).clone()  
            target_points_cd = gt_frame.unsqueeze(0) 

            # global CD loss             
            global_loss = chamfer_dist(predicted_points_cd, target_points_cd)
            loss = global_loss * 100
            
            print(f"Frame Loss: {loss.item()}")

            loss.backward()
 

            particle_pos, particle_velo, particle_F, particle_C = (
                particle_pos.detach(),
                particle_velo.detach(),
                particle_F.detach(),
                particle_C.detach(),
            )

            frame_index = f"{start_time_idx}_{end_time_idx}"
            
            with torch.no_grad():
                log_loss_dict["Frame_loss"].append(loss.item())
            
            frame_end_time = time.time()
            frame_duration = frame_end_time - frame_start_time
            frame_times.append((
                start_time_idx, end_time_idx, frame_duration
            ))
        
            # pdb.set_trace()
    
        current_loss_vector = np.array(log_loss_dict["Frame_loss"])
        self.loss_array[self.step, self.current_frame_idx] = current_loss_vector[0]
        
        # check_gradients(
        #     ("Before self.E_nu_list[0]", self.E_nu_list[0])  # 位置参数
        # )
        
        # 不能一起裁剪，因为材质和速度的梯度范数差距很大
        
        torch.nn.utils.clip_grad_norm_(
            self.trainable_params,
            self.max_grad_norm,
            error_if_nonfinite=False,
        )  # error if nonfinite is false
        
        self.velocity_optimizer.step()
        torch.cuda.empty_cache()
        self.velocity_optimizer.zero_grad()
        self.velocity_scheduler.step()

        

        _, youngs_modulus, _ = self.get_material_params(device)
        
        youngs_modulus_data = youngs_modulus.clone().detach()
        
        self.youngs_modulus_min.append(youngs_modulus_data.min().item())
        self.youngs_modulus_mean.append(youngs_modulus_data.mean().item())
        self.youngs_modulus_max.append(youngs_modulus_data.max().item())
        
        if self.step == self.train_iters - 1:
            self.final_youngs_modulus = youngs_modulus_data
        
        for k, v in log_loss_dict.items():
            log_loss_dict[k] = np.mean(v)
            
        log_losses(
            frame_loss=log_loss_dict["Frame_loss"],
            iteration=self.step
        )

        print(f"Iteration {self.step}: Average Frame Loss: {log_loss_dict['Frame_loss']:.6f}")
        
        #if log_loss_dict['Frame_loss'] <= 0.1 or self.step == self.train_iters - 1:
        if self.step == 0:
            # 可视化youngs modulus
            
            # 生成保存路径
            iter_str = f"iter_{self.step}"
            iter_output_dir = os.path.join(self.output_dir, "youngs_modulus_results", iter_str)
            os.makedirs(iter_output_dir, exist_ok=True)

            final_ply_path = os.path.join(iter_output_dir, f"youngs_map_{iter_str}.ply")
            final_html_path = os.path.join(iter_output_dir, f"youngs_map_{iter_str}.html")
            final_csv_path = os.path.join(iter_output_dir, f"youngs_map_{iter_str}.csv")
    
            init_pos = self.particle_init_position.clone().detach()
            final_young = youngs_modulus_data
            
            visualize_youngs_modulus( 
                positions=init_pos,
                young_values=final_young,
                ply_file_path=final_ply_path,
                html_file_path=final_html_path,
                csv_file_path=final_csv_path,
                colormap_name='viridis', # 低值是深紫，高值是亮黄
            )
            
            plot_youngs_distribution(csv_file=final_csv_path, output_dir=iter_output_dir)
            
            torch.cuda.empty_cache()
        
        def is_invalid(x):
            return (math.isnan(x) or math.isinf(x))

        # 如果young_min_val < 0 或者任意值是 NaN/Inf，就终止训练
        if (youngs_modulus_data.min().item() < 0) or any([
            is_invalid(youngs_modulus_data.min().item()),
            is_invalid(youngs_modulus_data.max().item())
        ]):
            print("NaN, Inf, or negative Young's encountered, aborting training...")
            sys.exit(1)
        
        
        print(
            "nu: ",
            self.E_nu_list[1].mean().item(),
            "young_min:",
            youngs_modulus_data.min().item(),
            "young_mean:",
            youngs_modulus_data.mean().item(),
            "young_max:",
            youngs_modulus_data.max().item(),
        )
        
        
        iteration_end_time = time.time()
        iteration_duration = iteration_end_time - iteration_start_time

        # 将时间信息写入 txt 文件
        time_log_path = os.path.join(self.output_dir, "time_logs.txt")
        os.makedirs(os.path.dirname(time_log_path), exist_ok=True)
        with open(time_log_path, "a", encoding="utf-8") as f:
            f.write(f"Iteration {self.step} total time: {iteration_duration:.4f}s\n")
            for (start_t, end_t, frame_dur) in frame_times:
                f.write(f"\tFrame {start_t:02d}->{end_t:02d} total: {frame_dur:.4f}s\n")
                
        # pdb.set_trace()
        
        # 返回最终的模拟状态（用于逐帧顺序训练）
        return {
            'particle_pos': particle_pos,  # 最终的粒子位置
            'particle_velo': particle_velo,  # 最终的粒子速度
            'particle_F': particle_F,  # 最终的变形梯度
            'particle_C': particle_C,  # 最终的速度梯度
            'cuboid_point': self.cuboid_point,  # 最终的cuboid位置
            'loss': log_loss_dict['Frame_loss'],  # 平均loss
        }
        

    def train(self):
        """逐帧训练：每次训练一个2帧窗口，状态向前传递"""
        
        total_frames = self.num_frames - 1  # 总共要训练的帧数
        
        print(f"num_frames={self.num_frames}: 使用逐帧训练逻辑，总共训练 {total_frames} 帧")
        
        # 初始化物理状态（第一帧时这些为None，会使用默认值）
        self.particle_init_velocity = None
        self.particle_init_F = None
        self.particle_init_C = None
        
        for frame_idx in range(total_frames):
            print(f"\n{'='*80}")
            print(f"Training Frame {frame_idx} -> {frame_idx+1}")
            print(f"{'='*80}\n")
            
            # 每帧重置 step，避免学习率降为 0
            self.step = 0
            self.current_frame_idx = frame_idx
            # 注意：scheduler 会在下面重新创建，因为优化器会重新创建

            # 设置 window_size = 2（当前帧 + 下一帧）
            self.window_size = 2
            
            # 设置 GT meshes：取当前帧和下一帧
            self.gt_meshes = [self.gt_meshes_original[frame_idx], self.gt_meshes_original[frame_idx + 1]]
            
            # 在逐帧训练中，每帧只优化对应的速度参数
            # 注意：虽然代码中使用 cuboid_velocity[0]，但实际上训练的是 frame_idx -> frame_idx+1 的速度
            # 为了正确性，应该使用 cuboid_velocity[frame_idx]，但当前代码结构使用 [0]
            # 所以我们需要确保只优化当前帧对应的速度参数
            
            # 重置当前帧的 cuboid velocity 为 0（避免使用基于GT的初始速度导致不匹配）
            # 注意：在逐帧训练中，每帧都使用 cuboid_velocity[0]，但训练的是不同的帧对
            self.cuboid_velocity[0].data.zero_()
            
            # 为当前帧创建独立的优化器，只优化当前帧的速度参数
            # 这样可以避免优化器状态影响其他帧的速度参数
            current_frame_velocity = [self.cuboid_velocity[0]]  # 只优化当前帧使用的速度参数
            self.velocity_optimizer = torch.optim.SGD(
                current_frame_velocity,
                lr=self.args.lr,
                momentum=0.0,
                weight_decay=0.0,
            )
            
            # 重新创建 scheduler
            self.velocity_scheduler = get_linear_schedule_with_warmup(
                optimizer=self.velocity_optimizer,
                num_warmup_steps=self.args.warmup_step,
                num_training_steps=self.args.train_iters,
            )
            
            # 更新 trainable_params 为当前帧的速度参数
            self.trainable_params = current_frame_velocity
            
            # 记录最佳状态（loss最低的状态）
            best_loss = float('inf')
            best_particle_pos = None
            best_particle_velo = None
            best_particle_F = None
            best_particle_C = None
            best_cuboid_point = None
            best_iteration = -1
            
            # 训练当前帧 train_iters 次，记录最佳状态
            for iteration in tqdm(range(self.train_iters), desc=f"Frame {frame_idx}"):
                final_state = self.train_one_step()
                self.step += 1
                
                # 记录最佳状态（loss最低的状态）
                if final_state['loss'] < best_loss:
                    best_loss = final_state['loss']
                    best_particle_pos = final_state['particle_pos'].detach().clone()
                    best_particle_velo = final_state['particle_velo'].detach().clone()
                    best_particle_F = final_state['particle_F'].detach().clone()
                    best_particle_C = final_state['particle_C'].detach().clone()
                    best_cuboid_point = final_state['cuboid_point'].detach().clone()
                    best_iteration = iteration
            
            # 使用最佳状态作为下一帧的初始状态
            self.particle_init_position = best_particle_pos
            self.particle_init_velocity = best_particle_velo  # 保存最佳速度
            self.particle_init_F = best_particle_F  # 保存最佳变形梯度
            self.particle_init_C = best_particle_C  # 保存最佳速度梯度
            self.cuboid_point = best_cuboid_point
            
            print(f"✓ Frame {frame_idx} complete. Best loss: {best_loss:.6f} at iteration {best_iteration+1}. Using best state for next frame.")
            
            # 保存最佳状态的结果
            self.final_particle_positions.append(best_particle_pos.detach().cpu().numpy())
            self.final_cuboid_positions.append(best_cuboid_point.detach().cpu().numpy())
            self.final_losses.append(best_loss)
            
            # 创建三个子文件夹
            loss_dir = os.path.join(self.output_dir, 'loss_history')
            results_dir = os.path.join(self.output_dir, 'frame_results')
            ply_dir = os.path.join(self.output_dir, 'particle_positions')
            
            os.makedirs(loss_dir, exist_ok=True)
            os.makedirs(results_dir, exist_ok=True)
            os.makedirs(ply_dir, exist_ok=True)
            
            # 保存当前帧的loss历史为txt文件
            frame_loss_file = os.path.join(loss_dir, f'frame_{frame_idx}_loss.txt')
            with open(frame_loss_file, 'w') as f:
                f.write(f"Frame {frame_idx} Loss History\n")
                f.write("=" * 50 + "\n")
                f.write(f"Step\tLoss\n")
                for step in range(self.train_iters):
                    if not np.isnan(self.loss_array[step, frame_idx]):
                        f.write(f"{step}\t{self.loss_array[step, frame_idx]:.6f}\n")
            
            # 立即保存当前帧的结果为txt文件
            frame_results_file = os.path.join(results_dir, f'frame_{frame_idx}_results.txt')
            with open(frame_results_file, 'w') as f:
                f.write(f"Frame {frame_idx} Results\n")
                f.write("=" * 50 + "\n")
                f.write(f"Best Loss: {best_loss:.6f} (at iteration {best_iteration+1})\n")
                f.write(f"Train Iterations: {self.train_iters}\n")
                f.write(f"Particle Count: {best_particle_pos.shape[0]}\n")
                f.write(f"Cuboid Count: {best_cuboid_point.shape[0]}\n")
                f.write("\nParticle Positions (first 10):\n")
                particle_pos = best_particle_pos.detach().cpu().numpy()
                for i in range(min(10, particle_pos.shape[0])):
                    f.write(f"  {i}: [{particle_pos[i, 0]:.6f}, {particle_pos[i, 1]:.6f}, {particle_pos[i, 2]:.6f}]\n")
                f.write("\nCuboid Positions:\n")
                cuboid_pos = best_cuboid_point.detach().cpu().numpy()
                for i in range(cuboid_pos.shape[0]):
                    f.write(f"  {i}: [{cuboid_pos[i, 0]:.6f}, {cuboid_pos[i, 1]:.6f}, {cuboid_pos[i, 2]:.6f}]\n")
            
            # 保存当前帧的最佳粒子位置为PLY文件
            particle_pos_original = best_particle_pos.detach().cpu().numpy() * self.scale.cpu().numpy() - self.shift.cpu().numpy()
            save_ply([particle_pos_original], frame_idx, ply_dir, frame_idx=frame_idx)
            # 生成当前帧的loss图表
            frame_plot_dir = os.path.join(self.output_dir, 'loss_plots')
            os.makedirs(frame_plot_dir, exist_ok=True)
           
            # 为当前帧创建独立的loss图表
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 5))
            frame_losses = self.loss_array[:, frame_idx]
            valid_losses = frame_losses[~np.isnan(frame_losses)]
            
            if len(valid_losses) > 0:
                plt.plot(range(len(valid_losses)), valid_losses, marker='o', label=f"Frame {frame_idx} loss")
                plt.xlabel("Iterations")
                plt.ylabel("Frame loss")
                plt.title(f"Frame {frame_idx} Loss over Iterations")
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(frame_plot_dir, f'frame_{frame_idx}_loss.png'))
                plt.close()
                print(f"  - Loss plot: {frame_plot_dir}/frame_{frame_idx}_loss.png ({len(valid_losses)} points)")
            else:
                print(f"  - Warning: No valid loss data for frame {frame_idx}")
            
            print(f"✓ Frame {frame_idx} complete. Best loss: {best_loss:.6f} (at iteration {best_iteration+1})")
            print(f"  - Loss history: {frame_loss_file}")
            print(f"  - Results: {frame_results_file}")
            print(f"  - PLY file: {ply_dir}/frame_{frame_idx}.ply")


    
     

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/taiga/illinois/eng/ece/n-ahuja/haozhang/tjx/PHYSDREAMER/PhysDreamer/exp_motion/train/config.yml")

    # dataset params
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/taiga/illinois/eng/ece/n-ahuja/haozhang/tjx/PHYSDREAMER/PhysDreamer/data/sphere1/",
    )
    
    
    parser.add_argument("--youngs", type=float, default=7e3)
    parser.add_argument("--nu", type=float, default=0.3)
    parser.add_argument("--velo_factor", type=float, default=0.0)
    parser.add_argument("--sample_particles", type=int, default=100)
    
    parser.add_argument("--model", type=str, default="se3_field")
    parser.add_argument("--feat_dim", type=int, default=64)
    parser.add_argument("--num_decoder_layers", type=int, default=3)
    parser.add_argument("--decoder_hidden_size", type=int, default=64)
    parser.add_argument("--spatial_res", type=int, default=32)
    parser.add_argument("--zero_init", type=bool, default=True)

    parser.add_argument("--num_frames", type=str, default=13) # Haolan: mesh sequences的数量

    parser.add_argument("--grid_size", type=int, default=64)
    parser.add_argument("--sim_res", type=int, default=8)
    parser.add_argument("--sim_output_dim", type=int, default=1)
    parser.add_argument("--substep", type=int, default=100) # Haolan: 每两帧之间的模拟steps
    parser.add_argument("--loss_decay", type=float, default=1.0)
    parser.add_argument("--compute_window", type=int, default=1)
    parser.add_argument("--grad_window", type=int, default=14)
    # -1 means no gradient checkpointing
    parser.add_argument("--checkpoint_steps", type=int, default=-1)
    parser.add_argument("--stride", type=int, default=1)

    parser.add_argument("--downsample_scale", type=float, default=0.04)
    parser.add_argument("--top_k", type=int, default=8)

    # Logging and checkpointing
    parser.add_argument("--wandb_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="../../output/train/sphere1_3")
    parser.add_argument("--seed", type=int, default=0)

    # training parameters
    parser.add_argument("--num_splits", type=int, default=0)
    parser.add_argument("--train_iters", type=int, default=300)
    parser.add_argument("--iter_material", type=int, default=10)
    parser.add_argument("--lr", type=float, default=8e-2)
    parser.add_argument("--lr_weights", type=float, default=1.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_step", type=int, default=5)
    
    # visualization overlap scale
    parser.add_argument("--overlap_scale", type=float, default=5.5)


    # distributed training args
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )

    args, extra_args = parser.parse_known_args()
    cfg = create_config(args.config, args, extra_args)

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    print(args.local_rank, "local rank")

    return cfg


if __name__ == "__main__":
    
    print("Training velocity in Cuboid Case")
    
    # pdb.set_trace()
    
    args = parse_args()
    
    trainer = Trainer(args)

    trainer.train()