
'''
Haolan：

单个youngs值能够有效学习，每个点的youngs值无法有效学习

基于随机选取的点插值全局youngs

只测试速度训练

一帧一帧学

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


import numpy as np
import logging
import argparse
import shutil
import wandb
import pdb
import trimesh
import math

import sys
sys.path.append("/scratch/bbqk/haozhang/files/physdreamer/PhysDreamer")

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
        
        self.iter4frame = args.iter4frame
        
        self.train_iters = self.iter4frame * (self.num_frames - 1)
        
        set_random_seed(42)
        
        # output
        args.wandb_name += (
            "vf_{}_SP_{}_youngs_{}_lr_{}_substep_{}_iters_{}_frames_{}".format(
                args.velo_factor,
                args.sample_particles,
                args.youngs,
                args.lr,
                args.substep,
                self.train_iters,
                self.num_frames,
            )
        )
        
        # path
        self.output_dir = os.path.join(args.output_dir, args.wandb_name)
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        dataset_dir = args.dataset_dir
        
        self.dataset_dir = dataset_dir
        
        gtmesh_dir = os.path.join(dataset_dir, "sinmat_gt_50SP_2.0grid")
        
        gtmesh_dir_og = os.path.join(dataset_dir, "gt_mesh")
        
        self.sim_path = os.path.join(dataset_dir, "infilled/infilled_0.ply")
        
        # gt and sim
        self.gt_meshes = load_mesh_sequences(gtmesh_dir)
        
        self.gt_meshes_og = load_mesh_sequences(gtmesh_dir_og)
        
        self.xyzs = pcu.load_mesh_v(self.sim_path)
        
        self.num_particles = torch.tensor(self.xyzs, dtype=torch.float32, device=self.device).shape[0]

        # setup simulation
        
        self.delta_time = 1/25
        
        self.sample_particles = args.sample_particles
        
        sk_path = os.path.join(dataset_dir, "skinning_masks.npy")
        
        self.skin_weights = np.load(sk_path)
        
        self.velo_factor = args.velo_factor
        
        E_nu_list = self.init_trainable_params()
        
        self.E_nu_list = E_nu_list
        
        self.setup_simulation(dataset_dir, grid_size=args.grid_size)
        
        self.youngs_modulus_min = []
        self.youngs_modulus_mean = []
        self.youngs_modulus_max = []
        
        
        visualize_sampled_points(self.xyzs, self.sample_indices, self.output_dir)
        
        # setup training
        
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
            num_training_steps=self.iter4frame,
        )

        self.step = 0
        
        self.gap = 1
        
        self.window_size = 1
        
        self.prev_window_size = 1
        
        self.loss_threshold = 0.001
        
        self.trained_iter = 1 # 03/01
        
        self.cuboid_velocity_log = []
        
        self.saved_states = {}
        
        self.saved_cuboids = {}
        
        self.loss_array = np.full((self.train_iters, self.num_frames - 1), np.nan)
        
        self.iter_material = args.iter_material

        self.max_grad_norm = args.max_grad_norm
  
        
    def init_trainable_params(self,):

        sim_points = torch.tensor(self.xyzs, dtype=torch.float32, device=self.device)
        
        self.sample_indices = farthest_point_sampling(sim_points, self.sample_particles)
        
        youngs_modulus = nn.Parameter(
            torch.ones(self.sample_particles, device=self.device, dtype=torch.float32) * self.args.youngs, 
            requires_grad=False
        )
        
        # sim mask
        
        mask_files = ["chest.ply", "belly.ply", "hip.ply"]

        mask_youngs_list = [          
            5e4,  
            5e4,
            5e4,
        ]

        mask_indices_list = [] 
        
        for mask_file in mask_files:
            sim_mask_path = os.path.join(self.dataset_dir, "sim_mask", mask_file)

            # 用 match_point 找到在整体mesh里的索引
            sim_mask_indices = match_point(self.sim_path, sim_mask_path)

            # 把这个 mask 的索引存起来
            mask_indices_list.append(sim_mask_indices)

        # 检查一下
        for i, indices in enumerate(mask_indices_list):
            print(f"第 {i} 个 mask 文件: {mask_files[i]} => 匹配到点数: {indices.shape[0]}")
    
        
        youngs_modulus = youngs_modulus.clone()
        
        N = self.xyzs.shape[0]  # 全局粒子数
        k = self.sample_indices.shape[0]  # 采样数量
        
        global_to_local = torch.full((N,), -1, dtype=torch.long, device=self.device)
        
        for local_i in range(k):
            g_i = self.sample_indices[local_i]  # 全局索引
            global_to_local[g_i] = local_i 

        for i, indices_global in enumerate(mask_indices_list):
            intersection_global_np = np.intersect1d(
                self.sample_indices.cpu().numpy(),
                indices_global.cpu().numpy()
            )
            intersection_global = torch.from_numpy(intersection_global_np).long().to(self.device)

            # 转换成本地索引
            local_indices = global_to_local[intersection_global]
            valid_mask = (local_indices >= 0)
            local_indices = local_indices[valid_mask]

            # 覆盖这些 local_indices 的 Young’s
            youngs_modulus[local_indices] = mask_youngs_list[i]

            # 输出一下信息
            print(f"对第 {i} 个 mask => {mask_files[i]} 共覆盖 {len(local_indices)} 个 sample 点, Young’s = {mask_youngs_list[i]}") 
            
        
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
        
        # Haolan：创建gt_mask
        gt_mesh_initial = os.path.join(dataset_dir, "./sinmat_gt_50SP_2.0grid/gt_0.ply")

        self.gt_mask_indices = match_point(self.sim_path, gt_mesh_initial)
        
        print(f"Number of matched indices: {self.gt_mask_indices.shape[0]}")
        
        # 对gt和mesh进行scale和shift
        sim_xyzs = (sim_xyzs + shift) / scale
        
        self.gt_meshes = [
            (mesh + shift) / scale for mesh in self.gt_meshes
        ]
        
        self.gt_meshes_og = [
            (mesh + shift) / scale for mesh in self.gt_meshes_og
        ]
            
        bbox = calculate_bounding_box(sim_xyzs)

        # 获取最大绝对值
        max_abs_value = max(abs(bbox['x_min']), abs(bbox['x_max']), 
                            abs(bbox['y_min']), abs(bbox['y_max']), 
                            abs(bbox['z_min']), abs(bbox['z_max']))
        # print(f"Bounding Box 中绝对值最大的值: {max_abs_value}")
        
        
        points_volume = get_volume(sim_xyzs.detach().cpu().numpy())
        
        
        wp.init()
        wp.config.mode = "debug"
        wp.config.verify_cuda = True
    
        
        # Haolan：grid_size和grid_lim对于模拟很重要，因为它能决定grid的大小
        grid_size = 80
        grid_lim = max_abs_value * 1.5
        grid_dx = grid_lim / grid_size
        print(f"Grid Size: {grid_size}; Grid Lim: {grid_lim}")
        
        # Haolan:初始化 cuboid 参数
        skin_weights = self.skin_weights

        self.masks = torch.from_numpy(skin_weights).to("cuda")
        
        mesh_path = os.path.join(dataset_dir, f"mesh/mesh_frame_0001.obj")
        mesh = trimesh.load(mesh_path, process=False)
        vertices = np.array(mesh.vertices)
        
        shift_np = shift.cpu().numpy() if isinstance(shift, torch.Tensor) else shift
        scale_np = scale.cpu().numpy() if isinstance(scale, torch.Tensor) else scale

        preprocessed_vertices = (vertices + shift_np) / scale_np
        skin_weights = skin_weights.T

        assignment = vertice_assignment(preprocessed_vertices, skin_weights)
        self.assignment = assignment
        
        cuboid_centers, cuboid_sizes, cuboid_types, boundary_points_list = cuboid_finding(preprocessed_vertices, assignment, grid_dx, mesh)
    
    
        # 检查 cuboid_finding 是否成功找到立方体
        if len(cuboid_centers) == 0 or len(cuboid_sizes) == 0:
            raise ValueError("cuboid_finding 未能找到有效的 cuboid，请检查输入顶点。")

            
        self.cuboid_point = torch.tensor(cuboid_centers, dtype=torch.float32, device=device)
        self.cuboid_size = torch.tensor(cuboid_sizes, dtype=torch.float32, device=device)
        
        ply_path = os.path.join(dataset_dir, f"gt_mesh/gt_1.ply")
        
        # part可视化
        visualize_parts(ply_path, self.masks, self.output_dir)
        
        # pdb.set_trace()

        # cuboid可视化
        output_html_path = os.path.join(self.output_dir, "cuboid_visualization.html")

        # 调用函数
        visualize_cuboids(
            selected_vertices=preprocessed_vertices,
            assignment=assignment,
            cuboid_centers=cuboid_centers,
            cuboid_sizes=cuboid_sizes,
            output_html_path=output_html_path
        )

        
        self.cuboid_velocity = assign_cuboid_velocity(
            cuboid_centers=cuboid_centers,
            cuboid_sizes=cuboid_sizes,
            cuboid_types=cuboid_types,
            boundary_points_list=boundary_points_list,
            total_frames = self.num_frames,
            gt_meshes=self.gt_meshes, # gt_meshes=self.gt_meshes_og, 
            vertices=preprocessed_vertices,
            vertices_assignment=assignment,
            grid_dx=grid_dx,
            delta_time=self.delta_time,
            device=device,
        )
        
        
        
        self.cuboid_velocity_gt = assign_cuboid_velocity(
            cuboid_centers=cuboid_centers,
            cuboid_sizes=cuboid_sizes,
            cuboid_types=cuboid_types,
            boundary_points_list=boundary_points_list,
            total_frames = self.num_frames,
            gt_meshes=self.gt_meshes_og,  
            vertices=preprocessed_vertices,
            vertices_assignment=assignment,
            grid_dx=grid_dx,
            delta_time=self.delta_time,
            device=device,
        )
        
        output_file1 = os.path.join(self.output_dir, 'cuboid_velo/cuboid_velocity_init.pt')
        os.makedirs(os.path.dirname(output_file1), exist_ok=True)
        torch.save(self.cuboid_velocity, output_file1)
        print(f"Saved cuboid_velocity to {output_file1}")
        
        output_file2 = os.path.join(self.output_dir, 'cuboid_velo/cuboid_velocity_gt.pt')
        os.makedirs(os.path.dirname(output_file2), exist_ok=True)
        torch.save(self.cuboid_velocity_gt, output_file2)
        print(f"Saved cuboid_velocity_gt to {output_file2}")
        
        for i in range(len(self.cuboid_velocity)):
            self.cuboid_velocity[i].data *= self.velo_factor

        
        # for idx, velocity_tensor in enumerate(self.cuboid_velocity):
        #     print(f"Index {idx}: Shape: {velocity_tensor.shape}")
        
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
    
        # 直接将初速度设定为常量0
        velocity = torch.zeros_like(initial_position, device=device)

        # init F, and C
        I_mat = torch.eye(3, dtype=torch.float32).to(device)
        
        particle_F = torch.repeat_interleave(
            I_mat[None, ...], initial_position.shape[0], dim=0
        )
        
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
    
    # 03/01
    def train_one_step(self):
        
        device = self.device
        
        iteration_start_time = time.time() # 用于记录时间

        window_size = self.window_size
        
        print(f"Window size: {self.window_size}")    

         
        if window_size == 1:
            particle_pos = self.particle_init_position.clone()
        else:
            saved_state = self.saved_states[self.prev_window_size - 1]
            particle_pos  = saved_state["particle_pos"]

            
        
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
        
        
        if window_size != 1:
            saved_state = self.saved_states[self.prev_window_size - 1]
            particle_velo = saved_state["particle_velo"]
            particle_F = saved_state["particle_F"]
            particle_C = saved_state["particle_C"]
    
        delta_time = self.delta_time 
        substep_size = delta_time / self.args.substep
        num_substeps = int(delta_time / substep_size)

        
        if self.step == 0:
            self.init_cuboid_point = self.cuboid_point.clone()
            
        if window_size == 1:
            self.cuboid_point = self.init_cuboid_point
        else:
            saved_cuboid = self.saved_cuboids[self.prev_window_size - 1]
            self.cuboid_point = saved_cuboid["cuboid_point"]


        
        frame_time_offset = 0.0
        
        frame_times = []
        
        for start_time_idx in range(window_size - 1, window_size):
            
            frame_start_time = time.time()
            
            end_time_idx = min(start_time_idx + 1, window_size)
            
            num_step_with_grad = num_substeps * (end_time_idx - start_time_idx)
            
            gt_frame = self.gt_meshes[start_time_idx + 1] # mesh sequences作为监督，下一帧作为上一帧模拟结果的监督
            
                
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
            
            predicted_points = particle_pos[self.gt_mask_indices].clone()  
            target_points = gt_frame
            
            predicted_points_cd = particle_pos[self.gt_mask_indices].unsqueeze(0).clone()  
            target_points_cd = gt_frame.unsqueeze(0) 

            # global CD loss             
            global_loss = chamfer_dist(predicted_points_cd, target_points_cd)
            loss = global_loss * 100
            
            # print(f"Frame Loss: {loss.item()}")

            loss.backward()

            current_frame = start_time_idx
            
            current_loss = loss.item()
            
            self.loss_array[self.step, current_frame] = current_loss
            
            
            if self.step == 0:
                self.saved_states[current_frame] = {
                    "particle_pos": particle_pos.detach(),
                    "particle_velo": particle_velo.detach(),
                    "particle_F": particle_F.detach(),
                    "particle_C": particle_C.detach(),
                }
     
                
                self.saved_cuboids[current_frame] = {
                    "cuboid_point": self.cuboid_point.clone(),
                }
            
            if self.step > 0:
                prev_loss = self.loss_array[self.step - 1, current_frame]
                
                if (not np.isnan(prev_loss) and current_loss < prev_loss) or np.isnan(prev_loss):
                
                    self.saved_states[current_frame] = {
                        "particle_pos": particle_pos.detach(),
                        "particle_velo": particle_velo.detach(),
                        "particle_F": particle_F.detach(),
                        "particle_C": particle_C.detach(),
                    }
            
                    self.saved_cuboids[current_frame] = {
                        "cuboid_point": self.cuboid_point.clone(),
                    }
                #     print(f"Update saved states and cuboid points")
                # else:
                #     print(f"Not update saved states and cuboid points")
                    
            frame_index = f"{start_time_idx}_{end_time_idx}"
            
            frame_end_time = time.time()
            frame_duration = frame_end_time - frame_start_time
            frame_times.append((
                start_time_idx, end_time_idx, frame_duration
            ))
        
            # pdb.set_trace()
        
        self.cuboid_velocity_log.append({
            "frame": current_frame,
            "velocity": current_cuboid_velocity.clone().detach()
        })
        
        torch.nn.utils.clip_grad_norm_(
            self.trainable_params,
            self.max_grad_norm,
            error_if_nonfinite=False,
        )  # error if nonfinite is false
        
        self.velocity_optimizer.step()
        torch.cuda.empty_cache()
        self.velocity_optimizer.zero_grad()
        self.velocity_scheduler.step()

        if (current_loss < self.loss_threshold or self.trained_iter >= self.iter4frame) and self.window_size < self.num_frames - 1:
            
            self.prev_window_size = self.window_size
            self.window_size = min(self.window_size + self.gap, self.num_frames)
            print(f"Expanding sequence to {self.window_size}")
            
            old_params = self.cuboid_velocity[:self.window_size - self.gap]
            new_params = self.cuboid_velocity[self.window_size - self.gap:self.window_size]
                
            param_groups = [
                {'params': old_params, 'lr': 0.0},  # 冻结旧帧
                {'params': new_params, 'lr': self.args.lr}  # 新帧
            ]
            
            self.velocity_optimizer = torch.optim.SGD(param_groups, momentum=0.0, weight_decay=0.0)

            # 重新初始化 scheduler
            self.velocity_scheduler = get_linear_schedule_with_warmup(
                optimizer=self.velocity_optimizer,
                num_warmup_steps=self.args.warmup_step,
                num_training_steps=self.iter4frame,
            )
            
            self.trained_iter = 1
        else:
            self.trained_iter += 1 

            
        log_losses(
            frame_loss=current_loss,
            iteration=self.step
        )

        print(f"Iteration {self.step}: Frame {current_frame} to Frame {current_frame + 1} Loss: {current_loss}")
        
        
        if self.step == self.train_iters - 1:
            min_per_column = np.nanmin(self.loss_array, axis=0) # (self.num_frames - 1,)
            average_min = np.mean(min_per_column)
            print(f"Average Frame Loss: {average_min}")

        
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
        
    # 03/01
    def train(self):
        
        for iteration in tqdm(range(self.train_iters), desc="Training progress"):

            self.train_one_step()
            self.step += 1
            
            if self.window_size == self.num_frames and self.trained_iter >= self.iter4frame:
                print(f"The last frame is trained for {self.iter4frame} iterations. Exiting training loop.")
                break
        
        # 保存loss array
        output_file = os.path.join(self.output_dir, 'loss_array.npy')
        np.save(output_file, self.loss_array)
        print(f"Loss array saved to {output_file}")
        
        # 保存cuboid velocity log
        output_file = os.path.join(self.output_dir, 'cuboid_velo/cuboid_velocity_log.pt')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        torch.save(self.cuboid_velocity_log, output_file)
        print(f"Cuboid velocity log saved to {output_file}")
        
        # 画图
        plot_losses(self.output_dir, num_iterations=self.train_iters) 
        plot_youngs_modulus(self.output_dir, self.youngs_modulus_mean, self.youngs_modulus_max, self.youngs_modulus_min)

        # 保存模拟点云
        for frame_idx in sorted(self.saved_states.keys()):
            state = self.saved_states[frame_idx]
            sim_points = state["particle_pos"][self.gt_mask_indices]
            sim_points = sim_points * self.scale - self.shift
            save_ply([sim_points], self.step, self.output_dir, frame_idx=frame_idx)
    
     

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/scratch/bbqk/haozhang/files/physdreamer/PhysDreamer/exp_motion/train/config.yml")

    # dataset params
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/scratch/bbqk/haozhang/files/physdreamer/PhysDreamer/data/whale/",
    )
    
    parser.add_argument("--youngs", type=float, default=6e4)
    parser.add_argument("--nu", type=float, default=0.3)
    parser.add_argument("--velo_factor", type=float, default=1.0)
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
    parser.add_argument("--substep", type=int, default=768) # Haolan: 每两帧之间的模拟steps
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
    parser.add_argument("--output_dir", type=str, default="../../output/whale")
    parser.add_argument("--seed", type=int, default=0)

    # training parameters
    parser.add_argument("--num_splits", type=int, default=0)
    parser.add_argument("--iter4frame", type=int, default=30)
    parser.add_argument("--train_iters", type=int, default=200)
    parser.add_argument("--iter_material", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_weights", type=float, default=1.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_step", type=int, default=5)


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