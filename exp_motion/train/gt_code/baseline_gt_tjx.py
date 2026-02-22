
'''
Haolan：

Generate the GT

/scratch/bbqk/haozhang/.conda/envs/physdreamer/bin/python -m pip install warp-lang==0.10.1

'''
#123
     
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
import json
import open3d as o3d

import sys
import os
# Add the PhysRig root directory to Python path
physrig_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if physrig_root not in sys.path:
    sys.path.insert(0, physrig_root)

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
from thirdparty_code.warp_mpm.mpm_solver_diff_cuboid_close import MPMWARPDiff
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
    # 删除match_point，因为不再使用mask
    calculate_bounding_box,
    check_gradients,
    save_ply,
    log_losses, 
    plot_losses,
    plot_youngs_modulus,
    visualize_cuboids,
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

import exp_motion.train.cuboid_utils as cu
print("实际用到的 cuboid_utils.py 路径:", cu.__file__)


from exp_motion.train.cuboid_utils import (
    cuboid_finding,
    assign_cuboid_velocity,
)


from interface import (
    MPMDifferentiableSimulationRig,
)

class Trainer:
    def __init__(self, args):
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.args = args
        
        self.num_frames = int(args.num_frames)
        self.num_intermediate_frames = int(args.num_intermediate_frames)

        self.window_size = self.num_frames+self.num_intermediate_frames
        
        set_random_seed(42)
        
        # output
        args.wandb_name += (
            "SP_{}_youngs_{}_lr_{}_substep_{}_iters_{}_sw_{}".format(
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

        self.cuboid_velocity_log_path = os.path.join(
            self.output_dir, "cuboid_velocity.txt"
        )
        with open(self.cuboid_velocity_log_path, "w", encoding="utf-8") as f:
            f.write("Cuboid velocity log\n")
        
        dataset_dir = args.dataset_dir
        
        self.dataset_dir = dataset_dir
        
        gtmesh_dir_og = args.skeleton_dir or os.path.join(dataset_dir, "skeleton")#读取的skeleton sequence
        
        self.sim_path = os.path.join(dataset_dir, "infilled/infilled_0.ply")#用于模拟的点云
        
        # gt and sim
        
        self.gt_meshes_og = load_mesh_sequences(gtmesh_dir_og)
        
        self.xyzs = pcu.load_mesh_v(self.sim_path)
        
        self.num_particles = torch.tensor(self.xyzs, dtype=torch.float32, device=self.device).shape[0]

        # setup simulation
        
        self.delta_time = 1/25
        
        self.sample_particles = args.sample_particles
        
        # 删除skin_weights相关代码，因为不再使用mask
        
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

        self.step = 0
        
        self.iter_material = args.iter_material

        self.max_grad_norm = args.max_grad_norm
  
        
    def init_trainable_params(self,):

        sim_points = torch.tensor(self.xyzs, dtype=torch.float32, device=self.device)
        
        self.sample_indices = farthest_point_sampling(sim_points, self.sample_particles)
        
        youngs_modulus = nn.Parameter(
            torch.ones(self.sample_particles, device=self.device, dtype=torch.float32) * self.args.youngs, 
            requires_grad=False
        )
        
        # 删除mask相关处理，因为现在使用cuboid包裹每个点
        
        youngs_modulus = youngs_modulus.clone()
        
        N = self.xyzs.shape[0]  # 全局粒子数
        k = self.sample_indices.shape[0]  # 采样数量
        
        global_to_local = torch.full((N,), -1, dtype=torch.long, device=self.device)
        
        for local_i in range(k):
            g_i = self.sample_indices[local_i]  # 全局索引
            global_to_local[g_i] = local_i 

        # 删除mask处理部分
        
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
        self.gt_meshes_og = [
            (mesh + shift) / scale for mesh in self.gt_meshes_og]
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
        self.grid_dx = grid_dx
        print("grid_dx:",grid_dx)
        print(f"X range: {x_range:.3f}m, Y range: {y_range:.3f}m, Z range: {z_range:.3f}m")
        
         # Haolan：创建gt_mask
        gt_mesh_initial = os.path.join(dataset_dir, "skeleton/gt_0.ply")
        pcd = o3d.io.read_point_cloud(gt_mesh_initial)
        gt_vertices = np.asarray(pcd.points)
        # 保证 shift/scale 是 numpy 数组
        shift_np = shift.cpu().numpy() if hasattr(shift, 'cpu') else np.array(shift)
        scale_np = scale.cpu().numpy() if hasattr(scale, 'cpu') else np.array(scale)
        preprocessed_vertices = (gt_vertices + shift_np) / scale_np
        self.num_vertices = preprocessed_vertices.shape[0]
        
        
        cuboid_centers, cuboid_sizes, cuboid_types, _ = cuboid_finding(
            preprocessed_vertices,
            grid_dx,
            vertices_assignment=None,
            mesh=None,
            sizing_mode=args.cuboid_size_mode,
            sizing_coeff=args.cuboid_size_coeff,
            knn_k=args.cuboid_knn_k,
            full_pointcloud=sim_xyzs.cpu().numpy(),
            clamp_min_radius=args.clamp_cuboid_min_radius,
        )
    
        self.num_cuboids = len(cuboid_centers)
    
        if len(cuboid_centers) == 0 or len(cuboid_sizes) == 0:
            raise ValueError("cuboid_finding_capsule 未能找到有效的 cuboid，请检查输入顶点。")

            
        self.cuboid_point = torch.tensor(cuboid_centers, dtype=torch.float32, device=device)
        self.cuboid_size = torch.tensor(cuboid_sizes, dtype=torch.float32, device=device)

        # cuboid可视化
        output_html_path = os.path.join(self.output_dir, "cuboid_visualization.html")

        grid_info = {
            'n_grid': grid_size,
            'grid_lim': grid_lim,
            'dx': grid_dx,
            'grid_dim_x': grid_size,
            'grid_dim_y': grid_size,
            'grid_dim_z': grid_size
        }

        visualize_cuboids(
            selected_vertices=preprocessed_vertices,
            assignment=None,
            cuboid_centers=cuboid_centers,
            cuboid_sizes=cuboid_sizes,
            output_html_path=output_html_path
        )

        self.cuboid_velocity, self.cuboid_positions, self.cuboid_update_mode = assign_cuboid_velocity(
            cuboid_centers=cuboid_centers,
            cuboid_sizes=cuboid_sizes,
            cuboid_types=cuboid_types,
            total_frames=self.num_frames,
            gt_meshes=self.gt_meshes_og,
            vertices=preprocessed_vertices,
            delta_time=self.delta_time,
            device=device,
            num_intermediate_frames=args.num_intermediate_frames,
            cuboid_update_mode=args.cuboid_update_mode,
            position_method=args.position_method,
        )
        
        for i in range(len(self.cuboid_velocity)):
            self.cuboid_velocity[i].data *= self.velo_factor
        
        print(f"✅ Cuboid update mode: {self.cuboid_update_mode}")
        print(f"   - Velocities: {len(self.cuboid_velocity)} frames")
        print(f"   - Positions: {len(self.cuboid_positions)} frames")


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

    def log_cuboid_velocity(self, start_idx, end_idx, cuboid_velocity):
        if self.step != self.train_iters - 1:
            return
        cuboid_velocity = cuboid_velocity.detach().cpu()
        with open(self.cuboid_velocity_log_path, "a", encoding="utf-8") as f:
            f.write(
                f"Iter {self.step:04d} | Frame {start_idx:02d}->{end_idx:02d} | "
                f"cuboids: {cuboid_velocity.shape[0]}\n"
            )
            for cuboid_idx, vec in enumerate(cuboid_velocity):
                f.write(
                    f"  cuboid_{cuboid_idx:04d}: "
                    f"{vec[0].item():.6f} {vec[1].item():.6f} {vec[2].item():.6f}\n"
                )
            f.write("\n")
    
        
    def train_one_step(self):
        
        device = self.device
        
        iteration_start_time = time.time() # 用于记录时间
        
        # scheduler, start from 0
        window_size = self.window_size - 1
        
        print(f"Window size: {self.window_size}")    
        
            
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

        temporal_stride = self.args.stride

        # Haolan：避免cuboid_point累积，并且更新全局模拟时间
        if self.step == 0:
            self.initial_cuboid_point = self.cuboid_point.clone()
            
        if temporal_stride < 0 or temporal_stride > window_size:
            temporal_stride = window_size
        
        self.cuboid_point = self.initial_cuboid_point.detach()
        
        frame_time_offset = 0.0
        
        frame_times = []
        
        for start_time_idx in range(0, window_size, temporal_stride):
            
            frame_start_time = time.time()
            
            end_time_idx = min(start_time_idx + temporal_stride, window_size)
            
            num_step_with_grad = num_substeps * (end_time_idx - start_time_idx)
            
            #tjx只跑inference
            #gt_frame = self.gt_meshes_og[start_time_idx + 1] # mesh sequences作为监督，下一帧作为上一帧模拟结果的监督
  
            if start_time_idx != 0:
                density, youngs_modulus, poisson = self.get_material_params(device)
            
                
            print(f"Processing frames from {start_time_idx} to {end_time_idx}")
            
            current_cuboid_velocity = self.cuboid_velocity[start_time_idx]
            self.log_cuboid_velocity(start_time_idx, end_time_idx, current_cuboid_velocity)
            
            if self.step == 0:
                if start_time_idx == 0:
                    sim_points = particle_pos.detach()
                    sim_points = sim_points * self.scale - self.shift
                    points_list = [sim_points]
                    save_ply(points_list, self.step, self.output_dir, frame_idx=start_time_idx, for_gt=True)
                
                
            if self.args.dynamic_cuboid_cap:
                # Dynamic per-frame cuboid radius capping: only shrink between
                # cuboids with divergent velocities (different body parts).
                dists = torch.cdist(self.cuboid_point.unsqueeze(0),
                                    self.cuboid_point.unsqueeze(0)).squeeze(0)
                dists.fill_diagonal_(float('inf'))

                vel = current_cuboid_velocity.detach()
                vel_norm = torch.norm(vel, dim=1, keepdim=True).clamp(min=1e-8)
                vel_dir = vel / vel_norm
                cosine_sim = vel_dir @ vel_dir.T
                friendly = cosine_sim > 0.5
                dists[friendly] = float('inf')

                nn_dist, _ = dists.min(dim=1)
                max_radius = nn_dist / 2.0
                original_radius = self.cuboid_size[:, 0]
                capped_radius = torch.min(original_radius, max_radius)
                capped_radius = torch.clamp(capped_radius, min=self.grid_dx)
                effective_cuboid_size = capped_radius.unsqueeze(1).expand_as(self.cuboid_size)
            else:
                effective_cuboid_size = self.cuboid_size

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
                    effective_cuboid_size, # dynamically capped to prevent overlap
                    frame_time_offset, # 全局时间累积
                    density,
                    device,
                    True,
                )
            )
            
            # Haolan：更新cuboid_point位置，根据 cuboid_update_mode 选择更新方式
            if self.cuboid_update_mode == "none":
                pass  # keep cuboid_point at initial position
            elif self.cuboid_update_mode in ["location_only", "both"]:
                # Use tracked point positions directly
                self.cuboid_point = self.cuboid_positions[start_time_idx].detach()
            else:
                # Use velocity-based update (velocity_only mode)
                self.cuboid_point = (self.cuboid_point + current_cuboid_velocity * delta_time).detach()
            
            # Haolan：累积全局时间
            frame_time_offset += delta_time
            
            if self.step == 0:
                sim_points = particle_pos.detach()
                sim_points = sim_points * self.scale - self.shift
                points_list = [sim_points]
                save_ply(points_list, self.step, self.output_dir, frame_idx=start_time_idx+1, for_gt=True)
            
 

            particle_pos, particle_velo, particle_F, particle_C = (
                particle_pos.detach(),
                particle_velo.detach(),
                particle_F.detach(),
                particle_C.detach(),
            )

            frame_index = f"{start_time_idx}_{end_time_idx}"
            
            frame_end_time = time.time()
            frame_duration = frame_end_time - frame_start_time
            frame_times.append((
                start_time_idx, end_time_idx, frame_duration
            ))
        
            # pdb.set_trace()
    
        
        # check_gradients(
        #     ("Before self.E_nu_list[0]", self.E_nu_list[0])  # 位置参数
        # )
        
        # 不能一起裁剪，因为材质和速度的梯度范数差距很大
        

        _, youngs_modulus, _ = self.get_material_params(device)
        
        youngs_modulus_data = youngs_modulus.clone().detach()
        
        self.youngs_modulus_min.append(youngs_modulus_data.min().item())
        self.youngs_modulus_mean.append(youngs_modulus_data.mean().item())
        self.youngs_modulus_max.append(youngs_modulus_data.max().item())
        
        if self.step == self.train_iters - 1:
            self.final_youngs_modulus = youngs_modulus_data

        
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
        

    def train(self):
        
        sim_start = time.time()
        for iteration in tqdm(range(1), desc="Training progress"):

            self.train_one_step()
            self.step += 1
        sim_elapsed = time.time() - sim_start

        num_sim_frames = self.window_size - 1
        stats = {
            "skeleton_points": self.num_vertices,
            "cuboids": self.num_cuboids,
            "points": self.num_particles,
            "frames": num_sim_frames,
            "inference_time_s": round(sim_elapsed, 2),
            "fps": round(num_sim_frames / sim_elapsed, 2) if sim_elapsed > 0 else 0,
        }
        stats_path = os.path.join(self.output_dir, "inference_stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        plot_losses(self.output_dir, num_iterations=self.train_iters) 
        plot_youngs_modulus(self.output_dir, self.youngs_modulus_mean, self.youngs_modulus_max, self.youngs_modulus_min)


    
     

def parse_args():
    parser = argparse.ArgumentParser()
    # Get config path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config = os.path.join(script_dir, '../config.yml')
    parser.add_argument("--config", type=str, default=default_config)

    # dataset params
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/taiga/illinois/eng/ece/n-ahuja/haozhang/tjx/PHYSDREAMER/PhysDreamer/data/dragon_tjx_1/",
    )

    parser.add_argument(
        "--skeleton_dir",
        type=str,
        default=None,
        help="Optional skeleton/gt_*.ply directory override",
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

    parser.add_argument("--num_frames", type=str, default=10) # Haolan: mesh sequences的数量
    parser.add_argument("--num_intermediate_frames", type=int, default=10)
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
    parser.add_argument("--wandb_name", type=str, required=True,default="tjx")
    parser.add_argument("--output_dir", type=str, default="../../output/dragon_tjx_1")
    parser.add_argument("--seed", type=int, default=0)

    # training parameters
    parser.add_argument("--num_splits", type=int, default=0)
    parser.add_argument("--train_iters", type=int, default=1)
    parser.add_argument("--iter_material", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_weights", type=float, default=1.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_step", type=int, default=5)
    parser.add_argument("--cuboid_update_mode", type=str, default="both", 
                        choices=["none", "velocity_only", "location_only", "both"],
                        help="Cuboid update mode: none, velocity_only, location_only, or both")
    parser.add_argument("--position_method", type=str, default="mean",
                        choices=["mean", "median", "weighted", "bbox", "adaptive", "pca", "optimized"],
                        help="Position calculation method for cuboid centers from tracked points")
    parser.add_argument("--cuboid_size_mode", type=str, default="fixed",
                        choices=["fixed", "adaptive", "knn", "hybrid", "ceil_and_floor", "raycast"],
                        help="Cuboid sizing strategy: fixed, adaptive, knn, hybrid, ceil_and_floor, or raycast")
    parser.add_argument("--cuboid_size_coeff", type=float, default=0.7,
                        help="Scaling coefficient for cuboid radius (used by all sizing modes)")
    parser.add_argument("--cuboid_knn_k", type=int, default=20,
                        help="K for KNN-based sizing modes (knn, hybrid, raycast)")
    parser.add_argument("--dynamic_cuboid_cap", action="store_true", default=False,
                        help="Enable dynamic per-frame cuboid radius capping based on velocity divergence")
    parser.add_argument("--clamp_cuboid_min_radius", action="store_true", default=False,
                        help="Clamp cuboid radii to at least grid_dx so every cuboid covers an MPM node")


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
    
    print("Generate GT")
    
    # pdb.set_trace()
    
    args = parse_args()
    
    trainer = Trainer(args)

    trainer.train()