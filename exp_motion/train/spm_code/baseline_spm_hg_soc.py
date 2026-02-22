
'''
Haolan：

单个youngs值能够有效学习，每个点的youngs值无法有效学习

基于随机选取的点插值全局youngs

只测试材质训练

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
import math

from exp_motion.train.local_utils import (
    get_volume,
    create_spatial_fields,
)

# Haolan：整合新加入的函数
from exp_motion.train.new_utils import (
    init_logit_for_youngs,
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
        
        dataset_dir = args.dataset_dir
        
        self.dataset_dir = dataset_dir
        
        gtmesh_dir = os.path.join(dataset_dir, "mulmat_gt_50SP_2.0grid")
        
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
        
        self.skin_weights = np.load(sk_path)  # shape: (num_bones, num_vertices = 3525)
        
        self.part_id = torch.from_numpy(self.skin_weights.astype(np.float32)).argmax(dim=0).to(self.device)
        
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
        
        self.trainable_params = self.E_nu_list
            
        self.material_optimizer = torch.optim.AdamW(
            self.E_nu_list,
            lr=args.lr,
            weight_decay=0.0,
        )
        
        # self.material_optimizer = torch.optim.SGD(
        #     self.E_nu_list,
        #     lr=args.lr,  
        #     momentum=0.0,  # 添加动量项，提高收敛速度
        #     weight_decay=0.0,  # 权重衰减
        # )
        
        self.material_scheduler = get_linear_schedule_with_warmup(
            optimizer=self.material_optimizer,
            num_warmup_steps=args.warmup_step,
            num_training_steps=args.train_iters,
        )

        self.step = 0

        self.max_grad_norm = args.max_grad_norm
  
        
    def init_trainable_params(self,):
        
        self.E_min = 1e2
        
        self.E_max = 6.5e4
        
        sim_points = torch.tensor(self.xyzs, dtype=torch.float32, device=self.device)
        
        self.sample_indices = farthest_point_sampling(sim_points, self.sample_particles)
        
        init_val = init_logit_for_youngs(self.args.youngs, self.E_min, self.E_max)
        
        unbounded_youngs_modulus = nn.Parameter(
            torch.full(
                (self.sample_particles,),
                init_val,   
                device=self.device,
                dtype=torch.float32
            ),
            requires_grad=True
        )
        
        # log_youngs_modulus = nn.Parameter(
        #     torch.log(
        #         torch.ones(self.sample_particles, device=self.device, dtype=torch.float32) * (self.args.youngs - self.E_min)
        #     ),
        #     requires_grad=True
        # )
            
        poisson_ratio = nn.Parameter(
            torch.ones(self.num_particles, device=self.device, dtype=torch.float32) * args.nu,
            requires_grad=False
        )
        
        trainable_params = [unbounded_youngs_modulus, poisson_ratio]
        # trainable_params = [log_youngs_modulus, poisson_ratio]

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
        gt_mesh_initial = os.path.join(dataset_dir, "./mulmat_gt_50SP_2.0grid/gt_0.ply")

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

        
        # 用于计算cuboid velocity，需要最初的gt，因为这样indices才能对上
        self.cuboid_velocity = assign_cuboid_velocity(
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
        
        # sample_youngs = self.E_min + torch.exp(self.E_nu_list[0].clone()) 
        
        sample_youngs = self.E_min + (self.E_max - self.E_min) * torch.sigmoid(self.E_nu_list[0])
        
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
        
        # sample_youngs = self.E_min + torch.exp(self.E_nu_list[0])
        
        sample_youngs = self.E_min + (self.E_max - self.E_min) * torch.sigmoid(self.E_nu_list[0])
        
        youngs_modulus = (self.normalized_weights * sample_youngs.unsqueeze(0)).sum(dim=1)
        
        # youngs_modulus = torch.clamp(youngs_modulus, 100.0, 2e4) # 不能对叶子节点使用

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
            
            old_gt_frame = self.gt_meshes[start_time_idx]
            
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
            
            predicted_points = particle_pos[self.gt_mask_indices].clone()  
            target_points = gt_frame

            # global CD loss             
            global_loss = chamfer_dist(predicted_points.unsqueeze(0), target_points.unsqueeze(0))
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

            
        torch.nn.utils.clip_grad_norm_(
            self.trainable_params,
            self.max_grad_norm,
            error_if_nonfinite=False,
        )  # error if nonfinite is false


        # 只更新材质
        self.material_optimizer.step()
        self.material_optimizer.zero_grad()
        self.material_scheduler.step()

        _, youngs_modulus, _ = self.get_material_params(device)
        
        youngs_modulus_data = youngs_modulus.clone().detach()
        
        self.youngs_modulus_min.append(youngs_modulus_data.min().item())
        self.youngs_modulus_mean.append(youngs_modulus_data.mean().item())
        self.youngs_modulus_max.append(youngs_modulus_data.max().item())
        
        for k, v in log_loss_dict.items():
            log_loss_dict[k] = np.mean(v)
            
        log_losses(
            frame_loss=log_loss_dict["Frame_loss"],
            iteration=self.step
        )

        print(f"Iteration {self.step}: Average Frame Loss: {log_loss_dict['Frame_loss']:.6f}")
        
        # 保存loss符合要求的结果
        if log_loss_dict['Frame_loss'] <= 0.1 or self.step == self.train_iters - 1:
        
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
        

    def train(self):

        for iteration in tqdm(range(self.train_iters), desc="Training progress"):

            self.train_one_step()
            self.step += 1
        
        plot_losses(self.output_dir, num_iterations=self.train_iters) 
        plot_youngs_modulus(self.output_dir, self.youngs_modulus_mean, self.youngs_modulus_max, self.youngs_modulus_min)


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
    parser.add_argument("--train_iters", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
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
    
    print("Training material in Cuboid Case")
    
    # pdb.set_trace()
    
    args = parse_args()
    
    trainer = Trainer(args)

    trainer.train()
