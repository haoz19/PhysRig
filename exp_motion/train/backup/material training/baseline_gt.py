
'''
Haolan：

Generate the GT

01/28 现在生成的GT的尾巴有问题，有撕裂感

'''
    
import argparse
import time
import os
import numpy as np
import torch
from tqdm import tqdm

from torch import Tensor
from jaxtyping import Float, Int, Shaped
from typing import List

import point_cloud_utils as pcu

from accelerate.utils import ProjectConfiguration
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from accelerate import Accelerator, DistributedDataParallelKwargs

import numpy as np
import logging
import argparse
import shutil
import wandb
import torch
import os
import pdb
import glob
import trimesh

import sys
sys.path.append("/scratch/bbqk/haozhang/files/physdreamer/PhysDreamer")

from motionrep.utils.config import create_config
from motionrep.utils.optimizer import get_linear_schedule_with_warmup
from omegaconf import OmegaConf
from PIL import Image
import imageio
import numpy as np
from chamferdist import ChamferDistance # Haolan:chamferdist lib

# from motionrep.utils.torch_utils import get_sync_time
from einops import rearrange, repeat

from motionrep.gaussian_3d.gaussian_renderer.feat_render import render_feat_gaussian
from motionrep.gaussian_3d.scene import GaussianModel
from motionrep.fields.se3_field import TemporalKplanesSE3fields

from motionrep.data.datasets.multiview_dataset import MultiviewImageDataset
from motionrep.data.datasets.multiview_video_dataset import (
    MultiviewVideoDataset,
    camera_dataset_collate_fn,
)

from motionrep.data.datasets.multiview_dataset import (
    camera_dataset_collate_fn as camera_dataset_collate_fn_img,
)

from typing import NamedTuple
import torch.nn as nn
import torch.nn.functional as F

from motionrep.utils.img_utils import compute_psnr, compute_ssim
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

from exp_motion.train.cuboid_utils import (
    shrink_adjacent_cuboids,
    scale_endpoint_cuboids,
    cuboid_finding,
    assign_cuboid_velocity,
)


from interface import (
    MPMDifferentiableSimulationRig,
)

logger = get_logger(__name__, log_level="INFO")


class Trainer:
    def __init__(self, args):
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.args = args
        
        self.num_frames = int(args.num_frames)
        
        self.window_size = self.num_frames
        

        
        # output
        args.wandb_name += (
            "youngs_{}_mask_youngs_{}_substep_{}_sw_{}".format(
                args.youngs,
                args.mask_youngs,
                args.substep,
                self.window_size,
            )
        )
        
        # path
        self.output_dir = os.path.join(args.output_dir, args.wandb_name)
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        dataset_dir = args.dataset_dir
        
        gtmesh_dir = os.path.join(dataset_dir, "gt")
        
        self.sim_path = os.path.join(dataset_dir, "infilled/infilled_0.ply")
        
        # gt and sim
        self.gt_meshes = load_mesh_sequences(gtmesh_dir)
        
        self.xyzs = pcu.load_mesh_v(self.sim_path)
        
        self.num_particles = torch.tensor(self.xyzs, dtype=torch.float32, device=self.device).shape[0]
        
        E_nu_list = self.init_trainable_params()
        
        self.E_nu_list = E_nu_list
        
        self.mask_youngs = args.mask_youngs
        
        self.step = 0
            
        self.setup_simulation(dataset_dir, grid_size=args.grid_size)
        
        self.youngs_modulus_min = []
        self.youngs_modulus_mean = []
        self.youngs_modulus_max = []
        
    def init_trainable_params(self,):

        # init young modulus and poisson ratio

        youngs_modulus = torch.ones(
            self.num_particles, device=self.device, dtype=torch.float32) * args.youngs

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
        
        sim_mask = os.path.join(dataset_dir, "sim_mask/whale_tail.ply")
        
        sim_mask_indices = match_point(self.sim_path, sim_mask)
        
        print(f"匹配到的点数: {sim_mask_indices.shape[0]}")
        
        
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
        gt_mesh_initial = os.path.join(dataset_dir, "gt/gt_1.ply")

        self.gt_mask_indices = match_point(self.sim_path, gt_mesh_initial)

        print(f"Number of matched indices: {self.gt_mask_indices.shape[0]}") # 获取匹配的索引数量
        
        # 对gt和mesh进行scale和shift
        
        sim_xyzs = (sim_xyzs + shift) / scale
        
        self.gt_meshes = [
            (mesh + shift) / scale for mesh in self.gt_meshes
        ]
            
        bbox = calculate_bounding_box(self.gt_meshes)

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
        grid_lim = max_abs_value * 1.2
        grid_dx = grid_lim / grid_size
        print(f"Grid Size: {grid_size}; Grid Lim: {grid_lim}")
        
        # Haolan:初始化 cuboid 参数
        sk_path = os.path.join(dataset_dir, "skinning_masks.npy")
        
        skin_weights = np.load(sk_path)  # shape: (num_bones, num_vertices = 3525)
    
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
        
        cuboid_centers, cuboid_sizes, cuboid_types, boundary_points_list = cuboid_finding(preprocessed_vertices, assignment, grid_dx)
    
    
        # 检查 cuboid_finding 是否成功找到立方体
        if len(cuboid_centers) == 0 or len(cuboid_sizes) == 0:
            raise ValueError("cuboid_finding 未能找到有效的 cuboid，请检查输入顶点。")

            
        self.cuboid_point = torch.tensor(cuboid_centers, dtype=torch.float32, device=device)
        self.cuboid_size = torch.tensor(cuboid_sizes, dtype=torch.float32, device=device)

        
#         ply_path = os.path.join(dataset_dir, f"gt/gt_1.ply")
        
#         # part可视化
#         visualize_parts(ply_path, self.masks, dataset_dir)
        
#         # pdb.set_trace()

#         # cuboid可视化
#         output_html_path = os.path.join(dataset_dir, "cuboid_visualization.html")

#         # 调用函数
#         visualize_cuboids(
#             selected_vertices=preprocessed_vertices,
#             assignment=assignment,
#             cuboid_centers=cuboid_centers,
#             cuboid_sizes=cuboid_sizes,
#             output_html_path=output_html_path
#         )
        
#         pdb.set_trace()
        
        
        self.cuboid_velocity = assign_cuboid_velocity(
            cuboid_centers=cuboid_centers,
            cuboid_sizes=cuboid_sizes,
            cuboid_types=cuboid_types,
            boundary_points_list=boundary_points_list,
            total_frames = self.num_frames,
            gt_meshes=self.gt_meshes,
            vertices=preprocessed_vertices,
            vertices_assignment=assignment,
            grid_dx=grid_dx,
            device=device
        )
        
#         # 调试打印
#         for idx, velocity_tensor in enumerate(self.cuboid_velocity):
#             # print(f"Index {idx}: Shape: {velocity_tensor.shape}; Value: {velocity_tensor}")
#             print(f"Index {idx}: Shape: {velocity_tensor.shape}")
    
#         pdb.set_trace()

        mpm_state = MPMStateStruct()
        mpm_state.init(self.num_particles, device=device, requires_grad=True)

        self.particle_init_position = sim_xyzs.clone()
    
        
        # pdb.set_trace()
        
        # grid_lim决定了模拟场的大小
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
        
        density = (
            torch.ones_like(self.particle_init_position[..., 0])
            * material_params["density"]
        )
        
        self.density = density
        
        # 应用掩码赋值
        
        self.E_nu_list[0][sim_mask_indices] = torch.tensor(
            self.mask_youngs, dtype=torch.float32, device=device
        )

        # 检查赋值后的数量
        unique_indices = torch.unique(sim_mask_indices)
        
        num_masked_points = (self.E_nu_list[0] == self.mask_youngs).sum().item()
        print(f"Total assigned points: {num_masked_points} (Expected: {unique_indices.shape[0]})")
        
        # set density, youngs, poisson
        mpm_state.reset_density(
            density.clone(),
            torch.ones_like(density).type(torch.int),
            device,
            update_mass=True,
        )
        mpm_solver.set_E_nu_from_torch(
            mpm_model, self.E_nu_list[0].clone(), self.E_nu_list[1].clone(), device
        )
        mpm_solver.prepare_mu_lam(mpm_model, mpm_state, device)

     

    def get_simulation_input(self, device):
        """
        Outs: All padded
            density: [N]
            youngs_modulus: [N]
            poisson_ratio: [N]
            velocity: [N, 3]
            query_mask: [N]
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

        youngs_modulus = self.E_nu_list[0]
        
        # print("sim field: ", sim_params[..., 0])

        density = self.density

        poisson = self.E_nu_list[1]

        return density, youngs_modulus, poisson
    
    
    def train_one_step(self):
        
        device = self.device
        
        iteration_start_time = time.time() # 用于记录时间
        
        window_size = self.num_frames - 1
        
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


        delta_time = 1/25 
        substep_size = delta_time / self.args.substep
        num_substeps = int(delta_time / substep_size)

        temporal_stride = self.args.stride

        self.initial_cuboid_point = self.cuboid_point.clone()
  
        if temporal_stride < 0 or temporal_stride > window_size:
            temporal_stride = window_size
        
        print(f"Current Young Modulus: is {youngs_modulus[0]}")
        
        self.cuboid_point = self.initial_cuboid_point.detach()
        
        frame_time_offset = 0.0
        
        frame_times = []
  
        for start_time_idx in range(0, window_size, temporal_stride):
            
            frame_start_time = time.time()
            
            end_time_idx = min(start_time_idx + temporal_stride, window_size)
            
            num_step_with_grad = num_substeps * (end_time_idx - start_time_idx)
            
            gt_frame = self.gt_meshes[start_time_idx + 1] # Haolan：mesh sequences作为监督，下一帧作为上一帧模拟结果的监督
            
            if start_time_idx != 0:
                density, youngs_modulus, poisson = self.get_material_params(device)
            
                
            print(f"Processing frames from {start_time_idx} to {end_time_idx}")
            
            current_cuboid_velocity = torch.stack([
                cuboid_velocity[start_time_idx, 0, :]  # 提取第 start_time_idx 帧的速度 (3,)
                for cuboid_velocity in self.cuboid_velocity  # 遍历每个 cuboid 的张量
            ], dim=0)  # 堆叠成 (num_cuboid, 3)
            
            
            # 保留第一帧
            if start_time_idx == 0:
                sim_points = particle_pos[self.gt_mask_indices].detach()
                sim_points = sim_points * self.scale - self.shift
                sim_points_full = particle_pos.detach()
                sim_points_full = sim_points_full * self.scale - self.shift
                points_list = [sim_points]
                save_ply(points_list, self.step, self.output_dir, frame_idx=start_time_idx, for_gt=True)
            
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
                    self.E_nu_list[1],
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
            
            # Haolan:每个frame模拟结束，累积全局时
            frame_time_offset += delta_time

            sim_points = particle_pos[self.gt_mask_indices].detach()
            sim_points = sim_points * self.scale - self.shift
            sim_points_full = particle_pos.detach()
            sim_points_full = sim_points_full * self.scale - self.shift
            points_list = [sim_points]
            save_ply(points_list, self.step, self.output_dir, frame_idx=start_time_idx+1, for_gt=True)
            
        
            particle_pos, particle_velo, particle_F, particle_C = (
                particle_pos.detach(),
                particle_velo.detach(),
                particle_F.detach(),
                particle_C.detach(),
            )

            frame_index = f"{start_time_idx}_{end_time_idx}"
            
            
            
#             # 可视化每一帧的cuboid所在位置
#             frame_output_dir = os.path.join(self.output_dir, f"cuboid_visualization/step_{self.step:04d}")
#             os.makedirs(frame_output_dir, exist_ok=True)

#             # 每帧保存一个单独的 HTML 文件
#             frame_html_path = os.path.join(frame_output_dir, f"step_{self.step:04d}_frame_{start_time_idx:03d}_cuboid.html")
            
#             visualize_points = particle_pos[self.gt_mask_indices].clone().cpu().numpy()
#             # visualize_points = gt_frame.cpu().numpy()
#             cuboid_centers_np = self.cuboid_point.detach().cpu().numpy()
#             cuboid_sizes_np = self.cuboid_size.detach().cpu().numpy()

#             # 调用可视化函数
#             visualize_cuboids(
#                 selected_vertices=visualize_points,
#                 assignment=self.assignment,
#                 cuboid_centers=cuboid_centers_np,
#                 cuboid_sizes=cuboid_sizes_np,
#                 output_html_path=frame_html_path
#             )

#             print(f"Step {self.step}, Frame {start_time_idx}: Cuboid visualization saved to {frame_html_path}")
            
#             # pdb.set_trace()
            
            frame_end_time = time.time()
            frame_duration = frame_end_time - frame_start_time
            frame_times.append((
                start_time_idx, end_time_idx, frame_duration
            ))
        
        self.youngs_modulus_min.append(youngs_modulus.min().item())
        self.youngs_modulus_mean.append(youngs_modulus.mean().item())
        self.youngs_modulus_max.append(youngs_modulus.max().item())
        
        
        print(
            "nu: ",
            self.E_nu_list[1].mean().item(),
            "young_min:",
            youngs_modulus.min().item(),
            "young_max:",
            youngs_modulus.max().item(),
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
        
        self.train_one_step()


    
     

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/scratch/bbqk/haozhang/files/physdreamer/PhysDreamer/exp_motion/train/config.yml")

    # dataset params
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="../../data/whale",
    )
    parser.add_argument("--video_dir_name", type=str, default="videos")
    parser.add_argument(
        "--dataset_res",
        type=str,
        default="large",  # ["middle", "small", "large"]
    )
    parser.add_argument(
        "--motion_model_path",
        type=str,
        default=None,  # not used
        help="path to load the pretrained motion model from",
    )
    
    parser.add_argument("--youngs", type=float, default=6e4)
    parser.add_argument("--nu", type=float, default=0.3)
    parser.add_argument("--mask_youngs", type=float, default=3e3, help="Young's modulus value for masked points")
    parser.add_argument("--model", type=str, default="se3_field")
    parser.add_argument("--feat_dim", type=int, default=64)
    parser.add_argument("--num_decoder_layers", type=int, default=3)
    parser.add_argument("--decoder_hidden_size", type=int, default=64)
    parser.add_argument("--spatial_res", type=int, default=32)
    parser.add_argument("--zero_init", type=bool, default=True)

    parser.add_argument("--entropy_cls", type=int, default=-1)
    parser.add_argument("--entropy_reg", type=float, default=1e-2)

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

    # loss parameters
    parser.add_argument("--tv_loss_weight", type=float, default=1e-4)
    parser.add_argument("--ssim", type=float, default=0.9)

    # Logging and checkpointing
    parser.add_argument("--output_dir", type=str, default="../../output/whale")
    parser.add_argument("--log_iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)

    # training parameters
    parser.add_argument("--num_splits", type=int, default=0)
    parser.add_argument("--train_iters", type=int, default=1) # For gt
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
    )


    # wandb parameters
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_entity", type=str, default="mit-cv")
    parser.add_argument("--wandb_project", type=str, default="inverse_sim")
    parser.add_argument("--wandb_iters", type=int, default=10)
    parser.add_argument("--wandb_name", type=str, required=True)
    parser.add_argument("--run_eval", action="store_true", default=False)
    parser.add_argument("--load_sim", action="store_true", default=False)
    parser.add_argument("--test_convergence", action="store_true", default=False)
    parser.add_argument("--update_velo", action="store_true", default=False)
    parser.add_argument("--eval_iters", type=int, default=8)
    parser.add_argument("--eval_ys", type=float, default=1e6)
    parser.add_argument("--demo_name", type=str, default="demo_3sec_sv_gres48_lr1e-2")
    parser.add_argument("--velo_scaling", type=float, default=5.0)

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
    
    print("Generate GT Data")
    
    # pdb.set_trace()
    
    args = parse_args()
    
    trainer = Trainer(args)

    trainer.train()
