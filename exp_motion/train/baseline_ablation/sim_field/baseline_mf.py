
'''
Haolan：

loss从l2 loss改为Chamfer Distance，监督是mesh表面的点，也能学好

改更新youngs tensor为更新sim fields

smp weights越大，youngs数值差异越大，这可能是学不同部分不同youngs的关键

优化速度和材质联合学习，先分开预训练，再联合微调【前n个iteration只更新一个参数，然后材质和速度一起更新】

是不是两个optimizer和scheduler更好呢

TO-DO:

1.测试不同部分不同youngs的情况 (l2和CD loss) (now)

对于loss为啥后期反而上升，有说法是模拟步太少，导致dt太大，从而模拟不稳定，导致梯度混乱：不同gt不同效果

现在测试发现sim field的空间分布能力并不好，该怎么优化呢

1.1 尝试优化MLP网络，目前改了sim field的resolution ：效果没改变

1.2 embedding part id：不对，应该只有mesh表面点有id，内部点没有映射

1.3 用cat替换sum来提高特征维度：只是让max和min的差距变小，但并没有其他改变

1.4 检查计算图中的问题，修改了loss_wp，记录每个点的l2，再相加得到loss_wp:没有解决问题

1.5 设计局部加权chamfer distance，基于预测得到的skinning weight，权重则由该部分运动幅度决定，幅度越大，权重越大，幅度越小，权重越小

*可能重点正在监督上，缺失局部监督以及信息，导致只能全局学习，无法有效学习到局部信息

1.6 去掉smooth loss: 依然不对

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

from exp_motion.train.local_utils import (
    get_volume,
    create_spatial_fields,
    IntervalAnneal_cuboid,
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

from exp_motion.train.new2_utils import (
    smooth_clip_gradients,
    get_knn_neighbors,
    compute_smoothness_loss,
    get_gradient_percentiles,
    get_points_in_cuboid_sphere,
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

class Trainer:
    def __init__(self, args):
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.args = args
        
        self.num_frames = int(args.num_frames)
        
        self.window_size = self.num_frames
        
        # output
        args.wandb_name += (
            "youngs_{}_lr_{}_substep_{}_iters_{}_sw_{}".format(
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
        
        gtmesh_dir = os.path.join(dataset_dir, "6e3gt")
        
        gtmesh_dir_og = os.path.join(dataset_dir, "gt_mesh")
        
        self.sim_path = os.path.join(dataset_dir, "infilled/infilled_0.ply")
        
        # gt and sim
        self.gt_meshes = load_mesh_sequences(gtmesh_dir)
        
        self.gt_meshes_og = load_mesh_sequences(gtmesh_dir_og)
        
        self.xyzs = pcu.load_mesh_v(self.sim_path)
        
        self.num_particles = torch.tensor(self.xyzs, dtype=torch.float32, device=self.device).shape[0]
        
        
        # setup simulation
        
        self.delta_time = 1/25
        
        self.velo_factor = args.velo_factor
        
        E_nu_list = self.init_trainable_params()
        
        self.E_nu_list = E_nu_list
        
        self.smp_weights = args.smp_weights
        
        self.setup_simulation(dataset_dir, grid_size=args.grid_size)
        
        self.youngs_modulus_min = []
        self.youngs_modulus_mean = []
        self.youngs_modulus_max = []
        
        # setup training
        
        self.entropy_reg = args.entropy_reg
        
        self.train_iters = args.train_iters
        
        # self.trainable_params = list(self.sim_fields.parameters()) + list(self.cuboid_velocity)
        
        self.trainable_params = list(self.sim_fields.parameters())
        
        optim_list = [                
            {
                "params": self.sim_fields.parameters(),
                "lr": args.lr,
                "weight_decay": 0.0,
            },
        ]
            
        # self.optimizer = torch.optim.AdamW(
        #     optim_list,
        #     lr=args.lr,
        #     weight_decay=0.0,
        # )
        
        self.material_optimizer = torch.optim.SGD(
            optim_list,
            lr=args.lr,  
            momentum=0.0,  # 添加动量项，提高收敛速度
            weight_decay=0.0,  # 权重衰减
        )
        
        self.material_scheduler = get_linear_schedule_with_warmup(
            optimizer=self.material_optimizer,
            num_warmup_steps=args.warmup_step,
            num_training_steps=args.train_iters,
        )
        
        # self.velocity_optimizer = torch.optim.SGD(
        #     self.cuboid_velocity,
        #     lr=args.lr,
        #     momentum=0.0,
        #     weight_decay=0.0,
        # )
        # self.velocity_scheduler = get_linear_schedule_with_warmup(
        #     optimizer=self.velocity_optimizer,
        #     num_warmup_steps=args.warmup_step,
        #     num_training_steps=args.train_iters,
        # )

        self.step = 0

        self.max_grad_norm = args.max_grad_norm
        

    def init_trainable_params(self,):
        
        youngs_modulus = torch.ones(
            self.num_particles, device=self.device, dtype=torch.float32) * args.youngs
            
        poisson_ratio = torch.ones(
            self.num_particles, device=self.device, dtype=torch.float32) * args.nu
        
        trainable_params = [youngs_modulus, poisson_ratio]

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
        gt_mesh_initial = os.path.join(dataset_dir, "./6e3gt/gt_0.ply")
        
        # gt_mesh_initial = os.path.join(dataset_dir, "gt_mesh/gt_1.ply")

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
        
        bbox = calculate_bounding_box(sim_xyzs) # 确保第一帧infill的ply是一致的

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
        sk_path = os.path.join(dataset_dir, "skinning_masks.npy")
        
        skin_weights = np.load(sk_path)  # shape: (num_bones, num_vertices = 3525)
    
        self.masks = torch.from_numpy(skin_weights).to("cuda")
        
        num_parts = self.masks.shape[0]
        
        self.part_id = torch.argmax(self.masks.float(), dim=0).long().to(device)

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

        
        # ply_path = os.path.join(dataset_dir, f"./train_velo/gt/gt_0.ply")
        
        # part可视化
#         visualize_parts(ply_path, self.masks, dataset_dir)
        
#         pdb.set_trace()

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
        
        for i in range(len(self.cuboid_velocity)):
            self.cuboid_velocity[i].data *= self.velo_factor
        
#         # 调试打印
#         for idx, velocity_tensor in enumerate(self.cuboid_velocity):
#             # print(f"Index {idx}: Shape: {velocity_tensor.shape}; Value: {velocity_tensor}")
#             print(f"Index {idx}: Shape: {velocity_tensor.shape}")
    
#         pdb.set_trace()


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
        
        # youngs_modulus = torch.exp(self.E_nu_list[0])
        
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
        
        sim_aabb = torch.stack(
            [torch.min(sim_xyzs, dim=0)[0], torch.max(sim_xyzs, dim=0)[0]], dim=0
        )
        
        sim_aabb = (
            sim_aabb - torch.mean(sim_aabb, dim=0, keepdim=True)
        ) * 1.5 + torch.mean(sim_aabb, dim=0, keepdim=True)
        
        self.args.sim_res = 48
        
        self.sim_fields = create_spatial_fields(self.args, 1, sim_aabb)
        
        self.sim_fields = self.sim_fields.to(device)
        
        self.sim_fields.train()
   

    def get_simulation_input(self, device):
        """
        Outs: All padded
            density: [N]
            youngs_modulus: [N]
            poisson_ratio: [N]
            velocity: [N, 3]
        """

        density, youngs_modulus, poisson, entropy = self.get_material_params(device)
        
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
            entropy,
        )

    def get_material_params(self, device):
        
        initial_position = self.particle_init_position.clone()
         
        sim_params = self.sim_fields(initial_position)
        
        entropy = torch.zeros(1).to(sim_params.device)
            
        sim_params = sim_params * self.smp_weights

        # print(f"Sim_params Weights: {self.smp_weights}")
        
        youngs_modulus = self.E_nu_list[0] + sim_params[..., 0]
        
        # youngs_modulus = torch.clamp(youngs_modulus, 1000.0, 5e8) # 不能对叶子节点使用

        density = self.density

        poisson = self.E_nu_list[1]

        return density, youngs_modulus, poisson, entropy
    
    
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
            entropy,
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
                density, youngs_modulus, poisson, entropy = self.get_material_params(device)
            
                
            print(f"Processing frames from {start_time_idx} to {end_time_idx}")
            
            current_cuboid_velocity = self.cuboid_velocity[start_time_idx]
            
            # # for debug
#             if self.step == 0 or self.step == self.train_iters - 1:
#                 if start_time_idx == 0:
#                     target_points_0 = self.gt_meshes[0]
#                     sim_points_full = particle_pos.clone()
#                     sim_points = sim_points_full[self.gt_mask_indices].clone()
#                     sim_points_full = sim_points_full * self.scale - self.shift
#                     target_points_0 = target_points_0 * self.scale - self.shift
#                     sim_points = sim_points * self.scale - self.shift
#                     points_list = [sim_points]
#                     save_ply(points_list, self.step, self.output_dir, frame_idx=start_time_idx)
                
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
            
            predicted_points_cd = particle_pos[self.gt_mask_indices].clone()  
            target_points_cd = gt_frame
            
#             # local CD loss
            
#             beta = 2000
#             local_loss = 0.0
#             num_valid_parts = 0
            
#             unique_parts = torch.unique(self.part_id)
            
#             for p in unique_parts:

#                 indices = (self.part_id == p).nonzero(as_tuple=True)[0]
#                 # 如果该区域点数较少，跳过（可选）
#                 if indices.numel() < 5:
#                     continue
#                 # 取出该部分的预测点和 GT 点
#                 pred_part = predicted_points_cd[indices]  # [Np, 3]
#                 target_part = target_points_cd[indices]     # [Np, 3]

#                 # 计算该区域的 Chamfer Loss（注意 chamfer_dist 接受的输入 shape 为 [B, N, 3]）
#                 part_loss = chamfer_dist(pred_part.unsqueeze(0), target_part.unsqueeze(0))

#                 # 计算该区域的运动幅度：取该部分在 GT 中上一帧与当前帧的平均位移
#                 gt_part_old = old_gt_frame[indices]
#                 gt_part_current = gt_frame[indices]
#                 # 计算 L2 距离（位移）并取均值
#                 motion = torch.mean(torch.norm(gt_part_current - gt_part_old, dim=1))
#                 # 定义该区域权重（运动越大，权重越大）
#                 weight = 1.0 + beta * motion
#                 # 加权累加该区域 loss
#                 local_loss += weight * part_loss
#                 num_valid_parts += 1
#                 # print(f"Part {p} weight: {weight}")
            
            
#             if num_valid_parts > 0:
#                 local_loss = local_loss / num_valid_parts
#             else:
#                 local_loss = 0.0

                
            # global CD loss             

            global_loss = chamfer_dist(predicted_points_cd.unsqueeze(0), target_points_cd.unsqueeze(0))
            loss = global_loss * 1000
            
            # loss = loss * (self.args.loss_decay**end_time_idx)
            
            # print(f"Global Loss: {global_loss}; Local Loss: {local_loss}")
            
            # # for debug
            # target_points = gt_frame
            # sim_points_full = particle_pos.clone()
            # sim_points = sim_points_full[self.gt_mask_indices].clone()
            # sim_points_full = sim_points_full * self.scale - self.shift
            # target_points = target_points * self.scale - self.shift
            # sim_points = sim_points * self.scale - self.shift
            # points_list = [target_points]
            # save_ply(points_list, self.step, self.output_dir, frame_idx=start_time_idx+1)
            
            

            
            # displacement = (gt_frame - old_gt_frame).norm(dim=1)
            # alpha = 6000.0 
            # weight = 1.0 + alpha * displacement
            
            # print(f"[DEBUG] weight -> min={weight.min().item():.4f}, max={weight.max().item():.4f}, mean={weight.mean().item():.4f}")
            
            # pdb.set_trace()
            
            # diff = ((particle_pos[self.gt_mask_indices] - gt_frame) ** 2).sum(dim=1)
            # # weighted_diff = diff * weight
            # # l2_loss = weighted_diff.mean()
            # l2_loss = diff.mean()
            # l2_weight = 1e6
            # loss = l2_loss * l2_weight
            
            sm_loss = self.sim_fields.compute_smoothess_loss()
            sm_loss_weight = 1e-2
            
            loss = loss + sm_loss * sm_loss_weight
            
            # loss = loss + entropy * self.entropy_reg
            
            # pdb.set_trace()
            
            print(f"Frame Loss: {loss.item()}")

            loss.backward()
                
            # if self.step == 0 or self.step == self.train_iters - 1:
            #     target_points = gt_frame.clone()
            #     sim_points_full = particle_pos.clone()
            #     sim_points = sim_points_full[self.gt_mask_indices].clone()
            #     sim_points_full = sim_points_full * self.scale - self.shift
            #     target_points = target_points * self.scale - self.shift
            #     sim_points = sim_points * self.scale - self.shift
            #     points_list = [sim_points]
            #     save_ply(points_list, self.step, self.output_dir, frame_idx=start_time_idx+1)
            
            
            # print(youngs_modulus.grad_fn)
            
            # print(f"Velocity: {current_cuboid_velocity}")

            # 应该检查叶子节点梯度
            
            # grad_check_path = os.path.join(self.output_dir, "./gradcheck")
            
            # check_gradients(
            #     start_time_idx,  # 位置参数
            #     grad_check_path,  # 位置参数
            #     ("self.E_nu_list[0]", self.E_nu_list[0])  # 位置参数
            # )
            
            # check_gradients(
            #     ("current cuboid velocity", self.cuboid_velocity[start_time_idx]),
            # )
        

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
        

        _, youngs_modulus, _, _ = self.get_material_params(device)
        
        youngs_modulus_data = youngs_modulus.clone()
        
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
        
        # n_pretrain_mat = self.train_iters # 至訓練材質
        # n_pretrain_vel = 0

        for iteration in tqdm(range(self.train_iters), desc="Training progress"):

            # # 分阶段：
            # if iteration < n_pretrain_vel:
            #     # 前 n_pretrain_mat 轮 => 只训练速度
            #     self.mode_of_training = "velocity"
            # elif iteration < (n_pretrain_mat + n_pretrain_vel):
            #     # 接下来 n_pretrain_vel 轮 => 只训练材质
            #     self.mode_of_training = "material"
            # else:
            #     # 剩余轮数 => 联合优化
            #     self.mode_of_training = "joint"

            self.train_one_step()
            self.step += 1
        
        
        
        plot_losses(self.output_dir, num_iterations=self.train_iters) 
        plot_youngs_modulus(self.output_dir, self.youngs_modulus_mean, self.youngs_modulus_max, self.youngs_modulus_min)
        
        # 可视化youngs modulus
        init_pos = self.particle_init_position.detach().clone()
        final_young = self.final_youngs_modulus.detach().clone()
        
        final_ply_path = os.path.join(self.output_dir, "youngs_map.ply")
        final_html_path = os.path.join(self.output_dir, "youngs_map.html")
        final_csv_path = os.path.join(self.output_dir, "youngs_map.csv")
        
        visualize_youngs_modulus( 
            positions=init_pos,
            young_values=final_young,
            ply_file_path=final_ply_path,
            html_file_path=final_html_path,
            csv_file_path=final_csv_path,
            colormap_name='viridis', # 低值是深紫，高值是亮黄
        )
        # self.save()


    
     

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
    parser.add_argument("--smp_weights", type=float, default=1e4)
    parser.add_argument("--velo_factor", type=float, default=1.0)
    parser.add_argument("--gt_youngs", type=float, default=1e5)
    
    parser.add_argument("--entropy_cls", type=int, default=-1)
    parser.add_argument("--entropy_reg", type=float, default=1e-2)
    
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
