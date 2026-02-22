'''
Haolan：
v3在改为我们的case

初始化：读npy文件，留接口

改粒子读入和gt读入的位置，然后改loss更新

读入的meshes是infilled，引入外表皮的mask用于loss
'''

import argparse
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
from time import time
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
import torch.nn.functional as F
from torch.nn.functional import pairwise_distance # Haolan

from motionrep.utils.img_utils import compute_psnr, compute_ssim
from thirdparty_code.warp_mpm.mpm_data_structure import (
    MPMStateStruct,
    MPMModelStruct,
    get_float_array_product,
)
from thirdparty_code.warp_mpm.mpm_solver_diff_cuboid import MPMWARPDiff
from thirdparty_code.warp_mpm.warp_utils import from_torch_safe
from thirdparty_code.warp_mpm.gaussian_sim_utils import get_volume
import warp as wp
import random

from local_utils import (
    cycle,
    load_motion_model,
    create_motion_model,
    create_spatial_fields,
    find_far_points,
    LinearStepAnneal,
    IntervalAnneal,
    apply_grid_bc_w_freeze_pts,
    render_gaussian_seq_w_mask_cam_seq,
    downsample_with_kmeans_gpu,
    render_gaussian_seq_w_mask_with_disp,
)

# Haolan：整合新加入的函数
from new_utils import (
    find_points_within_threshold,
    preprocess_mesh,
    split_part_into_subparts,
    split_bone_from_skin,
    cuboid_finding,
    bone_from_part_assignment,
    chamfer_distance,
    arap,
    pairwise_distance,
    check_gradients,
    save_ply,
    log_losses, 
    plot_losses,
)

from cuboid_visualizer import (
    visualize_mesh_and_cuboids,
)

from interface_v3 import (
    MPMDifferentiableSimulationRig,
)

logger = get_logger(__name__, log_level="INFO")

model_dict = {
}


class Trainer:
    def __init__(self, args):
        self.args = args

        self.ssim = args.ssim
        args.warmup_step = int(args.warmup_step * args.gradient_accumulation_steps)
        
        print("Training warmup_step:", args.warmup_step)
        
        args.train_iters = int(args.train_iters * args.gradient_accumulation_steps)
        os.environ["WANDB__SERVICE_WAIT"] = "600"
        # Haolan: save文件夹名称
        args.wandb_name += (
            "decay_{}_substep_{}_{}_lr_{}_tv_{}_iters_{}_sw_{}_cw_{}_split_{}".format(
                args.loss_decay,
                args.substep,
                args.model,
                args.lr,
                args.tv_loss_weight,
                args.train_iters,
                args.start_window_size, # 时间窗大小，保存了整个模拟的历史状态
                args.compute_window, # 计算窗口大小，当前训练步中会进行计算和优化的时间范围
                args.num_splits,
            )
        )
        
        self.loss_history = {
            "total_loss": [],
            "chamfer_loss": [],
            "arap_loss": [],
            "sm_loss": [],
            "entropy_loss": [],
        }
            
        logging_dir = os.path.join(args.output_dir, args.wandb_name)
        
        # Haolan:记录loss
        self.output_dir = logging_dir
        
        accelerator_project_config = ProjectConfiguration(logging_dir=logging_dir)
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(
            gradient_accumulation_steps=1,  # args.gradient_accumulation_steps,
            mixed_precision="no",
            log_with="wandb",
            project_config=accelerator_project_config,
            kwargs_handlers=[ddp_kwargs],
        )
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(accelerator.state, main_process_only=False)

        set_seed(args.seed + accelerator.process_index)
        print("process index", accelerator.process_index)
        if accelerator.is_main_process:
            output_path = os.path.join(logging_dir, f"seed{args.seed}")
            os.makedirs(output_path, exist_ok=True)
            self.output_path = output_path

        self.rand_bg = args.rand_bg
            
        self.num_frames = int(args.num_frames)
        
        dataset_dir = args.dataset_dir
        
        # Haolan：gt meshes以及gt_mask
        gtmesh_dir = "/scratch/bbqk/haozhang/files/physdreamer/PhysDreamer/data/carnations/walk/gt"
        self.gt_meshes = self.load_mesh_sequences(gtmesh_dir)
        
        # Haolan：新的时间窗口逻辑
        self.window_size_schduler = IntervalAnneal(
            args.train_iters, # Haolan：需要计算设定多少训练迭代次数，才能保证每帧都训练到
            start_state=[args.start_window_size], # window_size从1开始
            end_state=[self.num_frames - 1],
            plateau_iters=0, # Haolan：-1则默认为train_iters的20%，目前代码没有使用
            warmup_step=0, # Warmup:5
        )
        
        self.previous_velocity = None
        

        self.train_iters = args.train_iters
        self.accelerator = accelerator
        # init traiable params
        E_nu_list = self.init_trainable_params()
        
        for p in E_nu_list:
            p.requires_grad = True
        self.E_nu_list = E_nu_list

        self.setup_simulation(dataset_dir, grid_size=args.grid_size)
        
        # Haolan：不再训练velo_field，而是训练cuboid_velocity

        trainable_params = list(self.sim_fields.parameters()) + self.E_nu_list + [self.cuboid_velocity]
        optim_list = [
            {"params": self.E_nu_list, "lr": args.lr * 1e-10}, # Haolan:1e-10
            {
                "params": self.sim_fields.parameters(),
                "lr": args.lr,
                "weight_decay": 1e-4,
            },
            {
                "params": [self.cuboid_velocity],
                "lr": args.lr * 1e-2,  # Haolan:根据需要调整学习率
                "weight_decay": 1e-4,
            },
        ]

        self.trainable_params = trainable_params
        
        # Haolan：初始化全局的optimizer，用于材质参数和cuboid_velocity
        self.optimizer = torch.optim.AdamW(
            optim_list,
            lr=args.lr,
            weight_decay=0.0,
        )
        

        self.scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=args.warmup_step,
            num_training_steps=args.train_iters,
        )
        
        pdb.set_trace()
        

        self.sim_fields, self.optimizer, self.scheduler = accelerator.prepare(
            self.sim_fields, self.optimizer, self.scheduler
        )

        # setup train info
        self.arap_pos = None
        self.step = 0
        self.batch_size = args.batch_size
        self.tv_loss_weight = args.tv_loss_weight

        self.log_iters = args.log_iters
        self.max_grad_norm = args.max_grad_norm

        
    def init_trainable_params(
        self,
    ):

        # init young modulus and poisson ratio

        young_numpy = np.exp(np.random.uniform(np.log(1e-3), np.log(1e3))).astype(
            np.float32
        )

        young_numpy = np.array([1e5]).astype(np.float32)

        young_modulus = torch.tensor(young_numpy, dtype=torch.float32).to(
            self.accelerator.device
        )

        poisson_numpy = np.random.uniform(0.1, 0.4)
        poisson_ratio = torch.tensor(poisson_numpy, dtype=torch.float32).to(
            self.accelerator.device
        )

        trainable_params = [young_modulus, poisson_ratio]

        print(
            "init young modulus: ",
            young_modulus.item(),
            "poisson ratio: ",
            poisson_ratio.item(),
        )
        return trainable_params

    # Haolan：删去了downsample
    def setup_simulation(self, dataset_dir, grid_size=100):

        device = "cuda:{}".format(self.accelerator.process_index)
        
        # Haolan：mesh sequence
        sim_path = os.path.join(dataset_dir, "./walk/infilled/cattleAFK_WalkB_frame_0.ply")
        
        xyzs = pcu.load_mesh_v(sim_path)
        
        # Haolan：创建gt_mask
        reference_mesh = pcu.load_mesh_v(sim_path).astype(np.float32) # Haolan：需要float32
        gt_mesh_initial = self.gt_meshes[0].cpu().numpy()
        T = 1e-6
        matched_indices_ref, matched_indices_gt = find_points_within_threshold(reference_mesh, gt_mesh_initial, T)

        self.gt_mask = torch.zeros(len(reference_mesh), dtype=torch.bool, device=device)
        self.gt_mask[matched_indices_ref] = True
        
        true_points = xyzs[self.gt_mask.cpu().numpy()] # Haolan：用于测试mask正确与否
        
        # pdb.set_trace()
        
        sim_xyzs = torch.tensor(xyzs, dtype=torch.float32, device=device)
        sim_cov = torch.eye(3).expand(len(sim_xyzs), 3, 3).to(device)
        
        # scale, and shift
        pos_max = sim_xyzs.max()
        pos_min = sim_xyzs.min()
        scale = (pos_max - pos_min) * 1.8
        shift = -pos_min + (pos_max - pos_min) * 0.25
        self.scale, self.shift = scale, shift
        print("scale, shift", scale, shift)

        # filled
        filled_in_points_path = os.path.join(dataset_dir, "internal_filled_points.ply")

        if os.path.exists(filled_in_points_path):
            fill_xyzs = pcu.load_mesh_v(filled_in_points_path)  # [n, 3]
            fill_xyzs = fill_xyzs[
                np.random.choice(
                    fill_xyzs.shape[0], int(fill_xyzs.shape[0] * 0.25), replace=False
                )
            ]
            fill_xyzs = torch.from_numpy(fill_xyzs).float().to("cuda")
            self.fill_xyzs = fill_xyzs
            print(
                "loaded {} internal filled points from: ".format(fill_xyzs.shape[0]),
                filled_in_points_path,
            )
        else:
            self.fill_xyzs = None

        if self.fill_xyzs is not None:
            render_mask_in_sim_pts = torch.cat(
                [
                    torch.ones_like(sim_xyzs[:, 0]).bool(),
                    torch.zeros_like(fill_xyzs[:, 0]).bool(),
                ],
                dim=0,
            ).to(device)
            sim_xyzs = torch.cat([sim_xyzs, fill_xyzs], dim=0)
            sim_cov = torch.cat(
                [sim_cov, sim_cov.new_ones((fill_xyzs.shape[0], sim_cov.shape[-1]))],
                dim=0,
            )
            self.render_mask = render_mask_in_sim_pts
        else:
            self.render_mask = torch.ones_like(sim_xyzs[:, 0]).bool().to(device)

        sim_xyzs = (sim_xyzs + shift) / scale
        
        points_volume = get_volume(sim_xyzs.detach().cpu().numpy())
        
        num_particles = sim_xyzs.shape[0]
        
        sim_aabb = torch.stack(
            [torch.min(sim_xyzs, dim=0)[0], torch.max(sim_xyzs, dim=0)[0]], dim=0
        )
        sim_aabb = (
            sim_aabb - torch.mean(sim_aabb, dim=0, keepdim=True)
        ) * 20 + torch.mean(sim_aabb, dim=0, keepdim=True)

        print("simulation aabb: ", sim_aabb)

        wp.init()
        wp.config.mode = "debug"
        wp.config.verify_cuda = True
        

        # Haolan:初始化 cuboid 参数
        
        # Haolan：读入skinning weights
        
        # split_bone_from_skin, cuboid_finding, bone_from_part_assignment整合进 "files/physdreamer/PhysDreamer/exp_motion/train/new_utils.py"
        
        total_frames = self.num_frames
        
        frame_start = 0
        frame_end = total_frames - 1

        
        skin_path = os.path.join(dataset_dir, "./walk/skin.npy")
        skin_weights = np.load(skin_path)
        
        
        # pdb.set_trace()
        
        n_splits = args.num_splits
        
        print(f"num_splits: {n_splits}")
        
        delta_time = 1.0 / 30
        velocities = []
        
        mesh_path = os.path.join(dataset_dir, f"./walk/mesh/cattleAFK_WalkB_frame_{frame_start}.obj")
        
        mesh = trimesh.load(mesh_path)
        vertices = np.array(mesh.vertices)
        preprocessed_vertices = preprocess_mesh(vertices, shift, scale)
        
        assignment = split_bone_from_skin(preprocessed_vertices, skin_weights, n_splits)
        cuboid_centers, cuboid_sizes = cuboid_finding(preprocessed_vertices, assignment)
        
        # Haolan：给每个frame都初始化速度
        for frame_idx in range(frame_start, frame_end):
            # 当前帧的 mesh
            mesh_path_0 = os.path.join(dataset_dir, f"./walk/mesh/cattleAFK_WalkB_frame_{frame_idx}.obj")
            # 下一帧的 mesh
            mesh_path_1 = os.path.join(dataset_dir, f"./walk/mesh/cattleAFK_WalkB_frame_{frame_idx + 1}.obj")

            mesh_0 = trimesh.load(mesh_path_0)
            vertices_0 = preprocess_mesh(np.array(mesh_0.vertices), shift, scale)

            mesh_1 = trimesh.load(mesh_path_1)
            vertices_1 = preprocess_mesh(np.array(mesh_1.vertices), shift, scale)

            # 计算当前帧和下一帧的骨架中心
            bones_0 = bone_from_part_assignment(vertices_0, assignment)
            bones_1 = bone_from_part_assignment(vertices_1, assignment)

            # 计算速度
            velocity = (bones_1 - bones_0) / delta_time
            velocities.append(velocity)

        # 将所有速度合并到一个张量中
        velocities_tensor = torch.tensor(velocities, dtype=torch.float32, device=device)
        
        # print("Cuboid_velocity:", velocities_tensor)
        # pdb.set_trace()
        
        # self.velo_scale = torch.tensor(0.8, device=device) 
        # velocities_tensor = velocities_tensor * self.velo_scale
        
        self.cuboid_point = torch.tensor(cuboid_centers, dtype=torch.float32, device=device)
        self.initial_cuboid_point = None # 初始位置
        self.frame_time_offset = None # 模拟的全局时间累积
        self.size_scale = torch.tensor(0.9, device=device) # size_scale初始化为0.9
        self.cuboid_size = torch.tensor(cuboid_sizes, dtype=torch.float32, device=device) * self.size_scale
        self.cuboid_velocity = torch.nn.Parameter(velocities_tensor.clone(), requires_grad=True) 
        
        # output_html_path = "/scratch/bbqk/haozhang/files/physdreamer/PhysDreamer/data/carnations/walk/cuboid_visualization.html"
        
#         vi_cuboid_centers = self.cuboid_point.cpu().numpy().tolist()
#         vi_cuboid_sizes = self.cuboid_size.cpu().numpy().tolist()

#         visualize_mesh_and_cuboids(
#             mesh_path,
#             vi_cuboid_centers,
#             vi_cuboid_sizes,
#             shift=shift.cpu().numpy(),  # 将 shift 转换为 numpy 数组
#             scale=scale.cpu().numpy(),  # 将 scale 转换为 numpy 数组
#             output_html_path=output_html_path
#         )
        
        # pdb.set_trace()
        
        mpm_state = MPMStateStruct()
        mpm_state.init(num_particles, device=device, requires_grad=True)

        self.particle_init_position = sim_xyzs.clone()

        mpm_state.from_torch(
            self.particle_init_position.clone(),
            torch.from_numpy(points_volume).float().to(device).clone(),
            sim_cov,
            device=device,
            requires_grad=True,
            n_grid=grid_size,
            grid_lim=1.0,
        )
        mpm_model = MPMModelStruct()
        mpm_model.init(num_particles, device=device, requires_grad=True)
        mpm_model.init_other_params(n_grid=grid_size, grid_lim=1.0, device=device)

        material_params = {
            "material": "jelly",  # "jelly", "metal", "sand", "foam", "snow", "plasticine", "neo-hookean"
            "g": [0.0, 0.0, 0.0],
            "density": 2000,  # kg / m^3
            "grid_v_damping_scale": 0.999,  # 0.999,
        }

        self.v_damping = material_params["grid_v_damping_scale"]
        self.material_name = material_params["material"]
        mpm_solver = MPMWARPDiff(
            num_particles, n_grid=grid_size, grid_lim=1.0, device=device
        )
        mpm_solver.set_parameters_dict(mpm_model, mpm_state, material_params)

        self.mpm_state, self.mpm_model, self.mpm_solver = (
            mpm_state,
            mpm_model,
            mpm_solver,
        )
        
        
        # setup boundary condition:
        moving_pts_path = os.path.join(dataset_dir, "./walk/infilled/cattleAFK_WalkB_frame_0.ply")
        if os.path.exists(moving_pts_path):
            moving_pts = pcu.load_mesh_v(moving_pts_path)
            moving_pts = torch.from_numpy(moving_pts).float().to(device)
            moving_pts = (moving_pts + shift) / scale
            freeze_mask = find_far_points(
                sim_xyzs, moving_pts, thres=0.5 / grid_size
            ).bool()
            freeze_pts = sim_xyzs[freeze_mask, :]

            grid_freeze_mask = apply_grid_bc_w_freeze_pts(
                grid_size, 1.0, freeze_pts, mpm_solver
            )
            self.freeze_mask = freeze_mask

            # does not prefer boundary condition on particle
            # freeze_mask_select = setup_boundary_condition_with_points(sim_xyzs, moving_pts,
            #                                                         self.mpm_solver, self.mpm_state, thres=0.5 / grid_size)
            # self.freeze_mask = freeze_mask_select.bool()
        else:
            raise NotImplementedError

        num_freeze_pts = self.freeze_mask.sum()
        print(
            "num freeze pts in total",
            num_freeze_pts.item(),
            "num moving pts",
            num_particles - num_freeze_pts.item(),
        )
        
        # init fields for simulation, e.g. density, external force, etc.

        # padd init density, youngs,
        density = (
            torch.ones_like(self.particle_init_position[..., 0])
            * material_params["density"]
        )
        youngs_modulus = (
            torch.ones_like(self.particle_init_position[..., 0])
            * self.E_nu_list[0].detach()
        )
        poisson_ratio = torch.ones_like(self.particle_init_position[..., 0]) * 0.3

        # load stem for higher density
        stem_pts_path = os.path.join(dataset_dir, "stem_points.ply")
        if os.path.exists(stem_pts_path):
            stem_pts = pcu.load_mesh_v(stem_pts_path)
            stem_pts = torch.from_numpy(stem_pts).float().to(device)
            stem_pts = (stem_pts + shift) / scale
            no_stem_mask = find_far_points(
                sim_xyzs, stem_pts, thres=2.0 / grid_size
            ).bool()
            stem_mask = torch.logical_not(no_stem_mask)
            density[stem_mask] = 2000
            print("num stem pts", stem_mask.sum().item())

        self.density = density
        self.young_modulus = youngs_modulus
        self.poisson_ratio = poisson_ratio

        # set density, youngs, poisson
        mpm_state.reset_density(
            density.clone(),
            torch.ones_like(density).type(torch.int),
            device,
            update_mass=True,
        )
        mpm_solver.set_E_nu_from_torch(
            mpm_model, youngs_modulus.clone(), poisson_ratio.clone(), device
        )
        mpm_solver.prepare_mu_lam(mpm_model, mpm_state, device)

        self.sim_fields = create_spatial_fields(self.args, 1, sim_aabb)
        self.sim_fields.train()

        self.args.sim_res = 24
        

    def get_simulation_input(self, device):
        """
        Outs: All padded
            density: [N]
            young_modulus: [N]
            poisson_ratio: [N]
            velocity: [N, 3]
            query_mask: [N]
        """

        density, youngs_modulus, ret_poisson, entropy = self.get_material_params(device)
        initial_position_time0 = self.particle_init_position.clone()

        query_mask = torch.logical_not(self.freeze_mask)
        query_pts = initial_position_time0[query_mask, :]

        # velocity = self.velo_fields(torch.cat([query_pts, time_array.unsqueeze(-1)], dim=-1))[..., :3]
        
        # Haolan:直接将初速度设定为常量0
        # velocity = self.velo_fields(query_pts)[..., :3]
        velocity = torch.zeros_like(query_pts[..., :3], device=query_pts.device)

        # scaling
        velocity = velocity * 0.1  # not padded yet
        ret_velocity = torch.zeros_like(initial_position_time0)
        ret_velocity[query_mask, :] = velocity

        # init F, and C

        I_mat = torch.eye(3, dtype=torch.float32).to(device)
        particle_F = torch.repeat_interleave(
            I_mat[None, ...], initial_position_time0.shape[0], dim=0
        )
        particle_C = torch.zeros_like(particle_F)
        
        return (
            density,
            youngs_modulus,
            ret_poisson,
            ret_velocity,
            query_mask,
            particle_F,
            particle_C,
            entropy,
        )

    def get_material_params(self, device):

        initial_position_time0 = self.particle_init_position.detach()

        # query_mask = torch.logical_not(self.freeze_mask)
        query_mask = torch.ones_like(self.freeze_mask).bool()
        query_pts = initial_position_time0[query_mask, :]
        if self.args.entropy_cls > 0:
            sim_params, entropy = self.sim_fields(query_pts)
        else:
            sim_params = self.sim_fields(query_pts)
            entropy = torch.zeros(1).to(sim_params.device)

        sim_params = sim_params * 1000
        
        # pdb.set_trace()
        # sim_params = torch.exp(self.sim_fields(query_pts))

        # density = sim_params[..., 0]

        youngs_modulus = self.young_modulus.detach().clone()
        youngs_modulus[query_mask] += sim_params[..., 0]

        # young_modulus = torch.exp(sim_params[..., 0]) + init_young
        youngs_modulus = torch.clamp(youngs_modulus, 1000.0, 5e8)

        density = self.density.detach().clone()
        # density[self.freeze_mask] = 100000
        ret_poisson = self.poisson_ratio.detach().clone()

        return density, youngs_modulus, ret_poisson, entropy
    
    
    def train_one_step(self):

        self.sim_fields.train()
        accelerator = self.accelerator
        device = "cuda:{}".format(accelerator.process_index)
        
        # Haolan：重点在于修改训练scheduler的逻辑
        window_size = int(self.window_size_schduler.compute_state(self.step)[0]) # Haolan：schduler
        
        print(f"Window size: {window_size}")    
        
        log_loss_dict = {
            "Total_loss": [],
            "Chamfer_loss": [],
            "Arap_loss": [],
        }
            
        if self.previous_velocity is None:
            self.previous_velocity = self.cuboid_velocity.clone().detach()
            
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
            query_mask,
            particle_F,
            particle_C,
            entropy,
        ) = self.get_simulation_input(device)

        num_particles = particle_pos.shape[0]

        delta_time = 1.0 / 30  # 30 fps
        substep_size = delta_time / self.args.substep
        num_substeps = int(delta_time / substep_size)
        
        checkpoint_steps = self.args.checkpoint_steps

        temporal_stride = self.args.stride

        # Haolan：避免cuboid_point累积，并且更新全局模拟时间
        if self.step == 0:
            self.arap_pos = self.particle_init_position.clone()
            self.initial_cuboid_point = self.cuboid_point.clone()
            self.frame_time_offset = 0.0
            
        if temporal_stride < 0 or temporal_stride > window_size:
            temporal_stride = window_size

        for start_time_idx in range(0, window_size, temporal_stride):

            end_time_idx = min(start_time_idx + temporal_stride, window_size)
            
            num_step_with_grad = num_substeps * (end_time_idx - start_time_idx)
            
            gt_frame = self.gt_meshes[start_time_idx + 1] # Haolan：mesh sequences作为监督，下一帧作为上一帧模拟结果的监督
            gt_frame = (gt_frame.to(device) + self.shift) / self.scale # Haolan：注意模拟粒子的transformation
            
            # Haolan：第一帧用初始化状态，不用更新的变量注意使用detach来分离计算图
            if start_time_idx == 0:
                self.cuboid_point = self.initial_cuboid_point.detach()
            
            if start_time_idx != 0:
                density, youngs_modulus, poisson, entropy = self.get_material_params(
                    device
                )
            
            '''
            11/07/24 检查inference
            
            cuboid初始化很重要，位置和size的初始化还能优化
            
            split会让cuboid位置更好，但训练时长会很长并且帧数长了难以控制
            
            要避免cuboid相互影响
            
            可视化tranferred skinning weights和可视化cuboid已实现

            '''
            
            '''
            
            误差会累积，然后loss更新不明显。但为啥会上升，loss更新方向错误的
            
            可视化纯inference的模拟结果
            
            训练时长仍然是个很严峻的问题
            
            训练的时候不从单帧开始，而是多帧，因为多帧和单帧对于cuboid_velocity的训练没影响，但多帧开始会对材质训练很有影响，从一开始就让其通过梯度累积学习材质，可能会更快学好材质，从而影响整体效果
            
            只是完整的chamfer distance可能没办法有效训练，如果能涉及到每部分的chamfer distance，模型才知道哪些cuboid不太行，需要侧重优化，比如现在牛的右后腿
            
            修改训练逻辑：在当前window_size里面对cuboid_velocity进行梯度累积

            需要明确的是cuboid_velocity和材质参数训练是相辅相成的
            
            loss.backward()之前print参数梯度和MPMDifferentiableSimulationClean中print参数的梯度的差异来自于loss; MPMDifferentiableSimulationClean中print参数的梯度和loss.backward()之后print参数梯度的差异是来自于梯度累积：增强物理模拟参数更新的稳定性；更好地适应整个序列

            修改训练的scheduler，引入loss判断
            
            '''
                
            
            # Haolan：不冻结，每次训练都全部帧一起训练，然后引入loss判断，loss低于阈值引入下一帧
            current_cuboid_velocity = self.cuboid_velocity[start_time_idx]
            print(f"Processing frames from {start_time_idx} to {end_time_idx}, cuboid_velocity is trainable")
            
            # print(f"第{start_time_idx}帧的速度为: {self.cuboid_velocity[start_time_idx]}")
            
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
                    self.frame_time_offset, # 全局时间累积
                    density,
                    query_mask,
                    device,
                    True,
                    0,
                )
            )
            
            # Haolan：更新cuboid_point位置，detach()和计算图断开
            self.cuboid_point = (self.cuboid_point + self.cuboid_velocity[start_time_idx] * delta_time).detach()
            
            # Haolan:每个frame模拟结束，累积全局时
            self.frame_time_offset += delta_time
            
            # Haolan:loss 
            
            '''
          
            Chamfer Distance:拽着动
            
            ARAP Loss：不让动
            
            两者本质冲突，减小ARAP的权重  
            
            检查ARAP的实现
            
            '''
            chamfer_dist = ChamferDistance() # 基于chamferdist lib进行计算

            # 调整输入形状，chamferdist需要(batch_size, num_points, 3)
            predicted_points = particle_pos[self.gt_mask].unsqueeze(0).clone()  # shape: (1, num_points, 3)
            target_points = gt_frame.unsqueeze(0)  # shape: (1, num_points, 3)

            # pdb.set_trace()

            chamfer_loss = chamfer_dist(predicted_points, target_points)
            
            arap_loss = arap(particle_pos, self.arap_pos)

            self.arap_pos = particle_pos.clone().detach()
            
            loss = chamfer_loss + arap_loss * self.args.arap_weight # self.args.arap_weight = 0
            loss = loss * (self.args.loss_decay**end_time_idx)
            
            # 继续其余损失和优化更新
            sm_loss = self.sim_fields.compute_smoothess_loss()
            loss = loss + sm_loss * self.tv_loss_weight + entropy * self.args.entropy_reg
            loss = loss / self.args.compute_window

            print(f"Total Loss: {loss.item()}, Chamfer Loss: {chamfer_loss.item()}, ARAP Loss: {arap_loss.item()}, SM Loss: {sm_loss.item()}, Entropy: {entropy.item()}")
            
            
            check_gradients(
                # ("Before loss.backward() cuboid_velocity", self.cuboid_velocity),
                # ("Before loss.backward() youngs_modulus", youngs_modulus),
                # ("Before loss.backward() E_nu_list[1] (nu)", self.E_nu_list[1]),
            )
                
                
            sim_points = particle_pos[self.gt_mask].detach()
            points_list = [sim_points, gt_frame]
            # Part 0 is sim_point, Part 1 is gt_frame
            save_ply(points_list, self.step, self.args.dataset_dir, frame_idx=start_time_idx)
            
            
            
            loss.backward()
                
            # Haolan：检查梯度
            
            check_gradients(
                # ("cuboid_velocity", self.cuboid_velocity),
                # ("youngs_modulus", youngs_modulus),
                # ("E_nu_list[1] (nu)", self.E_nu_list[1]),
            )
            
            particle_pos, particle_velo, particle_F, particle_C = (
                particle_pos.detach(),
                particle_velo.detach(),
                particle_F.detach(),
                particle_C.detach(),
            )
            
            frame_index = f"{start_time_idx}_{end_time_idx}"
            
            with torch.no_grad():
                log_loss_dict["Total_loss"].append(loss.item())
                log_loss_dict["Chamfer_loss"].append(chamfer_loss.item())
                log_loss_dict["Arap_loss"].append(arap_loss.item())

                log_losses(
                    total_loss=loss.item(),
                    chamfer_loss=chamfer_loss.item(),
                    arap_loss=arap_loss.item(),
                    frame_index=frame_index
                )
                    
                
        # nu_grad_norm = self.E_nu_list[1].grad.norm(2).item()
        # spatial_grad_norm = 0
        # for p in self.sim_fields.parameters():
        #     if p.grad is not None:
        #         spatial_grad_norm += p.grad.norm(2).item()
        
        # Haolan：材质和cuboid_velocity均在这里更新
        if (
            self.step % self.gradient_accumulation_steps == 0 # gradient_accumulation_steps = 1
            or self.step == (self.train_iters - 1)
            or (self.step % self.log_iters == self.log_iters - 1)
        ):

            torch.nn.utils.clip_grad_norm_(
                self.trainable_params,
                self.max_grad_norm,
                error_if_nonfinite=False,
            )  # error if nonfinite is false
            
            self.optimizer.step()
            
            torch.cuda.empty_cache()
            
            self.optimizer.zero_grad()
            
            with torch.no_grad():
                self.E_nu_list[0].data.clamp_(1e-1, 1e8)
                self.E_nu_list[1].data.clamp_(1e-2, 0.449)
        self.scheduler.step()
        
        # threshold = 1e-6 # 用于print的阈值
        # for i, (new, old) in enumerate(zip(self.cuboid_velocity, self.previous_velocity)):
        #     diff = (new - old).abs()
        #     if diff.sum() > 0:  
        #         print(f"第 {i} 帧的速度变化: {diff}")
        
        self.previous_velocity = self.cuboid_velocity.detach().clone()
        
        for k, v in log_loss_dict.items():
            log_loss_dict[k] = np.mean(v)
        
        print(log_loss_dict)
        
        print(
            "nu: ",
            self.E_nu_list[1].item(),
            "young_mean:",
            youngs_modulus.mean().item(),
            "young_max:",
            youngs_modulus.max().item(),
            # "nu_grad_norm: ",
            # nu_grad_norm,
            # "spatial_grad_norm: ",
            # spatial_grad_norm,
        )
        
        # pdb.set_trace()
        

    def train(self):
        # might remove tqdm when multiple node
        
        for index in tqdm(range(self.step, self.train_iters), desc="Training progress"):
            self.train_one_step()
            if self.step % self.log_iters == self.log_iters - 1:
                if self.accelerator.is_main_process:
                    self.save()
                    # self.test()
            # self.accelerator.wait_for_everyone()
            self.step += 1
        
        plot_losses(self.output_dir, frame_indices=["0_1", "1_2", "2_3", "3_4", "4_5"]) # Haolan:记录loss
        
        print("Loss images saved")
        if self.accelerator.is_main_process:
            self.save()

    def load_mesh_sequences(self, mesh_dir):
        # 加载 mesh sequences 数据并返回
        mesh_files = sorted(
            glob.glob(os.path.join(mesh_dir, "cattleAFK_WalkB_frame_*_gt.ply")),
            key=lambda x: int(x.split("_")[-2])  # 按文件名中的帧序号排序
        )

        # 将每一帧的 mesh 数据加载为 torch tensor
        gt_meshes = [torch.tensor(pcu.load_mesh_v(mesh_file), dtype=torch.float32).to("cuda") for mesh_file in mesh_files]
        return gt_meshes
    
    def save(
        self,
    ):
        # training states
        output_path = os.path.join(
            self.output_path, f"checkpoint_model_{self.step:06d}"
        )
        os.makedirs(output_path, exist_ok=True)

        # 保存 sim_fields
        sim_fields_path = os.path.join(output_path, "sim_fields.pt")
        torch.save(self.accelerator.unwrap_model(self.sim_fields, keep_fp32_wrapper=True).state_dict(), sim_fields_path)

        # 保存 cuboid_velocity 的参数
        cuboid_velocity_path = os.path.join(output_path, "cuboid_velocity.pt")
        torch.save(self.cuboid_velocity.cpu(), cuboid_velocity_path)
    
    def load(self, checkpoint_dir):
        sim_fields_path = os.path.join(checkpoint_dir, "sim_fields.pt")
        print("=> loading: ", sim_fields_path)
        self.sim_fields.load_state_dict(torch.load(sim_fields_path))

        # 加载 cuboid_velocity 的参数
        cuboid_velocity_path = os.path.join(checkpoint_dir, "cuboid_velocity.pt")
        print("=> loading: ", cuboid_velocity_path)
        cuboid_velocity_data = torch.load(cuboid_velocity_path).to(self.cuboid_velocity.device)
        
        self.cuboid_velocity = torch.nn.Parameter(cuboid_velocity_data)
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yml")

    # dataset params
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="../../data/physics_dreamer/hat_nerfstudio/",
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
    parser.add_argument("--start_window_size", type=int, default=1) # Haolan：从1开始
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
    parser.add_argument("--output_dir", type=str, default="../../output/inverse_sim")
    parser.add_argument("--log_iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        # psnr 29.0
        default="../../output/inverse_sim/fast_alocasia_velopretrain_cleandecay_1.0_substep_96_se3_field_lr_0.01_tv_0.01_iters_300_sw_2_cw_2/seed0/checkpoint_model_000299",
        help="path to load velocity pretrain checkpoint from",
    )
    # training parameters
    parser.add_argument("--num_splits", type=int, default=0)
    parser.add_argument("--train_iters", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--arap_weight", 
        type=float, 
        default=0.01, 
        help="Weight for ARAP regularization",
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
    
    print("Code version is v3")
    
    # pdb.set_trace()
    
    args = parse_args()

    # torch.backends.cuda.matmul.allow_tf32 = True
    
    trainer = Trainer(args)

    if args.run_eval:
        trainer.demo(
            velo_scaling=args.velo_scaling,
            eval_ys=args.eval_ys,
            save_name=args.demo_name,
        )
    else:
        # trainer.debug()
        trainer.train()
