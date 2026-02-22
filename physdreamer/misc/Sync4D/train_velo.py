import argparse
import os
import sys
import logging
import shutil
import random
from time import time
from typing import List, NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
from jaxtyping import Float, Int, Shaped
from PIL import Image
import imageio
from einops import rearrange, repeat
import point_cloud_utils as pcu
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from omegaconf import OmegaConf
import wandb
import warp as wp
import pdb

# Third-party code
from thirdparty_code.warp_mpm.mpm_data_structure import (
    MPMStateStruct,
    MPMModelStruct,
    get_float_array_product,
)
from thirdparty_code.warp_mpm.mpm_solver_diff import MPMWARPDiff
from thirdparty_code.warp_mpm.warp_utils import from_torch_safe
from thirdparty_code.warp_mpm.gaussian_sim_utils import get_volume
from thirdparty_code.warp_mpm.backup.engine_utils import particle_position_tensor_to_ply

# Local utils
from local_utils import (
    cycle,
    load_motion_model,
    create_motion_model,
    create_spatial_fields,
    find_far_points,
    get_unrelated_parts,
    LinearStepAnneal,
    apply_grid_bc_w_freeze_pts,
    render_gaussian_seq_w_mask_cam_seq,
    render_gaussian_w_grid_sampling,
    downsample_with_kmeans_gpu,
    downsample_with_kmeans,
    grid_sample,
    render_gaussian_seq_w_mask_with_disp,
)

# Decode param
from decode_param import *

# Interface
from interface import (
    MPMDifferentiableSimulationWCheckpoint,
    MPMDifferentiableSimulationClean,
    MPMDifferentiableSimulationVelo,
)

from motionrep.utils.config import create_config
from motionrep.utils.optimizer import get_linear_schedule_with_warmup
from motionrep.utils.torch_utils import get_sync_time
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
from motionrep.utils.img_utils import compute_psnr, compute_ssim
from exp_motion.utils.camera_view_utils import *
from exp_motion.utils.motion import *
from exp_motion.utils.transformation_utils import *
from motionrep.gaussian_3d.scene.cameras import Camera as GSCamera

logger = get_logger(__name__, log_level="INFO")

class Trainer:
    def __init__(self, args):
        self.args = args
        
        # TODO: add to config
        # time params
        self.args.substep_dt = 1e-4
        self.args.frame_dt = 4e-2
        self.args.frame_num = 150
        self.temp_params = {
                "point": [
                    [0.23, 0.47, 0.2],
                    [0.47, 0.47, 0.2],
                    [0.23, 0.23, 0.2],
                    [0.47, 0.23, 0.2]
                ],
                "size": [
                    [0.1, 0.1, 0.15],
                    [0.1, 0.1, 0.15],
                    [0.1, 0.1, 0.15],
                    [0.1, 0.1, 0.15]
                ]
            }
    
        # 日志目录和加速器配置
        logging_dir = os.path.join(args.output_dir, args.name)
        accelerator_project_config = ProjectConfiguration(logging_dir=logging_dir)
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(
            gradient_accumulation_steps=1,
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

        # 数据集和评估设置
        dataset_dir = args.dataset_dir
        print(dataset_dir)
        gaussian_path = os.path.join(dataset_dir, "point_cloud", "iteration_5000", "point_cloud.ply")
        aabb = self.setup_eval(
            args,
            gaussian_path,
            dataset_dir,
            white_background=True,
        )
        self.aabb = aabb

        # 创建运动模型
        self.model = create_motion_model(
            args,
            aabb=aabb,
            num_frames=9,
        )
        self.model.eval()

        # 设置时间调度器
        self.num_frames = int(args.num_frames)
        self.window_size_schduler = LinearStepAnneal(
            args.train_iters,
            start_state=[args.start_window_size],
            end_state=[13],
            plateau_iters=-1,
            warmup_step=20,
        )

        self.train_iters = args.train_iters
        self.accelerator = accelerator

        # 初始化材料参数
        young_numpy = np.array([2e3]).astype(np.float32)
        young_modulus = torch.tensor(young_numpy, dtype=torch.float32).to(
            self.accelerator.device
        )
        poisson_numpy = np.random.uniform(0.1, 0.4)
        poisson_ratio = torch.tensor(poisson_numpy, dtype=torch.float32).to(
            self.accelerator.device
        )
        E_nu_list = [young_modulus, poisson_ratio]
        self.E_nu_list = E_nu_list

        self.model = accelerator.prepare(self.model)
        self.setup_simulation(dataset_dir, grid_size=args.grid_size)

        # 检查点设置
        if args.checkpoint_path == "None":
            args.checkpoint_path = None
        if args.checkpoint_path is not None:
            if args.video_dir_name in model_dict:
                args.checkpoint_path = model_dict[args.video_dir_name]
            self.load(args.checkpoint_path)
            trainable_params = list(self.sim_fields.parameters()) + self.E_nu_list
            optim_list = [
                {"params": self.E_nu_list, "lr": args.lr * 1e-10},
                {
                    "params": self.sim_fields.parameters(),
                    "lr": args.lr,
                    "weight_decay": 1e-4,
                },
            ]

            if args.update_velo:
                self.freeze_velo = False
                velo_optim = [
                    {
                        "params": self.velo_fields.parameters(),
                        "lr": args.lr * 1e-4,
                        "weight_decay": 1e-4,
                    },
                ]
                self.velo_optimizer = torch.optim.AdamW(
                    velo_optim,
                    lr=args.lr,
                    weight_decay=0.0,
                )
                self.velo_scheduler = get_linear_schedule_with_warmup(
                    optimizer=self.velo_optimizer,
                    num_warmup_steps=args.warmup_step,
                    num_training_steps=args.train_iters,
                )
            else:
                self.freeze_velo = True
                self.velo_optimizer = None
        else:
            self.freeze_velo = False

            velo_optim = [
                {
                    "params": self.velo_fields.parameters(),
                    "lr": args.lr,
                    "weight_decay": 1e-4,
                },
            ]
            self.velo_optimizer = torch.optim.AdamW(
                velo_optim,
                lr=args.lr,
                weight_decay=0.0,
            )
            self.velo_scheduler = get_linear_schedule_with_warmup(
                optimizer=self.velo_optimizer,
                num_warmup_steps=10, # 10
                num_training_steps=100,
            )
            self.velo_optimizer, self.velo_scheduler = accelerator.prepare(
                self.velo_optimizer, self.velo_scheduler
            )

        self.velo_fields = accelerator.prepare(self.velo_fields)

        # 设置训练信息
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

    # HX: simulation过程
    def setup_simulation(self, dataset_dir, grid_size=100):

        wp.init()

        device = "cuda:{}".format(self.accelerator.process_index)

        # 获取并处理模拟点云坐标和协方差
        xyzs = self.render_params.gaussians.get_xyz.detach().clone()
        sim_xyzs = xyzs[self.sim_mask_in_raw_gaussian, :]
        sim_cov = (
            self.render_params.gaussians.get_covariance()[
                self.sim_mask_in_raw_gaussian, :
            ]
            .detach()
            .clone()
        )

        # scale, and shift
        # already done in setup_eval()
        # convert back to PhysDreamer format
        pos_max = sim_xyzs.max()
        pos_min = sim_xyzs.min()
        scale = (pos_max - pos_min) * 1.8
        shift = -pos_min + (pos_max - pos_min) * 0.25
        self.scale, self.shift = scale, shift
        print("scale, shift", scale, shift)
        

        # 加载内部填充点云
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


        # sim_xyzs, _, _, _ = self.GSspace2MPMspace(sim_xyzs)
        sim_xyzs = (sim_xyzs + shift) / scale
        sim_aabb = torch.stack(
            [torch.min(sim_xyzs, dim=0)[0], torch.max(sim_xyzs, dim=0)[0]], dim=0
        )
        sim_aabb = (
            sim_aabb - torch.mean(sim_aabb, dim=0, keepdim=True)
        ) * 1.2 + torch.mean(sim_aabb, dim=0, keepdim=True)
        print("simulation aabb: ", sim_aabb)

        # point cloud resample with kmeans
        downsample_scale = self.args.downsample_scale # downsample_scale = 0.01
        num_cluster = int(sim_xyzs.shape[0] * downsample_scale)
        ds_grid_size_ = int(num_cluster ** (1/3))
        print("grid_size_1d: ", ds_grid_size_)
        ds_grid_size = (torch.max(sim_xyzs, dim=0)[0] - torch.min(sim_xyzs, dim=0)[0]) / ds_grid_size_
        print("grid_size: ", ds_grid_size)
        
        '''
        cluster[p2v_map[i][j]] = is the id of grid
        p2v_map: [
            a, b, 0, 0, ..., 0,
            c, 0, 0, 0, ..., 0,
            ...
        ]
        abc is the id of each pos
        '''
        unique, cluster, p2v_map = grid_sample(sim_xyzs, ds_grid_size, start=sim_aabb[0])
        
        # print(unique)
        # z_pad = unique // (grid_size_ ** 2)
        # y_pad = (unique - z_pad * (grid_size_ ** 2)) // grid_size_
        # x_pad = unique % grid_size_
        # unique_pos = torch.zeros((unique.shape[0], 3), device=sim_xyzs.device) + torch.min(sim_xyzs, dim=0)[0]
        # tmp = torch.stack((
        #     x_pad * grid_size[0] + 0.5 * grid_size[0], 
        #     y_pad * grid_size[1] + 0.5 * grid_size[1], 
        #     z_pad * grid_size[2] + 0.5 * grid_size[2]
        # ))
        # tmp = torch.transpose(tmp, 0, 1)
        # unique_pos += tmp
        # print(sim_xyzs)
        # sim_xyzs = unique_pos
        # print(sim_xyzs)
        # print(p2v_map)
        
        mask = p2v_map != 0
        num_non_zero_indices = mask.sum(dim=1).unsqueeze(-1)
        clusters_tensor_masked = p2v_map[mask]
        gathered_positions = torch.index_select(sim_xyzs, 0, clusters_tensor_masked.view(-1))
        pos_tmp = torch.zeros((p2v_map.shape[0], p2v_map.shape[1], sim_xyzs.shape[1]), device=gathered_positions.device)
        pos_tmp[mask] = gathered_positions
        sim_xyzs = torch.sum(pos_tmp, dim=1) / num_non_zero_indices
        print("sim_xyzs shape init: ", sim_xyzs.shape)
        
        self.unique = unique
        self.p2v_map = p2v_map
        

        # 获取并处理高斯点云坐标
        # sim_gaussian_pos, _, _, _ = self.GSspace2MPMspace(sim_gaussian_pos)
        sim_gaussian_pos = self.render_params.gaussians.get_xyz.detach().clone()[
            self.sim_mask_in_raw_gaussian, :
        ]
        sim_gaussian_pos = (sim_gaussian_pos + shift) / scale

        '''
        mx3 nx3 -> mxn
        mxn -> mxk
        '''
        # cdist = torch.cdist(sim_gaussian_pos, sim_xyzs) * -1.0
        # _, top_k_index = torch.topk(cdist, self.args.top_k, dim=-1)
        # self.top_k_index = top_k_index

        print("Downsampled to: ", sim_gaussian_pos.shape[0], "to", sim_xyzs.shape[0])
        
        # particle_position_tensor_to_ply(sim_xyzs, "./outputs/temp/sim_xyzs_griddownsampling.ply")
        # sys.exit()

        points_volume = get_volume(sim_xyzs.detach().cpu().numpy())
        num_particles = sim_xyzs.shape[0]
        sim_aabb = torch.stack(
            [torch.min(sim_xyzs, dim=0)[0], torch.max(sim_xyzs, dim=0)[0]], dim=0
        )
        sim_aabb = (
            sim_aabb - torch.mean(sim_aabb, dim=0, keepdim=True)
        ) * 1.2 + torch.mean(sim_aabb, dim=0, keepdim=True)
        print("simulation aabb: ", sim_aabb)

        # 初始化 Warp 模块
        wp.init()
        wp.config.mode = "debug"
        wp.config.verify_cuda = True

        # 初始化 MPM 结构和模型
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
            "material": "sand",  # Hao "jelly", "metal", "sand", "foam", "snow", "plasticine", "neo-hookean"
            "g": [0.0, 0.0, 0.0],
            "density": 200,  # kg / m^3
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
        moving_pts_path = os.path.join(dataset_dir, "moving_part_points.ply")
        if os.path.exists(moving_pts_path):
            # moving_pts = (moving_pts + shift) / scale
            moving_pts = pcu.load_mesh_v(moving_pts_path)
            moving_pts = torch.from_numpy(moving_pts).float().to(device)
            moving_pts, _, _, _ = self.GSspace2MPMspace(moving_pts)
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
            '''
            dummy one: freeze every area outside parts
            '''
            # TODO: get points in MPM space outsides parts area

            # HX: 平移值来算velocity

            motion_file_path = self.args.motion_file_path
            root_motion_file_path = self.args.root_motion_file_path
            corr_file_path = self.args.matching_file_path
            self.delta_x_orign = get_delta_x_orign(motion_file_path, corr_file_path)
            v_dt = self.args.v_dt
            simulation_dt = self.args.simulation_dt
            frame_per_motion = int(simulation_dt / self.args.frame_dt)
            self.frame_per_motion = frame_per_motion
            
            velocity_list = get_velocity_list(motion_file_path, self.scale, self.shift, dt=v_dt)
            # HX: Global transformation
            root_motion = get_root_motinon(root_motion_file_path, frame_per_motion)

            # HX: 传给物理模拟引擎，wrap_dv_params在decode_param.py中
            dv_params = wrap_dv_params(velocity_list, self.temp_params)
            # print("---------velocity list----------")
            # print(len(velocity_list), velocity_list[0].shape)
            
            # set boundary condition on particle
            # should be done in later training
            # set_driven_velocity(mpm_solver, dv_params, dt=simulation_dt)
            self.dv_params = dv_params
            self.root_motion = root_motion
            self.simulation_dt = simulation_dt
            # print("---------dv_params--------")
            # print(self.dv_params[0])
            
            # set boundary condition on grid to fix freeze parts
            freeze_mask_list, freeze_mask = get_unrelated_parts(sim_xyzs, torch.as_tensor(self.temp_params["point"]), torch.as_tensor(self.temp_params["size"]))
            freeze_pts = sim_xyzs[freeze_mask, :]
            # print("sim pts and freeze mask and freeze pts shape: ", sim_xyzs.shape, freeze_mask.shape, freeze_pts.shape)
            # print("freeze pts num: ", freeze_mask.sum())
            # print(sim_xyzs)
            # print(freeze_pts)
            # particle_position_tensor_to_ply(freeze_pts, "./outputs/temp/freeze_pts_griddownsampling.ply")
            # print(torch.min(freeze_pts, dim=0)[0], torch.max(freeze_pts, dim=0)[0])
            
            grid_freeze_mask = apply_grid_bc_w_freeze_pts(
                grid_size, 1.0, freeze_pts, mpm_solver
            )
            self.freeze_mask_list = freeze_mask_list
            self.freeze_mask = freeze_mask
        

        num_freeze_pts = self.freeze_mask.sum()
        print(
            "num freeze pts in total",
            num_freeze_pts.item(),
            "num moving pts",
            num_particles - num_freeze_pts.item(),
        )
        # sys.exit()


        # 初始化模拟场参数
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
            # stem_pts, _, _, _ = self.GSspace2MPMspace(stem_pts)
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

        # self.sim_fields = create_spatial_fields(self.args, 1, sim_aabb)
        # self.sim_fields.train()
        # self.velo_fields = create_velocity_model(self.args, sim_aabb)
        self.args.sim_res = 24
        self.velo_fields = create_spatial_fields(
            self.args, 3, sim_aabb, add_entropy=False
        )
        self.velo_fields.train()

    def get_simulation_input(self, device):
        """
        Outs: All padded
            density: [N]
            young_modulus: [N]
            poisson_ratio: [N]
            velocity: [N, 3]
            query_mask: [N]
        """

        # TODO: input fixed material, make it trainable
        # density, youngs_modulus, ret_poisson, entropy = self.get_material_params(device)
        density, youngs_modulus, ret_poisson, entropy = self.get_material_params_fixed(device)
        initial_position_time0 = self.particle_init_position.clone()

        query_mask = torch.logical_not(self.freeze_mask)
        query_pts = initial_position_time0[query_mask, :]

        # velocity = self.velo_fields(torch.cat([query_pts, time_array.unsqueeze(-1)], dim=-1))[..., :3]
        velocity = self.velo_fields(query_pts)[..., :3]

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

    def get_material_params_fixed(self, device):

        # initial_position_time0 = self.particle_init_position.detach()

        # query_mask = torch.logical_not(self.freeze_mask)
        # query_mask = torch.ones_like(self.freeze_mask).bool()
        # query_pts = initial_position_time0[query_mask, :]
        # if self.args.entropy_cls > 0:
        #     sim_params, entropy = self.sim_fields(query_pts)
        # else:
        #     sim_params = self.sim_fields(query_pts)
        entropy = torch.zeros(1).to(device)

        # sim_params = sim_params * 1000
        # sim_params = torch.exp(self.sim_fields(query_pts))

        # density = sim_params[..., 0]

        youngs_modulus = self.young_modulus.detach().clone()
        # youngs_modulus[query_mask] += sim_params[..., 0]

        # young_modulus = torch.exp(sim_params[..., 0]) + init_young
        # youngs_modulus = torch.clamp(youngs_modulus, 1000.0, 5e8)

        density = self.density.detach().clone()
        # density[self.freeze_mask] = 100000
        ret_poisson = self.poisson_ratio.detach().clone()

        return density, youngs_modulus, ret_poisson, entropy

    def train_one_step(self):

        # self.sim_fields.train()
        self.velo_fields.train()
        self.model.eval()
        accelerator = self.accelerator
        device = "cuda:{}".format(accelerator.process_index)
        # data = next(self.dataloader)
        # cam = data["cam"][0]
        cam = self.render_params.camera

        # gt_videos = data["video_clip"][0, 1 : self.num_frames, ...]

        # window_size = int(self.window_size_schduler.compute_state(self.step)[0])
        # stop_velo_opt_thres = 15
        do_velo_opt = not self.freeze_velo
        if not do_velo_opt:
            stop_velo_opt_thres = (
                0  # stop velocity optimization if we are loading from checkpoint
            )
            self.velo_fields.eval()

        rendered_video_list = []
        log_loss_dict = {
            "loss": [],
            "l2_loss": [],
            "psnr": [],
            "ssim": [],
            "entropy": [],
        }
        log_psnr_dict = {}

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

        init_velo_mean = particle_velo[query_mask, :].mean().item()
        init_velo_max = particle_velo[query_mask, :].max().item()

        if not do_velo_opt:
            particle_velo = particle_velo.detach()
        # print("does do velo opt": do_velo_opt)

        num_particles = particle_pos.shape[0]

        # delta_time = 1.0 / 30  # 30 fps
        # substep_size = delta_time / self.args.substep
        # num_substeps = int(delta_time / substep_size)
        substep_dt = self.args.substep_dt
        frame_dt = self.args.frame_dt
        frame_num = self.args.frame_num
        step_per_frame = int(self.args.frame_dt / self.args.substep_dt)
        velo_dt = 0.5 # 50%
        
        # checkpoint_steps = self.args.checkpoint_steps

        # start_time_idx = max(0, window_size - self.args.compute_window)

        # temporal_stride = self.args.stride

        # if temporal_stride < 0 or temporal_stride > window_size:
        #     temporal_stride = window_size

        

        for simulation_frame_idx in range(frame_num):
            # TODO: set driven velocity bc
            centroid_delta_pos_origin = None
            velocity_list = None
            # TODO: for loss calculation
            trans_scalor = 0.1
            # get initialized v in the simulation frame
            # consistent_velo = torch.zeros_like(particle_velo)
            
            # # velocity translation part
            # self.update(particle_pos, particle_velo, parti)
            particle_pos_prev = particle_pos.detach()
            particle_pos, particle_velo, particle_F, particle_C, particle_cov = (
                MPMDifferentiableSimulationVelo.apply(
                    self.mpm_solver,
                    self.mpm_state,
                    self.mpm_model,
                    substep_dt,
                    step_per_frame,
                    particle_pos,
                    particle_velo,
                    particle_F,
                    particle_C,
                    youngs_modulus,
                    self.E_nu_list[1],
                    density,
                    query_mask,
                    device,
                    True,
                    0,
                )
            )
            
            # calculate loss in MPM space
            pos_prev_min = torch.min(particle_pos_prev, dim=0)[0]
            pos_prev_max = torch.max(particle_pos_prev, dim=0)[0]
            pos_prev_centroid = (pos_prev_max + pos_prev_min) / 2
            pos_min = torch.min(particle_pos, dim=0)[0]
            pos_max = torch.max(particle_pos, dim=0)[0]
            pos_centroid = (pos_max + pos_min) / 2
            # l2_dist = torch.norm(pos_centroid - pos_prev_centroid, p=2)
            centroid_delta_pos = pos_centroid - pos_prev_centroid
            l2_dist = torch.norm(centroid_delta_pos - centroid_delta_pos_origin, p=2)
            
            loss = l2_dist
            
            loss.backward()
            
            particle_pos, particle_velo, particle_F, particle_C = (
                particle_pos.detach(),
                particle_velo.detach(),
                particle_F.detach(),
                particle_C.detach(),
            )
            
            # gaussian_pos.requires_grad = True
            
            
        
        for start_time_idx in range(0, window_size, temporal_stride):

            end_time_idx = min(start_time_idx + temporal_stride, window_size)

            num_step_with_grad = num_substeps * (end_time_idx - start_time_idx)

            # gt_frame = gt_videos[[end_time_idx - 1]]

            if start_time_idx != 0:
                # TODO: fix material params, make it trainable
                density, youngs_modulus, poisson, entropy = self.get_material_params_fixed(
                    device
                )

            if checkpoint_steps > 0 and checkpoint_steps < num_step_with_grad:
                for time_step in range(0, num_step_with_grad, checkpoint_steps):
                    num_step = min(num_step_with_grad - time_step, checkpoint_steps)
                    if num_step == 0:
                        break
                    particle_pos, particle_velo, particle_F, particle_C = (
                        MPMDifferentiableSimulationWCheckpoint.apply(
                            self.mpm_solver,
                            self.mpm_state,
                            self.mpm_model,
                            substep_size,
                            num_step,
                            particle_pos,
                            particle_velo,
                            particle_F,
                            particle_C,
                            youngs_modulus,
                            self.E_nu_list[1],
                            density,
                            query_mask,
                            device,
                            True,
                            0,
                        )
                    )
            else:
                particle_pos, particle_velo, particle_F, particle_C, particle_cov = (
                    MPMDifferentiableSimulationClean.apply(
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
                        density,
                        query_mask,
                        device,
                        True,
                        0,
                    )
                )

            # substep-3: render gaussian

            gaussian_pos = particle_pos * self.scale - self.shift
            undeformed_gaussian_pos = (
                self.particle_init_position * self.scale - self.shift
            )
            disp_offset = gaussian_pos - undeformed_gaussian_pos.detach()
            # gaussian_pos.requires_grad = True

            simulated_video = render_gaussian_seq_w_mask_with_disp(
                cam,
                self.render_params,
                undeformed_gaussian_pos.detach(),
                self.top_k_index,
                [disp_offset],
                self.sim_mask_in_raw_gaussian,
            )

            # print("debug", simulated_video.shape, gt_frame.shape, gaussian_pos.shape, init_xyzs.shape, density.shape, query_mask.sum().item())
            rendered_video_list.append(simulated_video.detach())

            l2_loss = 0.5 * F.mse_loss(simulated_video, gt_frame, reduction="mean")
            ssim_loss = compute_ssim(simulated_video, gt_frame)
            loss = l2_loss * (1.0 - self.ssim) + (1.0 - ssim_loss) * self.ssim

            loss = loss * (self.args.loss_decay**end_time_idx)
            sm_velo_loss = self.velo_fields.compute_smoothess_loss() * 10.0
            if not (do_velo_opt and start_time_idx == 0):
                sm_velo_loss = sm_velo_loss.detach()

            sm_spatial_loss = self.sim_fields.compute_smoothess_loss()

            sm_loss = (
                sm_velo_loss + sm_spatial_loss
            )  # typically 20 times larger than rendering loss

            loss = loss + sm_loss * self.tv_loss_weight
            loss = loss + entropy * self.args.entropy_reg
            loss = loss / self.args.compute_window
            loss.backward()

            # from IPython import embed; embed()
            # print(self.E_nu_list[1].grad)

            particle_pos, particle_velo, particle_F, particle_C = (
                particle_pos.detach(),
                particle_velo.detach(),
                particle_F.detach(),
                particle_C.detach(),
            )

            with torch.no_grad():
                psnr = compute_psnr(simulated_video, gt_frame).mean()
                log_loss_dict["loss"].append(loss.item())
                log_loss_dict["l2_loss"].append(l2_loss.item())
                log_loss_dict["psnr"].append(psnr.item())
                log_loss_dict["ssim"].append(ssim_loss.item())
                log_loss_dict["entropy"].append(entropy.item())

                print(
                    psnr.item(),
                    end_time_idx,
                    youngs_modulus.max().item(),
                    density.max().item(),
                )
                log_psnr_dict["psnr_frame_{}".format(end_time_idx)] = psnr.item()
                # print(psnr.item(), end_time_idx, youngs_modulus.max().item(), density.max().item())

        nu_grad_norm = self.E_nu_list[1].grad.norm(2).item()
        spatial_grad_norm = 0
        for p in self.sim_fields.parameters():
            if p.grad is not None:
                spatial_grad_norm += p.grad.norm(2).item()
        velo_grad_norm = 0
        for p in self.velo_fields.parameters():
            if p.grad is not None:
                velo_grad_norm += p.grad.norm(2).item()

        renderd_video = torch.cat(rendered_video_list, dim=0)
        renderd_video = torch.clamp(renderd_video, 0.0, 1.0)
        visual_video = (renderd_video.detach().cpu().numpy() * 255.0).astype(np.uint8)
        gt_video = (gt_videos.detach().cpu().numpy() * 255.0).astype(np.uint8)

        if (
            self.step % self.gradient_accumulation_steps == 0
            or self.step == (self.train_iters - 1)
            or (self.step % self.log_iters == self.log_iters - 1)
        ):

            torch.nn.utils.clip_grad_norm_(
                self.trainable_params,
                self.max_grad_norm,
                error_if_nonfinite=False,
            )  # error if nonfinite is false

            self.optimizer.step()
            self.optimizer.zero_grad()
            if do_velo_opt:
                assert self.velo_optimizer is not None
                torch.nn.utils.clip_grad_norm_(
                    self.velo_fields.parameters(),
                    self.max_grad_norm,
                    error_if_nonfinite=False,
                )  # error if nonfinite is false
                self.velo_optimizer.step()
                self.velo_optimizer.zero_grad()
                self.velo_scheduler.step()
            with torch.no_grad():
                self.E_nu_list[0].data.clamp_(1e-1, 1e8)
                self.E_nu_list[1].data.clamp_(1e-2, 0.449)
        self.scheduler.step()

        for k, v in log_loss_dict.items():
            log_loss_dict[k] = np.mean(v)

        print(log_loss_dict)
        print(
            "nu: ",
            self.E_nu_list[1].item(),
            nu_grad_norm,
            spatial_grad_norm,
            velo_grad_norm,
            "young_mean, max:",
            youngs_modulus.mean().item(),
            youngs_modulus.max().item(),
            do_velo_opt,
            "init_velo_mean:",
            init_velo_mean,
        )

        if accelerator.is_main_process and (self.step % self.wandb_iters == 0):
            with torch.no_grad():
                wandb_dict = {
                    "nu_grad_norm": nu_grad_norm,
                    "spatial_grad_norm": spatial_grad_norm,
                    "velo_grad_norm": velo_grad_norm,
                    "nu": self.E_nu_list[1].item(),
                    # "mean_density": density.mean().item(),
                    "mean_E": youngs_modulus.mean().item(),
                    "max_E": youngs_modulus.max().item(),
                    "min_E": youngs_modulus.min().item(),
                    "smoothness_loss": sm_loss.item(),
                    "window_size": window_size,
                    "max_particle_velo": particle_velo.max().item(),
                    "init_velo_mean": init_velo_mean,
                    "init_velo_max": init_velo_max,
                }

                wandb_dict.update(log_psnr_dict)
                simulated_video = self.inference(cam, substep=num_substeps)
                sim_video_torch = (
                    torch.from_numpy(simulated_video).float().to(device) / 255.0
                )
                gt_video_torch = torch.from_numpy(gt_video).float().to(device) / 255.0

                full_psnr = compute_psnr(sim_video_torch[1:], gt_video_torch)

                first_psnr = full_psnr[:6].mean().item()
                last_psnr = full_psnr[-6:].mean().item()
                full_psnr = full_psnr.mean().item()
                wandb_dict["full_psnr"] = full_psnr
                wandb_dict["first_psnr"] = first_psnr
                wandb_dict["last_psnr"] = last_psnr
                wandb_dict.update(log_loss_dict)

                # add young render

                youngs_norm = youngs_modulus - youngs_modulus.min() + 1e-2
                young_color = youngs_norm / torch.quantile(youngs_norm, 0.99)
                young_color = torch.clamp(young_color, 0.0, 1.0)
                young_color[self.freeze_mask] = 0.0
                queryed_young_color = young_color[self.top_k_index]  # [n_raw, topk]
                young_color = queryed_young_color.mean(dim=-1)

                young_color_full = torch.ones_like(
                    self.render_params.gaussians._xyz[:, 0]
                )

                young_color_full[self.sim_mask_in_raw_gaussian] = young_color
                young_color = torch.stack(
                    [young_color_full, young_color_full, young_color_full], dim=-1
                )

                young_img = render_feat_gaussian(
                    cam,
                    self.render_params.gaussians,
                    self.render_params.render_pipe,
                    self.render_params.bg_color,
                    young_color,
                )["render"]
                young_img = (
                    (young_img.detach().cpu().numpy() * 255.0)
                    .astype(np.uint8)
                    .transpose(1, 2, 0)
                )
                wandb_dict["young_img"] = wandb.Image(young_img)

                if self.step % int(10 * self.wandb_iters) == 0:

                    wandb_dict["rendered_video"] = wandb.Video(
                        visual_video, fps=visual_video.shape[0]
                    )

                    wandb_dict["gt_video"] = wandb.Video(
                        gt_video,
                        fps=gt_video.shape[0],
                    )

                    wandb_dict["inference_video"] = wandb.Video(
                        simulated_video,
                        fps=simulated_video.shape[0],
                    )

                    simulated_video = self.inference(
                        cam, velo_scaling=5.0, num_sec=3, substep=num_substeps
                    )
                    wandb_dict["inference_video_v5_t3"] = wandb.Video(
                        simulated_video,
                        fps=30,
                    )

                if self.use_wandb:
                    wandb.log(wandb_dict, step=self.step)

        self.accelerator.wait_for_everyone()

    # HX: Field训练过程
    def train_one_motion(self, frame, idx):
        self.velo_fields.train() # velo field
        self.model.eval()
        accelerator = self.accelerator
        device = "cuda:{}".format(accelerator.process_index)
        
        particle_pos = self.particle_init_position_training.clone()
        # clean grid, stress, F, C and rest initial position
        self.mpm_state.reset_state(
            particle_pos.clone(),
            None,
            None,  # .clone(),
            device=device,
            requires_grad=True,
        )

        ''' 
        HX: 训练velocity/material field所需要的参数 在get_velocity_before_simulation中进行初始化
            但实际并没有训练material field
            因此这里的物理参数youngs_modulus, poisson, paritcle_F, paritcle_C是hard coded
        '''
        self.mpm_state.set_require_grad(True)

        (   
            density,
            youngs_modulus,
            poisson,
            particle_velo, # HX: velocity field所需要训练的核心参数
            query_mask,
            particle_F,
            particle_C,
            entropy,
        ) = self.get_velocity_before_simulation(frame, device)

        init_velo_mean = torch.abs(particle_velo[query_mask, :]).mean().item()
        init_velo_max = torch.abs(particle_velo[query_mask, :]).max().item()
        
        num_particles = particle_pos.shape[0]

        substep_dt = self.args.substep_dt
        frame_dt = self.args.frame_dt
        frame_num = self.args.frame_num
        step_per_frame = int(self.args.frame_dt / self.args.substep_dt)
        
        centroid_delta_pos_origin = np.array(self.delta_x_orign)[:, frame, :] # Hao
        centroid_delta_pos_origin = torch.tensor(centroid_delta_pos_origin, device=device)
        
        particle_pos_prev = particle_pos.detach()
        
        particle_pos, particle_velo, particle_F, particle_C, particle_cov = (
            MPMDifferentiableSimulationVelo.apply(
                self.mpm_solver,
                self.mpm_state,
                self.mpm_model,
                substep_dt,
                step_per_frame * self.frame_per_motion, 
                particle_pos,
                particle_velo,
                particle_F,
                particle_C,
                youngs_modulus,
                self.E_nu_list[1],
                density,
                query_mask,
                device,
                True,
                0,
            )
        )
        
        # HX: 我们需要重新设计loss
        # calculate loss in MPM space
        loss = 0.
        for i in range(len(self.freeze_mask_list)):
            '''
            delta_x = source_delta_x / source_aabb_len = simulated_delta_x / sim_aabb_len
            '''
            particle_pos_prev_part = particle_pos_prev[self.freeze_mask_list[i]]
            pos_prev_min = torch.min(particle_pos_prev_part, dim=0)[0]
            pos_prev_max = torch.max(particle_pos_prev_part, dim=0)[0]
            pos_prev_aabb = (pos_prev_max - pos_prev_min).max()
            pos_prev_centroid = (pos_prev_max + pos_prev_min) / 2
            
            particle_pos_part = particle_pos[self.freeze_mask_list[i]]
            pos_min = torch.min(particle_pos_part, dim=0)[0]
            pos_max = torch.max(particle_pos_part, dim=0)[0]
            pos_centroid = (pos_max + pos_min) / 2
            
            centroid_delta_pos = pos_centroid - pos_prev_centroid
            
            # HX: L2 Loss进行监督
            l2_dist = torch.norm(centroid_delta_pos - centroid_delta_pos_origin * pos_prev_aabb, p=2)
            
            loss += l2_dist

        # HX: Regu loss for smooth
        sm_velo_loss = self.velo_fields.compute_smoothess_loss() 
        loss += sm_velo_loss
        loss.backward()
        
        # print(particle_velo[query_mask, :])
        
        print("loss... : ", loss)
        # print("init velo mean: ", init_velo_mean)
        # print("init velo max: ", init_velo_max)    
        
        particle_pos, particle_velo, particle_F, particle_C = (
            particle_pos.detach(),
            particle_velo.detach(),
            particle_F.detach(),
            particle_C.detach(),
        )
            
        if idx % 10 == 0:
            print(f"rendering at iteration{idx} at {self.output_path}")
            with torch.no_grad():
                current_camera = self.render_params.camera
                gaussian_pos = particle_pos * self.scale - self.shift
                undeformed_gaussian_pos = self.particle_init_position_training * self.scale - self.shift
                disp_offset = gaussian_pos - undeformed_gaussian_pos.detach()
                
                rendering = render_gaussian_w_grid_sampling(current_camera, self.render_params, disp_offset, self.p2v_map, frame * self.frame_per_motion, self.frame_per_motion, self.root_motion)
                
                import cv2
                cv2_img = rendering.permute(1, 2, 0).detach().cpu().numpy()
                cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
                height = cv2_img.shape[0] // 2 * 2
                width = cv2_img.shape[1] // 2 * 2
                render_out_dir = os.path.join(self.output_path, f"iteration_{idx}")
                os.makedirs(render_out_dir, exist_ok=True)
                cv2.imwrite(
                    os.path.join(self.output_path, f"iteration_{idx}", f"{frame}.png".rjust(8, "0")),
                    255 * cv2_img,
                )
            # TODO: store inputs velo
        
        
        # fps = int(1.0 / self.args.frame_dt)
        # os.system(
        #     f"ffmpeg -framerate {fps} -i {self.output_path}/%04d.png -c:v libx264 -s {width}x{height} -y -pix_fmt yuv420p {self.output_path}/output.mp4"
        # )
        
        self.particle_init_position_training = particle_pos
        self.save_velo = particle_velo

    # def train(self):
    #     # might remove tqdm when multiple node
    #     for index in tqdm(range(self.step, self.train_iters), desc="Training progress"):
    #         self.train_one_step()
    #         if self.step % self.log_iters == self.log_iters - 1:
    #             if self.accelerator.is_main_process:
    #                 self.save()
    #                 # self.test()
    #         # self.accelerator.wait_for_everyone()
    #         self.step += 1
    #     if self.accelerator.is_main_process:
    #         self.save()
            
    def train(self):
        # online training, optimizing every motion seperately
        self.particle_init_position_training = self.particle_init_position
        # print(self.num_frames, self.frame_per_motion, self.num_frames // self.frame_per_motion)
        # sys.exit()
        for index in range(self.args.frame_num // self.frame_per_motion):
            print("--------Training {}th frame--------".format(index))
            
            
            for p in self.velo_fields.parameters():
                # pdb.set_trace()
                if p.grad is not None:
                    if p.dim() == 1: continue # Hao
                    torch.nn.init.kaiming_normal_(p)
            
            for i in tqdm(range(100), desc="Training progress"):
                self.train_one_motion(index, i)
                
                velo_grad_norm = 0
                for p in self.velo_fields.parameters():
                    if p.grad is not None:
                        velo_grad_norm += p.grad.norm(2).item()
                print("velo grad norm : ", velo_grad_norm)
                
                # if i % 50 == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.velo_fields.parameters(),
                    2.0,
                    error_if_nonfinite=False,
                )  # error if nonfinite is false
                self.velo_optimizer.step()
                self.velo_optimizer.zero_grad()
                self.velo_scheduler.step()
            
            # Save after every period
            os.makedirs(os.path.join(self.output_path, "save_tensor"), exist_ok=True)
            data = {
                "position": self.particle_init_position_training,
                "velocity": self.save_velo,
                "frame": index
            }
            torch.save(data, os.path.join(self.output_path, "save_tensor", f"frame_{index}.pt"))
            
        print("---------------------------")
        print(f"Loop for this frame is done!!!")
            
    def save(
        self,
    ):
        # training states
        output_path = os.path.join(
            self.output_path, f"checkpoint_model_{self.step:06d}"
        )
        os.makedirs(output_path, exist_ok=True)

        name_list = [
            "velo_fields",
            "sim_fields",
        ]
        for i, model in enumerate(
            [
                self.accelerator.unwrap_model(self.velo_fields, keep_fp32_wrapper=True),
                self.accelerator.unwrap_model(self.sim_fields, keep_fp32_wrapper=True),
            ]
        ):
            model_name = name_list[i]
            model_path = os.path.join(output_path, model_name + ".pt")
            torch.save(model.state_dict(), model_path)

    def load(self, checkpoint_dir):
        name_list = [
            "velo_fields",
            "sim_fields",
        ]
        for i, model in enumerate([self.velo_fields, self.sim_fields]):
            model_name = name_list[i]
            if model_name == "sim_fields" and (not self.args.load_sim):
                continue
            model_path = os.path.join(checkpoint_dir, model_name + ".pt")
            print("=> loading: ", model_path)
            model.load_state_dict(torch.load(model_path))

    def setup_eval(self, args, gaussian_path, dataset_dir, white_background=True):
        # setup gaussians
        class RenderPipe(NamedTuple):
            convert_SHs_python = False
            compute_cov3D_python = False
            debug = False

        class RenderParams(NamedTuple):
            render_pipe: RenderPipe
            bg_color: bool
            gaussians: GaussianModel
            # camera_list: list
            camera: GSCamera

        gaussians = GaussianModel(3)
        gaussians.load_ply(gaussian_path)
        gaussians.detach_grad()
        print(
            "load gaussians from: {}".format(gaussian_path),
            "... num gaussians: ",
            gaussians._xyz.shape[0],
        )
        bg_color = [1, 1, 1] if white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        render_pipe = RenderPipe()
        
        # convert from GS to MPM space
        # transformed_pos, scale_origin, original_mean_pos, rotation_matrices = self.GSspace2MPMspace(gaussians._xyz)
        # self.scale_origin = scale_origin
        # self.original_mean_pos = original_mean_pos
        # self.rotation_matrices = rotation_matrices
        preprocessing_params = {
            "rotation_degree": [],
            "rotation_axis": []
        }
        rotation_matrices = generate_rotation_matrices(
            torch.tensor(preprocessing_params["rotation_degree"]),
            preprocessing_params["rotation_axis"],
        )
        
        
        pos = gaussians.get_xyz.detach().clone()
        pos_max_ = torch.max(pos, dim=0)[0]
        pos_min_ = torch.min(pos, dim=0)[0]
        scale_origin = 1 / (pos_max_ - pos_min_).max()
        original_mean_pos = (pos_min_ + pos_max_) / 2.0
        
        
        pos_max = pos.max()
        pos_min = pos.min()
        scale = (pos_max - pos_min) * 1.8
        shift = -pos_min + (pos_max - pos_min) * 0.25
        self.scale, self.shift = scale, shift
        print("scale, shift", scale, shift)
        
        # camera_list = self.dataset.test_camera_list
        # TODO: load from config
        camera_params = {
            "mpm_space_vertical_upward_axis": [0,0,1],
            "mpm_space_viewpoint_center": [0.95,1.07,1],
            "default_camera_index": -1,
            "show_hint": False,
            "init_azimuthm": -36.7,
            "init_elevation": -68.96,
            "init_radius": 8.11,
            "move_camera": False,
            "delta_a": 0.4,
            "delta_e": 0.0,
            "delta_r": 0.0
        }
        
        mpm_space_viewpoint_center = (
            torch.tensor(camera_params["mpm_space_viewpoint_center"]).reshape((1, 3)).cuda()
        )
        mpm_space_vertical_upward_axis = (
            torch.tensor(camera_params["mpm_space_vertical_upward_axis"])
            .reshape((1, 3))
            .cuda()
        )
        (
            viewpoint_center_worldspace,
            observant_coordinates,
        ) = get_center_view_worldspace_and_observant_coordinate(
            mpm_space_viewpoint_center,
            mpm_space_vertical_upward_axis,
            rotation_matrices,
            scale_origin,
            original_mean_pos,
        )
        
        # TODO: fixed view now, support dynamic camera
        current_camera = get_camera_view_motion(
            dataset_dir,
            default_camera_index=camera_params["default_camera_index"],
            center_view_world_space=viewpoint_center_worldspace,
            observant_coordinates=observant_coordinates,
            show_hint=camera_params["show_hint"],
            init_azimuthm=camera_params["init_azimuthm"],
            init_elevation=camera_params["init_elevation"],
            init_radius=camera_params["init_radius"],
            move_camera=camera_params["move_camera"],
            current_frame=0,
            delta_a=camera_params["delta_a"],
            delta_e=camera_params["delta_e"],
            delta_r=camera_params["delta_r"],
            field2cam=None
        )
        
        

        render_params = RenderParams(
            render_pipe=render_pipe,
            bg_color=background,
            gaussians=gaussians,
            camera=current_camera,
        )
        self.render_params = render_params

        # get_gaussian scene box
        scaler = 1.1
        points = gaussians._xyz

        min_xyz = torch.min(points, dim=0)[0]
        max_xyz = torch.max(points, dim=0)[0]

        center = (min_xyz + max_xyz) / 2

        scaled_min_xyz = (min_xyz - center) * scaler + center
        scaled_max_xyz = (max_xyz - center) * scaler + center

        aabb = torch.stack([scaled_min_xyz, scaled_max_xyz], dim=0)
        
        # get_mpm space box
        # rotated_aabb = apply_rotations(aabb, rotation_matrices)
        # transformed_aabb, scale_origin, original_mean_pos = transform2origin(rotated_aabb)
        # mpm_aabb = shift2center111(transformed_aabb)

        # add filled in points
        # TODO: set up different simulation area based on parts
        gaussian_dir = os.path.dirname(gaussian_path)

        clean_points_path = os.path.join(gaussian_dir, "clean_object_points.ply")
        if os.path.exists(clean_points_path):
            clean_xyzs = pcu.load_mesh_v(clean_points_path)
            clean_xyzs = torch.from_numpy(clean_xyzs).float().to("cuda")
            self.clean_xyzs = clean_xyzs
            print(
                "loaded {} clean points from: ".format(clean_xyzs.shape[0]),
                clean_points_path,
            )
            # we can use tight threshold here
            not_sim_maks = find_far_points(
                gaussians._xyz, clean_xyzs, thres=0.01
            ).bool()
            sim_mask_in_raw_gaussian = torch.logical_not(not_sim_maks)
            # [N]
            self.sim_mask_in_raw_gaussian = sim_mask_in_raw_gaussian
        else:
            self.clean_xyzs = None
            self.sim_mask_in_raw_gaussian = torch.ones_like(gaussians._xyz[:, 0]).bool()

        return aabb

    def get_velocity_before_simulation(self, frame, device):
        """
        Outs: All padded
            density: [N]
            young_modulus: [N]
            poisson_ratio: [N]
            velocity: [N, 3]
            query_mask: [N]
        """

        # TODO: input fixed material, make it trainable
        # density, youngs_modulus, ret_poisson, entropy = self.get_material_params(device)
        density, youngs_modulus, ret_poisson, entropy = self.get_material_params_fixed(device)
        initial_position_time0 = self.particle_init_position_training.clone()

        query_mask = torch.logical_not(self.freeze_mask)
        query_pts = initial_position_time0[query_mask, :]

        
        # HX: 在此train velocity，tri-plane的结果
        # velocity = self.velo_fields(torch.cat([query_pts, time_array.unsqueeze(-1)], dim=-1))[..., :3]
        velocity_train = self.velo_fields(query_pts)[..., :3]
        velocity_train = velocity_train * 0.1 # Hao 100 撕裂 
        # get initial from dv_params
        ret_velocity = torch.zeros_like(initial_position_time0)
        print("Set velocity of {} frame".format(frame))

        
        for part_id, value in self.dv_params.items():
            # HX: 初始化velocity
            velocity = torch.tensor(value["velocity"][frame], dtype=torch.float, device=self.particle_init_position.device) * 5
            part_mask = self.freeze_mask_list[part_id]
            part_mask = torch.logical_not(part_mask)
            # check part
            # idx_one = 
            
            ret_velocity[part_mask, :] = velocity
        ret_velocity[query_mask, :] += velocity_train

        # scaling
        # velocity = velocity * 0.1  # not padded yet
        # ret_velocity = torch.zeros_like(initial_position_time0)
        # ret_velocity[query_mask, :] = velocity

        # init F, and C

        I_mat = torch.eye(3, dtype=torch.float32).to(device)

        # HX: F和C初始化为0
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
        
    def set_velocity_before_simulation(self, frame, device):
        """
        Outs: All padded
            density: [N]
            young_modulus: [N]
            poisson_ratio: [N]
            velocity: [N, 3]
            query_mask: [N]
        """

        # TODO: input fixed material, make it trainable
        # density, youngs_modulus, ret_poisson, entropy = self.get_material_params(device)
        density, youngs_modulus, ret_poisson, entropy = self.get_material_params_fixed(device)
        initial_position_time0 = self.particle_init_position.clone()

        query_mask = torch.logical_not(self.freeze_mask)
        query_pts = initial_position_time0[query_mask, :]

        # velocity = self.velo_fields(torch.cat([query_pts, time_array.unsqueeze(-1)], dim=-1))[..., :3]
        # velocity_train = self.velo_fields(query_pts)[..., :3]
        # get initial from dv_params
        ret_velocity = torch.zeros_like(initial_position_time0)
        print("Set velocity of {} frame".format(frame))
        for part_id, value in self.dv_params.items():
            velocity = torch.tensor(value["velocity"][frame], dtype=torch.float, device=self.particle_init_position.device) * 5
            part_mask = self.freeze_mask_list[part_id]
            part_mask = torch.logical_not(part_mask)
            # check part
            # idx_one = 
            
            ret_velocity[part_mask, :] = velocity
        # ret_velocity += velocity_train

        # scaling
        # velocity = velocity * 0.1  # not padded yet
        # ret_velocity = torch.zeros_like(initial_position_time0)
        # ret_velocity[query_mask, :] = velocity

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

    # HX: 用test函数测试代码是否可以run
    def test(self):
        
        # pos_init = self.particle_init_position.clone()
        # gaussian_pos = self.MPMspace2GSspace(pos_init, self.scale_origin, self.original_mean_pos, self.rotation_matrices)
        # # particle_position_tensor_to_ply(gaussian_pos, "outputs/temp/gaussian_pos_downsampling.ply")
        # prev_state = self.mpm_state
        # init_pos = wp.to_torch(prev_state.particle_x).detach().clone()
        # print("init_pos: ", init_pos.shape, torch.min(init_pos, dim=0)[0], torch.max(init_pos, dim=0)[0])
        # particle_position_tensor_to_ply(init_pos, "outputs/temp/before_p2g2p.ply")
        # sys.exit()
        # next_state = prev_state.partial_clone(requires_grad=False)
        # next_pos = wp.to_torch(next_state.particle_x).detach().clone()
        # particle_position_tensor_to_ply(next_pos, "outputs/temp/next_before_p2g2p.ply")
        # self.mpm_solver.p2g2p_differentiable(self.mpm_model, prev_state, next_state, self.args.substep_dt, device="cuda:0")
        # pos = wp.to_torch(next_state.particle_x).detach().clone()
        # print("pos: ", pos.shape, torch.min(pos, dim=0)[0], torch.max(pos, dim=0)[0])
        # particle_position_tensor_to_ply(pos, "outputs/temp/after_p2g2p.ply")
        # sys.exit()
        
        
        # print("successfully reach test!!!")
        # print(self.mpm_state)
        # set_driven_velocity(self.mpm_solver, self.mpm_state, self.dv_params, dt=self.simulation_dt)
        # tmp_pos = wp.to_torch(self.mpm_state.particle_x).detach().clone()
        # tmp_pos = self.MPMspace2GSspace(tmp_pos, self.scale_origin, self.original_mean_pos, self.rotation_matrices)
        # print(torch.max(tmp_pos, dim=0)[0], torch.min(tmp_pos, dim=0)[0])
        step_per_frame = int(self.args.frame_dt / self.args.substep_dt)
        gs_num = self.particle_init_position.shape[0]
        init_pos_per_motion = self.particle_init_position
        prev_state = self.mpm_state
        for frame in tqdm(range(self.args.frame_num)):
            current_camera = self.render_params.camera
            
            if frame % self.frame_per_motion == 0:
                (
                    density,
                    youngs_modulus,
                    poisson,
                    particle_velo,
                    query_mask,
                    particle_F,
                    particle_C,
                    entropy,
                ) = self.set_velocity_before_simulation(frame // self.frame_per_motion, self.particle_init_position.device)
                # print("------youngs modules-----")
                # print(youngs_modulus)
                # print("------nu-----")
                # print(self.E_nu_list[1])
                prev_state.continue_from_torch(
                    init_pos_per_motion, particle_velo, particle_F, particle_C, device=self.particle_init_position.device, requires_grad=False
                )
            
            for step in range(step_per_frame): 
                # print(step)
                next_state = prev_state.partial_clone(requires_grad=False)
                # self.mpm_solver.p2g2p(self.mpm_model, self.mpm_state, frame, self.args.substep_dt, device="cuda:0")
                self.mpm_solver.p2g2p_differentiable(self.mpm_model, prev_state, next_state, self.args.substep_dt, device="cuda:0")
                prev_state = next_state
            
            pos = wp.to_torch(next_state.particle_x).detach().clone()
            init_pos_per_motion = pos
            # cov3D = self.mpm_solver.export_particle_cov_to_torch()
            # rot = self.mpm_solver.export_particle_R_to_torch(self.mpm_model, self.mpm_state)
            # cov3D = cov3D.view(-1, 6)[:gs_num].to(self.accelerator.device)
            # rot = rot.view(-1, 3, 3)[:gs_num].to(self.accelerator.device)
        
            
            # print(torch.max(self.render_params.gaussians._xyz, dim=0)[0], torch.min(self.render_params.gaussians._xyz, dim=0)[0])
            # gaussian_pos = self.MPMspace2GSspace(pos, self.scale_origin, self.original_mean_pos, self.rotation_matrices)
            gaussian_pos = pos * self.scale - self.shift
            # undeformed_gaussian_pos = self.MPMspace2GSspace(self.particle_init_position, self.scale_origin, self.original_mean_pos, self.rotation_matrices)
            undeformed_gaussian_pos = self.particle_init_position * self.scale - self.shift
            disp_offset = gaussian_pos - undeformed_gaussian_pos.detach()
            
            rendering = render_gaussian_w_grid_sampling(current_camera, self.render_params, disp_offset, self.p2v_map, frame, self.frame_per_motion, self.root_motion)
            
            import cv2
            cv2_img = rendering.permute(1, 2, 0).detach().cpu().numpy()
            cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
            height = cv2_img.shape[0] // 2 * 2
            width = cv2_img.shape[1] // 2 * 2
            cv2.imwrite(
                os.path.join(self.output_path, f"{frame}.png".rjust(8, "0")),
                255 * cv2_img,
            )
        
        fps = int(1.0 / self.args.frame_dt)
        os.system(
            f"ffmpeg -framerate {fps} -i {self.output_path}/%04d.png -c:v libx264 -s {width}x{height} -y -pix_fmt yuv420p {self.output_path}/output.mp4"
        )
            
            
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yml", help="Path to the configuration file")

    # 数据集参数
    parser.add_argument("--dataset_dir", type=str, default=None, help="Directory of the dataset")
    parser.add_argument("--name", type=str, default=None, help="Name of the experiment")

    # 模型参数
    parser.add_argument("--model", type=str, default="se3_field", help="Model type")
    parser.add_argument("--feat_dim", type=int, default=64, help="Feature dimension")
    parser.add_argument("--num_decoder_layers", type=int, default=3, help="Number of decoder layers")
    parser.add_argument("--decoder_hidden_size", type=int, default=64, help="Decoder hidden size")
    parser.add_argument("--spatial_res", type=int, default=32, help="Spatial resolution")
    parser.add_argument("--zero_init", type=bool, default=True, help="Zero initialization flag")

    # 熵相关参数
    parser.add_argument("--entropy_cls", type=int, default=-1, help="Entropy classification")
    parser.add_argument("--entropy_reg", type=float, default=1e-2, help="Entropy regularization")

    # 帧数和网格参数
    parser.add_argument("--num_frames", type=str, default=14, help="Number of frames")
    parser.add_argument("--grid_size", type=int, default=64, help="Grid size")
    parser.add_argument("--sim_res", type=int, default=8, help="Simulation resolution")
    parser.add_argument("--sim_output_dim", type=int, default=1, help="Simulation output dimension")
    parser.add_argument("--loss_decay", type=float, default=1.0, help="Loss decay factor")
    parser.add_argument("--start_window_size", type=int, default=6, help="Start window size")
    parser.add_argument("--compute_window", type=int, default=1, help="Compute window size")
    parser.add_argument("--grad_window", type=int, default=14, help="Gradient window size")
    parser.add_argument("--checkpoint_steps", type=int, default=-1, help="Checkpoint steps (-1 for no checkpointing)")
    parser.add_argument("--stride", type=int, default=1, help="Stride size")

    parser.add_argument("--downsample_scale", type=float, default=0.04, help="Downsample scale")
    parser.add_argument("--top_k", type=int, default=8, help="Top k selection")

    # 损失参数
    parser.add_argument("--tv_loss_weight", type=float, default=1e-4, help="Total variation loss weight")
    parser.add_argument("--ssim", type=float, default=0.9, help="Structural similarity index")

    # 日志记录和检查点
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--log_iters", type=int, default=10, help="Log iteration interval")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    # 训练参数
    parser.add_argument("--train_iters", type=int, default=200, help="Number of training iterations")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps")

    # wandb 参数
    parser.add_argument("--update_velo", action="store_true", default=False, help="Update velocity flag")
    
    # 匹配参数
    parser.add_argument("--motion_file_path", type=str, default=None, help="Path to the motion file")
    parser.add_argument("--root_motion_file_path", type=str, default=None, help="Path to the root motion file")
    parser.add_argument("--matching_file_path", type=str, default=None, help="Path to the matching file")
    parser.add_argument("--v_dt", type=float, default=1.0, help="Velocity time step")
    parser.add_argument("--simulation_dt", type=float, default=0.08, help="Simulation time step")
    
    # HX: 运行test function
    parser.add_argument("--run_test", action="store_true", help="Run the test function")

    # 分布式训练参数
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")

    args, extra_args = parser.parse_known_args()

    cfg = create_config(args.config, args, extra_args)

    # 设置本地 rank
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    print(args.local_rank, "local rank")

    return cfg


if __name__ == "__main__":
    args = parse_args()

    # torch.backends.cuda.matmul.allow_tf32 = True

    trainer = Trainer(args)

    if args.run_test:  
        trainer.test()
    else:
        trainer.train()
        
    # if args.run_eval:
    #     trainer.demo(
    #         velo_scaling=args.velo_scaling,
    #         eval_ys=args.eval_ys,
    #         save_name=args.demo_name,
    #     )
    # else:
    #     # trainer.debug()
    #     trainer.test()
    # trainer.train()
    # trainer.test()


