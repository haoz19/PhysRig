import os
import torch
import pdb
from jaxtyping import Float, Int, Shaped
from torch import Tensor
from time import time
from omegaconf import OmegaConf
from motionrep.fields.triplane_field import TriplaneFields, TriplaneFieldsWithEntropy

import cv2
import numpy as np


def get_volume(xyzs: np.ndarray, resolution=128) -> np.ndarray:

    # set a grid in the range of [-1, 1], with resolution
    voxel_counts = np.zeros((resolution, resolution, resolution))

    points_xyzindex = ((xyzs + 1) / 2 * (resolution - 1)).astype(np.uint32)
    cell_volume = (2.0 / (resolution - 1)) ** 3

    for x, y, z in points_xyzindex:
        voxel_counts[x, y, z] += 1

    points_number_in_corresponding_voxel = voxel_counts[
        points_xyzindex[:, 0], points_xyzindex[:, 1], points_xyzindex[:, 2]
    ]

    points_volume = cell_volume / points_number_in_corresponding_voxel

    points_volume = points_volume.astype(np.float32)

    # some statistics
    num_non_empyt_voxels = np.sum(voxel_counts > 0)
    max_points_in_voxel = np.max(voxel_counts)
    min_points_in_voxel = np.min(voxel_counts)
    avg_points_in_voxel = np.sum(voxel_counts) / num_non_empyt_voxels
    print("Number of non-empty voxels: ", num_non_empyt_voxels)
    print("Max points in voxel: ", max_points_in_voxel)
    print("Min points in voxel: ", min_points_in_voxel)
    print("Avg points in voxel: ", avg_points_in_voxel)

    return points_volume


def create_spatial_fields(
    args, output_dim, aabb: Float[Tensor, "2 3"], add_entropy=True
):

    sp_res = args.sim_res

    resolutions = [sp_res, sp_res, sp_res]
    reduce = "sum"


    model = TriplaneFields(
        aabb,
        resolutions,
        feat_dim=32,
        init_a=0.1,
        init_b=0.5,
        reduce=reduce,
        num_decoder_layers=2,
        decoder_hidden_size=32,
        output_dim=output_dim,
        zero_init=args.zero_init,
    )
    
    if args.zero_init:
        print("=> zero init the last layer for Spatial MLP")

    return model


class IntervalAnneal_cuboid(object):
    def __init__(
        self,
        total_iters,
        start_state=[1],  # 初始窗口大小
        end_state=[10],  # 完整窗口大小
        plateau_iters=-1,
        warmup_step=5,
        loss_threshold=None,  # 动态更新窗口的loss阈值
    ):
        self.total_iters = total_iters

        if plateau_iters < 0:
            plateau_iters = int(total_iters * 0.1)

        if warmup_step <= 0:
            warmup_step = 0

        self.total_iters = max(total_iters - plateau_iters - warmup_step, 10)
        self.start_state = start_state
        self.end_state = end_state
        self.warmup_step = warmup_step
        self.loss_threshold = loss_threshold

        if start_state[0] != (end_state[0] + 1):
            self.interval_iters = self.total_iters // (int(end_state[0]) + 1 - int(start_state[0]))
        else:
            self.interval_iters = self.total_iters

        self.current_window_size = start_state[0]
        self.fixed_update = True  # 是否固定更新
        
        print("total_iters: ", self.total_iters)
        print("interval_iters: ", self.interval_iters)

    def compute_state(self, cur_iter, loss=None):
        """
        动态或固定更新 window_size:
        - 如果 loss 提供，且小于 loss_threshold，则动态更新 window_size。
        - 否则按照固定步长更新。
        """
        # 动态更新逻辑
        if loss is not None and self.loss_threshold is not None:
            if loss < self.loss_threshold:
                # 边界检查
                if self.current_window_size < self.end_state[0]:
                    self.current_window_size += 1 # 每次增加新的一帧
                    self.fixed_update = False  # 保持动态更新
                    print(f"Dynamic update: loss={loss}, new window_size={self.current_window_size}")
                else:
                    print(f"Dynamic update reached max window size: {self.end_state[0]}")
                    return self.end_state  # 达到最大窗口时直接返回
            else:
                print(f"Loss not below threshold. Keeping current window_size: {self.current_window_size}")
                self.fixed_update = True  # loss不够小，但保持固定更新

        if self.fixed_update:
            if self.warmup_step > 0:
                cur_iter = max(0, cur_iter - self.warmup_step) # 保证在warmup中，cur_iter是0

            # 如果迭代已经到达总次数，返回最终状态
            if cur_iter >= self.total_iters:
                self.current_window_size = self.end_state[0]
            elif (cur_iter - self.warmup_step) % self.interval_iters == 0 and cur_iter > 0:
                # 仅当 cur_iter 满足 interval_iters 的整倍数时，增加 window_size，其余情况不改变 window_size
                self.current_window_size += 1

            if self.current_window_size >= self.end_state[0]:
                self.current_window_size = self.end_state[0]

        return [self.current_window_size]

