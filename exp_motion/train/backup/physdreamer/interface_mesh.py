from typing import Optional, Tuple
from jaxtyping import Float, Int, Shaped
import torch
import torch.autograd as autograd
import torch.nn as nn
from torch import Tensor

import warp as wp
import pdb
import time

from thirdparty_code.warp_mpm.warp_utils import from_torch_safe, MyTape, CondTape
from thirdparty_code.warp_mpm.mpm_solver_diff_cuboid import MPMWARPDiff
from thirdparty_code.warp_mpm.mpm_utils import compute_position_l2_loss, aggregate_grad, compute_posloss_with_grad
from thirdparty_code.warp_mpm.mpm_data_structure import MPMStateStruct, MPMModelStruct, get_float_array_product
from thirdparty_code.warp_mpm.mpm_utils import (compute_Closs_with_grad, compute_Floss_with_grad, 
                                                compute_posloss_with_grad, compute_veloloss_with_grad)


# Haolan: 引入cuboid_velocity的学习
class MPMDifferentiableSimulationRig(autograd.Function):
    """
    Current version does not support grad for density. 
    Please set vol, mass before calling this function.
    """

    @staticmethod
    @torch.no_grad()
    def forward(
        ctx: autograd.function.FunctionCtx,
        mpm_solver: MPMWARPDiff,
        mpm_state: MPMStateStruct,
        mpm_model: MPMModelStruct,
        substep_size: float, 
        num_substeps: int,
        particle_x: Float[Tensor, "n 3"], 
        particle_v: Float[Tensor, "n 3"], # Haolan:初速度应该设为0，改为cubiod的速度影响
        particle_F: Float[Tensor, "n 3 3"],
        particle_C: Float[Tensor, "n 3 3"],
        E: Float[Tensor, "n"] | Float[Tensor, "1"],
        nu: Float[Tensor, "n"] | Float[Tensor, "1"],
        cuboid_velocity: Float[Tensor, "cuboid_num 3"],  # Haolan: cuboid的速度参数
        cuboid_point: Float[Tensor, "cuboid_num 3"],     # cuboid的位置
        cuboid_size: Float[Tensor, "cuboid_num 3"],      # cuboid的大小
        frame_time_offset: float,
        particle_density: Optional[Float[Tensor, "n"] | Float[Tensor, "1"]]=None,
        query_mask: Optional[Int[Tensor, "n"]] = None,
        device: str="cuda:0",
        requires_grad: bool=True,
        extra_no_grad_steps: int=0,
    ) -> Tuple[Float[Tensor, "n 3"], Float[Tensor, "n 3"], Float[Tensor, "n 9"], Float[Tensor, "n 9"], Float[Tensor, "n 6"]]:
        """
        Args:
            query_mask: [n] 0 or 1.  1 means the density or young's modulus, or poisson'ratio of this particle can change.
        """
        

        # initialization work is done before calling forward! 

        num_particles = particle_x.shape[0]
        
        mpm_state.continue_from_torch(
            particle_x, particle_v, particle_F, particle_C, device=device, requires_grad=True
        )
        # set x, v, F, C.

        if E.ndim == 0:
            E_inp = E.item() # float
            ctx.aggregating_E = True
        else:
            E_inp = from_torch_safe(E, dtype=wp.float32, requires_grad=True)
            ctx.aggregating_E = False
        if nu.ndim == 0:
            nu_inp = nu.item() # float
            ctx.aggregating_nu = True
        else:
            nu_inp = from_torch_safe(nu, dtype=wp.float32, requires_grad=True)
            ctx.aggregating_nu = False
            
        mpm_solver.set_E_nu(mpm_model, E_inp, nu_inp, device=device)
        mpm_solver.prepare_mu_lam(mpm_model, mpm_state, device=device)
        
        mpm_state.reset_density(
            tensor_density=particle_density,
            selection_mask=query_mask,
            device=device,
            requires_grad=True,
            update_mass=True)
        
        prev_state = mpm_state
        
        # Haolan: 这里进行无梯度模拟的目的是让模拟先趋于稳定
        if extra_no_grad_steps > 0:
            with torch.no_grad():
                for i in range(extra_no_grad_steps):
                    next_state = prev_state.partial_clone(requires_grad=True)
                    mpm_solver.p2g2p_differentiable(mpm_model, prev_state, next_state, substep_size, device=device)
                    prev_state = next_state

        # following steps will be checkpointed. then replayed in backward
        ctx.prev_state = prev_state

        wp_tape = MyTape()
        cond_tape: CondTape = CondTape(wp_tape, requires_grad)
        
        next_state_list = [] 
        
        # Haolan:将 cuboid_velocity 转换为 Warp 数组，方便传入set_velocity_on_cuboid_diff
        cuboid_velocity_wp_list = []
        for i in range(cuboid_velocity.shape[0]):
            velocity_vec3 = wp.vec3(
                cuboid_velocity[i][0],
                cuboid_velocity[i][1],
                cuboid_velocity[i][2]
            )
            velocity_wp = wp.array([velocity_vec3], dtype=wp.vec3, requires_grad=True, device=device)
            cuboid_velocity_wp_list.append(velocity_wp)
        
        # Haolan：保存用于backward
        ctx.cuboid_velocity_wp_list = cuboid_velocity_wp_list
        
        # pdb.set_trace()
        
        with cond_tape:
            wp.launch(
                kernel=get_float_array_product,
                dim=num_particles,
                inputs=[
                    prev_state.particle_density,
                    prev_state.particle_vol,
                    prev_state.particle_mass,
                ],
                device=device,
            )
            mpm_solver.prepare_mu_lam(mpm_model, prev_state, device=device)
            
            # Haolan:使用cuboid_velocity引入到Tape中，使得速度可微
            for i in range(cuboid_velocity.shape[0]):
                mpm_solver.set_velocity_on_cuboid_diff(
                    point=cuboid_point[i],
                    size=cuboid_size[i],
                    velocity_wp_array=cuboid_velocity_wp_list[i], 
                    start_time=frame_time_offset,
                    end_time=substep_size*num_substeps+frame_time_offset, # Haolan:每个substep进行更新
                )
                    
            for substep_local in range(num_substeps):
                next_state = prev_state.partial_clone(requires_grad=True)
                
                mpm_solver.p2g2p_differentiable(mpm_model, prev_state, next_state, substep_size, device=device) # Haolan：p2g2p_differentiable让p2g2p过程可微，这样才能让autograd和tape协同计算梯度
                next_state_list.append(next_state)
                prev_state = next_state
        
        # pdb.set_trace() # Haolan: For testing
        
        ctx.mpm_solver = mpm_solver
        ctx.mpm_model = mpm_model
        ctx.next_state_list = next_state_list
        ctx.device = device
        ctx.num_particles = num_particles
        ctx.tape = cond_tape.tape
        
        # Haolan:保存cuboid_velocity
        
        # 确保在device上
        ctx.cuboid_velocity_device = cuboid_velocity.device 
        # ctx.save_for_backward(cuboid_velocity, query_mask)
        ctx.save_for_backward(query_mask)
        
        
        last_state = next_state
        particle_pos = wp.to_torch(last_state.particle_x).detach().clone()
        particle_velo = wp.to_torch(last_state.particle_v).detach().clone()
        particle_F = wp.to_torch(last_state.particle_F_trial).detach().clone()
        particle_C = wp.to_torch(last_state.particle_C).detach().clone()
        # [N * 6, ]
        particle_cov = wp.to_torch(last_state.particle_cov).detach().clone()

        particle_cov = particle_cov.view(-1, 6)
        
        
        return particle_pos, particle_velo, particle_F, particle_C, particle_cov
    

    @staticmethod
    def backward(ctx, out_pos_grad: Float[Tensor, "n 3"], out_velo_grad: Float[Tensor, "n 3"], 
                 out_F_grad: Float[Tensor, "n 9"], out_C_grad: Float[Tensor, "n 9"], out_cov_grad: Float[Tensor, "n 6"]):
        
        
        num_particles = ctx.num_particles
        device = ctx.device
        mpm_solver, mpm_model = ctx.mpm_solver, ctx.mpm_model
        tape = ctx.tape
        starting_state = ctx.prev_state
        
        next_state_list = ctx.next_state_list
        next_state = next_state_list[-1]

        # Haolan：在 backward 中恢复 cuboid_velocity
        # cuboid_velocity, query_mask = ctx.saved_tensors
        query_mask = ctx.saved_tensors
    
        # print("Device of cuboid_velocity after loading:", cuboid_velocity.device)
        
        with wp.ScopedDevice(device):
            
            grad_pos_wp = from_torch_safe(out_pos_grad, dtype=wp.vec3, requires_grad=False)
            
            # Haolan：tape记录和追踪在前向传播中涉及到的所有计算操作，帮助pytorch的autograd计算梯度
            with tape:
                loss_wp = torch.zeros(1, device=device)
                loss_wp = wp.from_torch(loss_wp, requires_grad=True)
                target_pos_detach = wp.clone(next_state.particle_x, device=device, requires_grad=False)
                wp.launch(
                    compute_posloss_with_grad, 
                    dim=num_particles,
                    inputs=[
                        next_state,
                        target_pos_detach,
                        grad_pos_wp,
                        0.1, # 0.5
                        loss_wp,
                    ],
                    device=device,
                )
            
                
            # print("grad_pos_wp:", grad_pos_wp)
            print("Loss_wp:", loss_wp)
            # wp.synchronize_device(device)   
            tape.backward(loss_wp) # Haolan：优化所需时间


            
            # from IPython import embed; embed()
            
        # Haolan：提取各个参数的梯度
        pos_grad = wp.to_torch(starting_state.particle_x.grad).detach().clone()
        velo_grad = wp.to_torch(starting_state.particle_v.grad).detach().clone()
        F_grad = wp.to_torch(starting_state.particle_F_trial.grad).detach().clone()
        C_grad = wp.to_torch(starting_state.particle_C.grad).detach().clone()
        # print("debug back", velo_grad)
        
        
        # Haolan：提取 cuboid 速度梯度
        cuboid_velocity_grad_list = []
        for velocity_wp in ctx.cuboid_velocity_wp_list:
            velocity_grad = wp.to_torch(velocity_wp.grad, requires_grad=False).detach().clone().to(ctx.cuboid_velocity_device) # requires_grad=False for backward
            velocity_grad = velocity_grad.squeeze(0)
            cuboid_velocity_grad_list.append(velocity_grad)
        cuboid_velocity_grad = torch.stack(cuboid_velocity_grad_list) # 将多个cuboid的梯度转换成一个张量

        
        # grad for E, nu. TODO: add spatially varying E, nu later
        if ctx.aggregating_E:
            E_grad = wp.from_torch(torch.zeros(1, device=device), requires_grad=False)
            wp.launch(
                aggregate_grad, # Haolan：如果是标量则需要对粒子的梯度进行聚合
                dim=num_particles,
                inputs=[
                    E_grad,
                    mpm_model.E.grad,
                ],
                device=device,
            )
            E_grad = wp.to_torch(E_grad)[0] / num_particles
        else:
            E_grad = wp.to_torch(mpm_model.E.grad).detach().clone() # Haolan：如果是张量则直接提取

        if ctx.aggregating_nu:
            nu_grad = wp.from_torch(torch.zeros(1, device=device), requires_grad=False)
            wp.launch(
                aggregate_grad,
                dim=num_particles,
                inputs=[nu_grad, mpm_model.nu.grad],
                device=device,
            )
            nu_grad = wp.to_torch(nu_grad)[0] / num_particles   
        else:
            nu_grad = wp.to_torch(mpm_model.nu.grad).detach().clone()

        
        # grad for density
        if starting_state.particle_density.grad is None:
            density_grad = None
        else:
            density_grad = wp.to_torch(starting_state.particle_density.grad).detach()
        density_mask_grad = None

        tape.zero()
        
        
        # print("After loss_wp Cuboid_velocity.grad", cuboid_velocity_grad)
        
        # print("mpm_model.E_grad:", E_grad)
        
        # print("mpm_model.nu.grad:", nu_grad)

        # print(density_grad.abs().sum(), velo_grad.abs().sum(), E_grad.abs().item(), nu_grad.abs().item(), "in sim func")
        # from IPython import embed; embed()
        
        
        # Haolan: 返回cuboid velocity的grad 
        return (None, None, None, None, None,
                pos_grad, velo_grad, F_grad, C_grad, 
                E_grad, nu_grad, 
                cuboid_velocity_grad, None, None, None, # Haolan:对应 cuboid_velocity, cuboid_point, cuboid_size, cuboid_time
                density_grad, density_mask_grad,
                None, None, None)
