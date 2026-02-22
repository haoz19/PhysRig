'''
引入cuboid
'''


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
        particle_v: Float[Tensor, "n 3"],
        particle_F: Float[Tensor, "n 3 3"],
        particle_C: Float[Tensor, "n 3 3"],
        E: torch.Tensor,
        nu: torch.Tensor,
        cuboid_velocity: torch.Tensor,  # Haolan: cuboid的速度参数
        cuboid_point: Float[Tensor, "cuboid_num 3"],     # cuboid的位置
        cuboid_size: Float[Tensor, "cuboid_num 3"],      # cuboid的大小
        frame_time_offset: float,
        particle_density: Optional[Float[Tensor, "n"] | Float[Tensor, "1"]]=None,
        device: str="cuda:0",
        requires_grad: bool=True,
    ) -> Tuple[Float[Tensor, "n 3"], Float[Tensor, "n 3"], Float[Tensor, "n 9"], Float[Tensor, "n 9"], Float[Tensor, "n 6"]]:


        E_inp = from_torch_safe(E, dtype=wp.float32, requires_grad=True)

        nu_inp = from_torch_safe(nu, dtype=wp.float32, requires_grad=True)

        wp_tape = MyTape()
           
        with wp_tape:
            
            num_particles = particle_x.shape[0]
        
            mpm_state.continue_from_torch(
                particle_x, particle_v, particle_F, particle_C, device=device, requires_grad=True
            )
        
            next_state_list = []
 
            mpm_solver.set_E_nu(mpm_model, E_inp, nu_inp, device=device)
            mpm_solver.prepare_mu_lam(mpm_model, mpm_state, device=device)

            mpm_state.reset_density(
                tensor_density=particle_density,
                selection_mask=None,
                device=device,
                requires_grad=True,
                update_mass=True)

            prev_state = mpm_state
            
            mpm_solver.prepare_mu_lam(mpm_model, prev_state, device=device)
            
            for i in range(cuboid_velocity.shape[0]):
                
                if i == 0:
                    clear = 1
                else:
                    clear = 0
                
                mpm_solver.set_velocity_on_cuboid_diff(
                    point=cuboid_point[i],
                    size=cuboid_size[i],
                    velocity_wp=cuboid_velocity[i], 
                    start_time=frame_time_offset,
                    end_time=substep_size*num_substeps+frame_time_offset,
                    reset=0,
                    clear=clear,
                )
                
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
            
            
            for substep_local in range(num_substeps):
                next_state = prev_state.partial_clone(requires_grad=True)
                
                mpm_solver.p2g2p_differentiable(mpm_model, prev_state, next_state, substep_size, device=device)
                next_state_list.append(next_state)
                prev_state = next_state
        
        ctx.prev_state = prev_state
        ctx.mpm_solver = mpm_solver
        ctx.mpm_model = mpm_model
        ctx.next_state_list = next_state_list
        ctx.device = device
        ctx.num_particles = num_particles
        ctx.tape = wp_tape
        
        ctx.cuboid_velocity = cuboid_velocity
        ctx.cuboid_velocity_device = cuboid_velocity.device 
        
        last_state = next_state
        
        particle_pos = wp.to_torch(last_state.particle_x).detach().clone().requires_grad_(True)
        particle_velo = wp.to_torch(last_state.particle_v).detach().clone().requires_grad_(True)
        particle_F = wp.to_torch(last_state.particle_F_trial).detach().clone().requires_grad_(True)
        particle_C = wp.to_torch(last_state.particle_C).detach().clone().requires_grad_(True)
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


        with wp.ScopedDevice(device):
            
            loss_wp = torch.zeros(1, device=device)
            loss_wp = wp.from_torch(loss_wp, requires_grad=True)
            target_pos_detach = wp.clone(next_state.particle_x, device=device, requires_grad=False)
            grad_pos_wp = from_torch_safe(out_pos_grad, dtype=wp.vec3, requires_grad=False)

            with tape:
                
                wp.launch(
                    compute_posloss_with_grad, 
                    dim=num_particles,
                    inputs=[
                        next_state,
                        target_pos_detach,
                        grad_pos_wp,
                        0.5,
                        loss_wp,
                    ],
                    device=device,
                )

            print("Loss_wp:", loss_wp)
            tape.backward(loss_wp) # Haolan：优化所需时间

        # grad for E, nu

        E_grad = wp.to_torch(mpm_model.E.grad).clone().to(device) 
        
        nu_grad = wp.to_torch(mpm_model.nu.grad).clone().to(device)

        tape.zero()

        
        # grad_min = E_grad.min().item()
        # grad_max = E_grad.max().item()
        # grad_mean = E_grad.mean().item()
        # print(f"  Min: {grad_min:.5e}, Max: {grad_max:.5e}, Mean: {grad_mean:.5e}")

        return (None, None, None, None, None,
                None, None, None, None, 
                E_grad, nu_grad, 
                None, None, None, None, # Haolan:对应 cuboid_velocity_grad, cuboid_point, cuboid_size, cuboid_time
                None, None, None, None)
