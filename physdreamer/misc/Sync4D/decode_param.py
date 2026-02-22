import json
import warp as wp
from thirdparty_code.warp_mpm.mpm_solver_diff import MPMWARPDiff
from thirdparty_code.warp_mpm.mpm_data_structure import (
    MPMStateStruct,
    MPMModelStruct,
    get_float_array_product,
)

# HX: 需要修改point和size，来适应我们的skeleton

def wrap_dv_params(velocity_list, bc_params):
    dv_params = {}
    for part_id, velocity in enumerate(velocity_list):
        dv_params[part_id] = {
            "velocity": velocity,
            "point" : bc_params["point"][part_id],
            "size" : bc_params["size"][part_id]
        }
    return dv_params
    

def set_driven_velocity(
    mpm_solver: MPMWARPDiff, mpm_state: MPMStateStruct, dv_params: dict, dt: float
):
    # keep object unmoved
    mpm_solver.enforce_particle_velocity_translation(
        mpm_state=mpm_state,
        point=[1, 1, 1.2],
        size=[0.2, 0.2, 0.2],
        velocity=[0, 0, 0],
        start_time=0,
        end_time=1e3,
    )
    driven_p = 0.5
    # set velocity each part during dt iteration
    for part_id, bc in dv_params.items():
        for i, v in enumerate(bc["velocity"]):
            mpm_solver.enforce_particle_velocity_translation_fixed(
                mpm_state,
                point=bc["point"],
                size=bc["size"],
                velocity=v / driven_p * 2,
                part_id=part_id,
                start_time=i * dt,
                end_time=(i + driven_p) * dt 
            )
    

