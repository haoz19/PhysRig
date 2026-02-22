import torch
from time import time
from PIL import Image
import numpy as np
import sys
import json
import quaternion


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

@torch.jit.script
def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions: quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.shape[-1] != 3 or matrix.shape[-2] != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        torch.nn.functional.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5,
        :,  # pyre-ignore[16]
    ].reshape(batch_dim + (4,))


def se3_to_quaternion_translation(se3, tuple=True):
    q = matrix_to_quaternion(se3[..., :3, :3])
    t = se3[..., :3, 3]
    if tuple:
        return q, t
    else:
        return torch.cat((q, t), -1)

def interpolate_relative_root_motion(relative_root_motion, k):
    # sampling units
    sampling = [i/k for i in range(1, k)]
    # convert rotation matrix to quaternion
    relative_root_motion = torch.from_numpy(relative_root_motion)
    q_relative_root_motion = se3_to_quaternion_translation(relative_root_motion)
    q_relative_root_motion = torch.cat(q_relative_root_motion)
    q_relative_root_motion = q_relative_root_motion.numpy()
    translation = np.array(q_relative_root_motion[4:])
    # interpolate: Linear for translation / SLERP for rotation
    t_identity = np.array([0., 0., 0.])
    q_identity = np.quaternion(1, 0, 0, 0)
    q_rotation = np.quaternion(q_relative_root_motion[0], q_relative_root_motion[1], 
                               q_relative_root_motion[2], q_relative_root_motion[3])
    interpolated_q = [quaternion.slerp_evaluate(q_identity, q_rotation, i) for i in sampling]
    interpolated_rotation = [quaternion.as_rotation_matrix(q) for q in interpolated_q] # [np.array]
    interpolated_translation = [(1-i)*t_identity+i*translation for i in sampling] # [np.array]
    assert len(interpolated_rotation)==len(interpolated_translation)
    interpolated_root_motion = []
    for i in range(len(interpolated_rotation)):
        m = np.zeros((4, 4))
        R = interpolated_rotation[i]
        T = interpolated_translation[i]
        m[:3, :3] = R
        m[:3, 3] = T
        m[3, 3] = 1
        interpolated_root_motion.append(m)
    interpolated_root_motion.append(relative_root_motion.numpy())
    return interpolated_root_motion

def transform2origin(position):
    min_pos = np.min(position, 0)
    max_pos = np.max(position, 0)
    max_diff = np.max(max_pos - min_pos)
    original_mean_pos = (min_pos + max_pos) / 2.0
    scale = 1.0 / max_diff
    # original_mean_pos = original_mean_pos.to(device="cuda")
    # scale = scale.to(device="cuda")
    new_position = (position - original_mean_pos) * scale

    return new_position, scale, original_mean_pos

def sequence_t2o(position, scale, mean_pos):
    new_position = (position - mean_pos) * scale
    
    return new_position

def shift2center111(position):
    np111 = np.array([1.0, 1.0, 1.0])
    return position + np111


# motion_file_path = "./meshes/cat-pikachu-0-fg-skel/export_0000/fg-motion.json"


def color_vertices_by_label(obj_path, labels_array):
    # 读取 OBJ 文件
    with open(obj_path, 'r') as file:
        lines = file.readlines()
    
    # 创建一个空列表存储修改后的 OBJ 数据
    new_lines = []
    
    # 定义颜色映射
    colors = {
        4: "1.0 0.0 0.0",  # 红色 (red)
        6: "0.0 1.0 0.0",  # 绿色 (green)
        10: "0.0 0.0 1.0", # 蓝色 (blue)
        12: "1.0 1.0 0.0", # 黄色 (yellow)
    }
    
    # 默认白色 (white)
    default_color = "1.0 1.0 1.0"
    
    # 初始化顶点计数
    vertex_count = 0
    
    # 处理每一行
    for line in lines:
        if line.startswith('v '):
            # 将行拆分为顶点坐标和法线信息
            parts = line.strip().split()
            vertex_position = parts[1:4]
            
            # 获取该顶点的编号
            label = labels_array[vertex_count]
            
            # 根据编号染色
            if label in colors:
                color = colors[label]
            else:
                color = default_color
            
            # 添加带颜色的顶点信息
            new_line = f"v {' '.join(vertex_position)} {color}\n"
            new_lines.append(new_line)
            
            # 增加顶点计数
            vertex_count += 1
        else:
            new_lines.append(line)
    
    # 保存新的带有纹理的 OBJ 文件
    new_obj_path = obj_path.replace(".obj", "_colored.obj")
    with open(new_obj_path, 'w') as file:
        file.writelines(new_lines)
    
    print(f"Colored OBJ file saved as {new_obj_path}")

import trimesh

def get_delta_x_orign(motion_file_path, corr_file_path):
    motion_file = json.loads(open(motion_file_path, 'r').read())
    corr_file = json.loads(open(corr_file_path, 'r').read())
    
    # transform coordinates to related data
    bone_centers = np.array(motion_file["bone_centers"])
    bone_centers[:, :, [1, 2]] = bone_centers[:, :, [2, 1]]
    bone_centers[:, :, -2] *= -1
    motion = np.diff(bone_centers, axis=0)
    
    mapping = np.array(corr_file["source_vert_to_bone"])
    mapping_lf = np.where(mapping == 4)[0]
    mapping_rf = np.where(mapping == 6)[0]
    mapping_lb = np.where(mapping == 10)[0]
    mapping_rb = np.where(mapping == 12)[0]
    
    
    source_mesh_path = corr_file["source_mesh_path"]
    s_mesh = trimesh.load_mesh(source_mesh_path)
    vertices = s_mesh.vertices
    vertices_lf = vertices[mapping_lf]
    vertices_rf = vertices[mapping_rf]
    vertices_lb = vertices[mapping_lb]
    vertices_rb = vertices[mapping_rb]
    aabb_lf = np.max(vertices_lf, 0) - np.min(vertices_lf, 0)
    aabb_rf = np.max(vertices_rf, 0) - np.min(vertices_rf, 0)
    aabb_lb = np.max(vertices_lb, 0) - np.min(vertices_lb, 0)
    aabb_rb = np.max(vertices_rb, 0) - np.min(vertices_rb, 0)
    
    # color_vertices_by_label(source_mesh_path, mapping)

    # print(djshjhk) # breaking point

    delta_x_orign = [motion[:, 4, :] / aabb_lf.max(), motion[:, 6, :] / aabb_rf.max(), motion[:, 10, :] / aabb_lb.max(), motion[:, 12, :] / aabb_rb.max()]
    # print(aabb_lf.max(), aabb_rf.max(), aabb_lb.max(), aabb_rb.max())
    # print(np.array(delta_x_orign)[:, 0, :])
    # sys.exit()
    return delta_x_orign


def get_delta_x_orign_new(motion_file_path, corr_file_path):
    motion_file = json.loads(open(motion_file_path, 'r').read())
    corr_file = json.loads(open(corr_file_path, 'r').read())
    
    # Transform coordinates to related data
    bone_centers = np.array(motion_file["bone_centers"])
    bone_centers[:, :, [1, 2]] = bone_centers[:, :, [2, 1]]
    bone_centers[:, :, -2] *= -1
    motion = np.diff(bone_centers, axis=0)
    
    mapping = np.array(corr_file["source_vert_to_bone"])
    
    # Load the source mesh
    source_mesh_path = corr_file["source_mesh_path"]
    s_mesh = trimesh.load_mesh(source_mesh_path)
    vertices = s_mesh.vertices
    
    # Initialize lists to store results
    delta_x_orign = []
    
    # Find unique bone indices
    unique_bones = np.unique(mapping)
    
    for bone_index in unique_bones:
        # Find vertices associated with the current bone
        mapping_bone = np.where(mapping == bone_index)[0]
        vertices_bone = vertices[mapping_bone]
        
        # Calculate the axis-aligned bounding box (AABB)
        aabb_bone = np.max(vertices_bone, 0) - np.min(vertices_bone, 0)
        
        # Calculate the delta motion for this bone
        delta_x = motion[:, bone_index, :] / aabb_bone.max()
        
        # Append the result
        delta_x_orign.append(delta_x)
    
    return delta_x_orign

    
    
    

def get_velocity_list(motion_file_path, scale, shift, dt):
    
    velocity_list = []
    motion_file = json.loads(open(motion_file_path, 'r').read())
    
    # transform coordinates to related data
    bone_centers = np.array(motion_file["bone_centers"])
    bone_centers[:, :, [1, 2]] = bone_centers[:, :, [2, 1]]
    bone_centers[:, :, -2] *= -1
    
    rest_bone_centers = np.array(motion_file["rest_bone_center"])
    rest_bone_centers[:, [1, 2]] = rest_bone_centers[:, [2, 1]]
    rest_bone_centers[:, -2] *= -1

    # transform min-max
    # transform to origin with canonical center points
    # apply t2o and shift2center111 to all frames (align with PhysGaussian)

    rest_bone_centers_transformed, scale, original_mean_pos = transform2origin(rest_bone_centers)
    bone_centers = sequence_t2o(bone_centers, scale, original_mean_pos)
    # bone_centers = shift2center111(bone_centers)

    motion = np.diff(bone_centers, axis=0)
    
    # print("----motion----")
    left_leg_fore_v = motion[:, 4, :] / dt
    right_leg_fore_v = motion[:, 6, :] / dt
    left_leg_back_v = motion[:, 10, :] / dt 
    right_leg_back_v = motion[:, 12, :] / dt
    # left_leg_fore_v = motion[:, 7, :] / dt
    # right_leg_fore_v = motion[:, 11, :] / dt
    # left_leg_back_v = motion[:, 20, :] / dt 
    # right_leg_back_v = motion[:, 24, :] / dt
    velocity_list = [left_leg_fore_v, right_leg_fore_v, left_leg_back_v, right_leg_back_v]
    return velocity_list
    
def get_root_motinon(root_motion_file_path, frame_per_motion):
    interpolated_root_motion = []
    root_motion_file = json.loads(open(root_motion_file_path, 'r').read())
    root_motion = np.array(root_motion_file["root_motion"])
    for relative_root_motion in root_motion:
        root_motion_per_motion = interpolate_relative_root_motion(relative_root_motion, frame_per_motion)
        interpolated_root_motion += root_motion_per_motion
    print("total len of interpolated root motionlen: ", len(interpolated_root_motion))
    return interpolated_root_motion

def get_extrinsic(root_motion_file_path):
    root_motion_file = json.loads(open(root_motion_file_path, 'r').read())
    extrinsic = np.array(root_motion_file["bg2bev"][0])
    return extrinsic
    