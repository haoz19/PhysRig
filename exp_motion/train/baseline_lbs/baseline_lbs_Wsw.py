# 这一版不仅学transformation也学skinning weights

import os
import torch
import torch.nn as nn
import numpy as np
import trimesh
from pytorch3d.loss import point_mesh_face_distance
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)
from pytorch3d.io import load_obj, save_obj

import os
import torch
import numpy as np
from tqdm.notebook import tqdm

# Data structures and functions for rendering
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    SoftSilhouetteShader,
    SoftPhongShader,
    TexturesVertex
)

# add path for demo utils functions 
import sys
import os
import pdb

import torch
import torch.nn as nn
import argparse

# from nnutils import loss_utils

parser = argparse.ArgumentParser(description='Baseline LBS Training')
parser.add_argument('--dataset_dir', type=str, required=True, 
                    help='Path to the dataset directory (input)')
parser.add_argument('--output_dir', type=str, required=True, 
                    help='Path to the output directory')
args = parser.parse_args()


def get_subdirectories(path):
    """
    Returns a list of paths to all subdirectories within a given directory.

    Parameters:
    - path: The path to the directory whose subdirectories you want to list.

    Returns:
    - A list of paths to subdirectories.
    """
    # List all entries in the directory given by "path"
    entries = os.listdir(path)
    
    # Combine the entries with the base path to get absolute paths
    full_paths = [os.path.join(path, entry) for entry in entries]
    
    # Filter out the full paths that are directories
    subdirectory_paths = [entry for entry in full_paths if os.path.isdir(entry)]
    
    return subdirectory_paths


def quaternion_to_rotation_matrix(quaternion):
    w, x, y, z = quaternion.unbind(-1)
    return torch.stack([
        1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w,
        2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w,
        2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2
    ], dim=-1).reshape(-1, 3, 3)

class BonesModule(nn.Module):
    def __init__(self, num_bones, device):
        super(BonesModule, self).__init__()
        init_quaternions = torch.zeros(num_bones, 4, device=device)
        init_quaternions[:, 0] = 1.0  
        self.quaternions = nn.Parameter(init_quaternions, requires_grad=True)
        
        self.translations = nn.Parameter(torch.zeros(num_bones, 3, device=device), requires_grad=True)
        
    def forward(self):
        normalized_quaternions = self.quaternions / (self.quaternions.norm(dim=-1, keepdim=True)+ 1e-8)
        Rmat = quaternion_to_rotation_matrix(normalized_quaternions)
        # pdb.set_trace()
        Tmat = self.translations
        return Rmat, Tmat
    
def count_obj_files(directory):
    """
    计算指定目录中以 '.obj' 结尾的文件数量。

    参数:
    - directory: 要检查的目录路径。
    
    返回:
    - count: 以 '.obj' 结尾的文件数量。
    """
    count = 0
    # 检查目录是否存在
    if not os.path.exists(directory):
        print(f"给定的路径不存在: {directory}")
        return count

    # 遍历目录
    for entry in os.listdir(directory):
        # 构建完整的文件路径
        full_path = os.path.join(directory, entry)
        # 确保是文件且以 .obj 结尾
        if os.path.isfile(full_path) and full_path.endswith('.obj'):
            count += 1

    return count
    
def obj_to_cam(verts, Rmat, Tmat, nmesh, skin, tocam=True):
    """
    transform from canonical object coordinates to camera coordinates
    """
    verts = verts.clone().float()
    Rmat = Rmat.clone().float()
    Tmat = Tmat.clone().float()
    verts = verts.view(-1,3)

    bodyR = Rmat[::nmesh].clone()
    bodyT = Tmat[::nmesh].clone()
    if nmesh>1:
        vs = []
        for k in range(nmesh-1):
            partR = Rmat[k+1::nmesh].clone()
            partT = Tmat[k+1::nmesh].clone()
            vs.append( (verts.matmul(partR) + partT)[:,np.newaxis])
        vs = torch.cat(vs,1) # N, K, Nv, 3
        vs = vs.squeeze(0)
        vs = (vs * skin).sum(0)
    else:
        vs = verts
        
    vs = vs.float()
    if tocam:
        vs = vs.clone().matmul(bodyR) + bodyT
        vs = vs.view(-1,3)
    else:
        vs = vs.clone()
    return vs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
inpth = args.dataset_dir
mesh_inpth = os.path.join(inpth, "gt_mesh_obj")
outpth = args.output_dir
if not os.path.exists(outpth):
    os.makedirs(outpth)

    
skin_file = os.path.join(inpth, "skinning_weights.npy")

print(f"Vertices: {np.load(skin_file).shape[0]}; Bones: {np.load(skin_file).shape[1]}")
# pdb.set_trace()

gt_weights = torch.tensor(np.load(skin_file).T).to(device)
gt_weights = gt_weights.unsqueeze(2)
print(f"GT weights shape: {gt_weights.shape}")

num = count_obj_files(mesh_inpth)
count = 0
print(f"num of meshes: {num}")

# 定义列表保存每一帧的最终 chamfer loss
final_chamfer_loss_list = []


class LearnableSkinningWeights(nn.Module):
    def __init__(self, num_rows, num_cols):
        """
        :param num_rows: 权重矩阵的行数（例如，每个顶点对应一行）
        :param num_cols: 权重矩阵的列数（例如，骨骼数）
        """
        super(LearnableSkinningWeights, self).__init__()
        # 定义一个可学习参数，初始值可以随机初始化
        self.logits = nn.Parameter(torch.randn(num_rows, num_cols)).to(device)
    
    def forward(self):
        # 对每一行做 softmax，保证每一行的和为 1
        weights = torch.softmax(self.logits, dim=1)
        return weights

# 示例用法
num_vertices = np.load(skin_file).shape[0] # 例如：顶点数量
num_bones = np.load(skin_file).shape[1] # 例如：骨骼数量

skinning_weights_model = LearnableSkinningWeights(num_vertices, num_bones)
weights = skinning_weights_model()
print(f"Learnable weights shape: {weights.shape}")
weights = weights.transpose(0, 1).unsqueeze(2)
print(f"Learnable weights shape: {weights.shape}")

# weights = torch.tensor(np.load(skin_file).T).to(device)
# weights = weights.unsqueeze(2)
# weights.requires_grad=False
# pdb.set_trace()

for frame in range(1, num + 1):
    
    src_obj = os.path.join(inpth, "mesh", "mesh_frame_0001.obj")
    
    trg_obj = os.path.join(inpth, "gt_mesh_obj", f"deformed_mesh_frame_{frame:04d}.obj")
    
    if frame > 1:
        last_obj = os.path.join(outpth, f"output_mesh_frame_{frame-1:04d}.obj")
    else:
        last_obj = src_obj
    
    #rest_pth = target

    # We read the target 3D model using load_obj
    verts, faces, aux = load_obj(trg_obj)

    # verts is a FloatTensor of shape (V, 3) where V is the number of vertices in the mesh
    # faces is an object which contains the following LongTensors: verts_idx, normals_idx and textures_idx
    # For this tutorial, normals and textures are ignored.
    faces_idx = faces.verts_idx.to(device)
    verts = verts.to(device)

    # We scale normalize and center the target mesh to fit in a sphere of radius 1 centered at (0,0,0). 
    # (scale, center) will be used to bring the predicted mesh to its original center and scale
    # Note that normalizing the target mesh, speeds up the optimization but is not necessary!
    center = verts.mean(0)
    verts = verts - center
    scale = max(verts.abs().max(0)[0])
    verts = verts / scale

    # We construct a Meshes structure for the target mesh
    trg_mesh = Meshes(verts=[verts], faces=[faces_idx])


    # We read the src 3D model using load_obj
    verts2, faces2, aux2 = load_obj(src_obj)

    # verts is a FloatTensor of shape (V, 3) where V is the number of vertices in the mesh
    # faces is an object which contains the following LongTensors: verts_idx, normals_idx and textures_idx
    # For this tutorial, normals and textures are ignored.
    faces_idx2 = faces2.verts_idx.to(device)
    verts2 = verts2.to(device)

    # We scale normalize and center the target mesh to fit in a sphere of radius 1 centered at (0,0,0). 
    # (scale, center) will be used to bring the predicted mesh to its original center and scale
    # Note that normalizing the target mesh, speeds up the optimization but is not necessary!
    center2 = verts2.mean(0)
    verts2 = verts2 - center2
    scale2 = max(verts2.abs().max(0)[0])
    verts2 = verts2 / scale2

    
    # We construct a Meshes structure for the target mesh
    src_mesh = Meshes(verts=[verts2], faces=[faces_idx2])
    
    # We read the src 3D model using load_obj
    verts3, faces3, aux3 = load_obj(last_obj)

    # verts is a FloatTensor of shape (V, 3) where V is the number of vertices in the mesh
    # faces is an object which contains the following LongTensors: verts_idx, normals_idx and textures_idx
    # For this tutorial, normals and textures are ignored.
    faces_idx3 = faces3.verts_idx.to(device)
    verts3 = verts3.to(device)

    # We scale normalize and center the target mesh to fit in a sphere of radius 1 centered at (0,0,0). 
    # (scale, center) will be used to bring the predicted mesh to its original center and scale
    # Note that normalizing the target mesh, speeds up the optimization but is not necessary!
    center3 = verts3.mean(0)
    verts3 = verts3 - center3
    scale3 = max(verts3.abs().max(0)[0])
    verts3 = verts3 / scale3

    # We construct a Meshes structure for the target mesh
    last_mesh = Meshes(verts=[verts3], faces=[faces_idx3])
    

    #print(weights.shape)

    B = weights.shape[0]+1  
    model = BonesModule(B, 'cuda')
    
    # arap_loss_fn = loss_utils.ARAPLoss(src_mesh.verts_list()[0].cpu(), src_mesh.faces_list()[0].cpu()).cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=1.0, momentum=0.9)

    # Number of optimization steps
    Niter = 1000
    # Weight for the chamfer loss
    w_chamfer = 1.0 
    # Weight for mesh edge loss
    w_edge = 1.0 
    # Weight for mesh normal consistency
    w_normal = 0.05
    # Weight for mesh laplacian smoothing
    w_laplacian = 0.1 
    # Plot period for the losses
    plot_period = 250
    loop = tqdm(range(Niter))
    
    w_lm = 0.07
    w_arap = 0

    chamfer_losses = []
    laplacian_losses = []
    edge_losses = []
    normal_losses = []
    
    count += 1
    for i in loop:
        # Initialize optimizer
        optimizer.zero_grad()

        R, T = model()
        R = R.float()
        T = T.float()

        # Deform the mesh
        sample_src_all = obj_to_cam(src_mesh.verts_list()[0], R, T, B, weights)

        new_src_mesh = Meshes(verts=[sample_src_all.float()], faces=[src_mesh.faces_list()[0]])

        sample_trg = trg_mesh.verts_list()[0]

        sample_trg = sample_points_from_meshes(trg_mesh, 5000)
        sample_src = sample_points_from_meshes(new_src_mesh, 5000)

        # We compare the two sets of pointclouds by computing (a) the chamfer loss
        loss_chamfer, _ = chamfer_distance(sample_trg, sample_src)

        # and (b) the edge length of the predicted mesh
        loss_edge = mesh_edge_loss(new_src_mesh)


        # mesh normal consistency
        loss_normal = mesh_normal_consistency(new_src_mesh)

        # mesh laplacian smoothing
        loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="uniform")
        
        lmotion_loss = (sample_src_all.float() - last_mesh.verts_list()[0].float()).norm(2,-1).mean(-1)
        
        
        # Weighted sum of the losses
        loss = loss_chamfer * w_chamfer + loss_edge * w_edge + loss_normal * w_normal + loss_laplacian * w_laplacian + w_lm * lmotion_loss


        # Print the losses
        loop.set_description('total_loss = %.6f' % loss)

        # Save the losses for plotting
        chamfer_losses.append(float(loss_chamfer.detach().cpu()))
        edge_losses.append(float(loss_edge.detach().cpu()))
        normal_losses.append(float(loss_normal.detach().cpu()))
        laplacian_losses.append(float(loss_laplacian.detach().cpu()))


        # Optimization step
        loss.backward(retain_graph=True)
        optimizer.step()

        if i % 100 == 0:
            print('chanmfer loss:', float(loss_chamfer.detach().cpu()))
            # print('arap:', arap_loss.detach().cpu())
            print('lm:', lmotion_loss.detach().cpu())
            # Fetch the verts and faces of the final predicted mesh
        if i == Niter-1 or float(loss_chamfer.detach().cpu()) <= 0.0003:
            final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)

            # Scale normalize back to the original target size
            final_verts = final_verts * scale + center

            # Store the predicted mesh using save_obj
            final_obj = os.path.join(outpth, f"output_mesh_frame_{frame:04d}.obj")
            save_obj(final_obj, final_verts, final_faces)
            
            # 记录每帧最后的loss
            print('final chanmfer loss:', float(loss_chamfer.detach().cpu()))
            final_loss = float(loss_chamfer.detach().cpu())
            final_chamfer_loss_list.append(final_loss)

            print(f'Frame {frame} has finished!')
            break

# 计算所有帧的平均CD loss            
avg_chamfer_loss = sum(final_chamfer_loss_list) / len(final_chamfer_loss_list)
print(f'Average chamfer loss across {len(final_chamfer_loss_list)} frames: {avg_chamfer_loss}')

# 构造输出文件路径，并写入平均loss
output_file = os.path.join(outpth, "avg_chamfer_loss.txt")
with open(output_file, "w") as f:
    f.write(f"Average chamfer loss across {len(final_chamfer_loss_list)} frames: {avg_chamfer_loss}")
