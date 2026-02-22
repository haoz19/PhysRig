import numpy as np
import torch
import open3d as o3d

class PointCloudProcessor:
    def __init__(self, reference_ply_path):
        """
        初始化点云处理器，计算 scale 和 shift。
        :param reference_ply_path: 用于计算 scale 和 shift 的参考点云文件路径
        """
        ref_pcd = o3d.io.read_point_cloud(reference_ply_path)
        ref_points = np.asarray(ref_pcd.points)
        
        # 转换为 PyTorch 张量
        sim_xyzs = torch.tensor(ref_points, dtype=torch.float32)
        
        # 计算 scale 和 shift
        pos_max = sim_xyzs.max()
        pos_min = sim_xyzs.min()
        scale = (pos_max - pos_min) * 1.8
        shift = -pos_min + (pos_max - pos_min) * 0.25
        
        self.scale, self.shift = scale, shift
        print("scale, shift", scale.item(), shift.item())

    def process(self, ply_path, output_path):
        """
        处理点云数据。
        :param ply_path: 需要处理的点云文件路径
        :param output_path: 处理后点云的保存路径
        """
        pcd = o3d.io.read_point_cloud(ply_path)
        points = np.asarray(pcd.points)
        
        # 处理点云
        points = points * self.scale.item() - self.shift.item()
        
        # 保存处理后的点云
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(output_path, pcd)
        print(f"处理后的点云已保存至 {output_path}")

# 示例用法
if __name__ == "__main__":
    reference_ply = "/scratch/bbqk/haozhang/files/physdreamer/PhysDreamer/data/process/init_ply/infilled_0.ply"
    target_ply = "/scratch/bbqk/haozhang/files/physdreamer/PhysDreamer/data/process/raw_ply/youngs_map_iter_0.ply"
    output_ply = "/scratch/bbqk/haozhang/files/physdreamer/PhysDreamer/data/process/processed_ply/processed_youngs_map.ply"
    
    processor = PointCloudProcessor(reference_ply)
    processor.process(target_ply, output_ply)
