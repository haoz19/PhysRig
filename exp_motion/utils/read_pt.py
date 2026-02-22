import torch
import numpy as np

# 读取 pt 文件
file_path = "/scratch/bbqk/haozhang/files/physdreamer/PhysDreamer/physdreamer/misc/models/physdreamer/alocasia/model/sim_fields.pt"  # 替换为你的 .pt 文件路径
data = torch.load(file_path, map_location=torch.device('cpu'))  # 加载到 CPU

# 检查数据类型
if isinstance(data, torch.Tensor):
    np_data = data.numpy()
    print("数据是一个张量，转换为 NumPy 格式：")
    print(np_data)
elif isinstance(data, dict):
    print("数据是一个字典，包含以下键：", data.keys())
    np_data_dict = {k: v.numpy() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
    for key, value in np_data_dict.items():
        print(f"{key}: {type(value)}")
elif isinstance(data, list):
    print("数据是一个列表，长度为：", len(data))
    np_data_list = [v.numpy() if isinstance(v, torch.Tensor) else v for v in data]
    print(np_data_list[:3])  # 仅预览前3个元素
else:
    print("数据格式未知:", type(data))
