import numpy as np
import os

# 定义目录变量
data_dir = "/scratch/bbqk/haozhang/files/physdreamer/PhysDreamer/output/bellyman/pushup/velo_learn/bellyman_pushup_spv_train_fbf_newsche_vf_1.0_SP_50_youngs_50000.0_lr_0.005_substep_100_iters_580_frames_30/"

# 读取 npy 文件
loss_array = np.load(os.path.join(data_dir, "loss_array.npy"))

# print("loss_array:\n", loss_array)

min_values = []

# 遍历每一列，即每个frame
for i in range(loss_array.shape[1]):  # 遍历列
    col_data = loss_array[:, i]  # 获取当前列数据
    valid_data = col_data[~np.isnan(col_data)]  # 过滤掉 NaN 值
    
    if valid_data.size > 0:
        first_val = valid_data[0]  # 第一个非 NaN 值
        last_val = valid_data[-1]  # 最后一个非 NaN 值
        min_val = np.min(valid_data)  # 计算最小值
        min_values.append(min_val)  # 存储最小值
        print(f"列 {i}: 第一个非 NaN 值 = {first_val:.6f}, 最后一个非 NaN 值 = {last_val:.6f}, 最小值 = {min_val:.6f}")
        
        # print(f"最小值 = {min_val:.6f}")
    else:
        print(f"列 {i}: 全是 NaN，没有有效值")

if min_values:
    avg_min_value = np.mean(min_values)
    print(f"\n所有列的最小值的平均值: {avg_min_value:.6f}")
else:
    print("\n没有有效的最小值，无法计算平均值。")
    
# 指定保存路径
save_path = os.path.join(data_dir, "loss_array.txt")

# 保存为 txt 文件，保持矩阵格式
np.savetxt(save_path, loss_array, fmt="%.6f", delimiter=" ")

print(f"完整的 loss_array 数据已保存到 {save_path}")

