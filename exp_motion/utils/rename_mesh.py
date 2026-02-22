import os
import re
import shutil

def copy_and_rename_meshes(input_folder, output_folder, prefix="textured_mesh_frame", zero_pad=4):
    """
    从 input_folder 复制网格文件到 output_folder，并改名为:
    mesh_frame_0001.obj, mesh_frame_0002.obj, ...
    排序依据文件名中的数字部分排序。
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 构造正则匹配原始文件名中的数字部分
    pattern = re.compile(r'textured_mesh_frame_(\d+)\.obj$', re.IGNORECASE)
    
    # 收集匹配的文件及其数字
    file_list = []
    for filename in os.listdir(input_folder):
        m = pattern.match(filename)
        if m:
            num = int(m.group(1))
            file_list.append((filename, num))
    
    # 根据数字排序，而不是直接的字典序
    file_list.sort(key=lambda x: x[1])
    
    # 依次复制并改名
    for idx, (filename, num) in enumerate(file_list):
        old_path = os.path.join(input_folder, filename)
        _, ext = os.path.splitext(filename)
        # 如果你想从1开始编号，则 idx+1
        new_filename = f"textured_mesh_frame_{idx+1:0{zero_pad}d}{ext}"
        new_path = os.path.join(output_folder, new_filename)
        shutil.copy2(old_path, new_path)
        print(f"{filename} -> {new_filename}")

if __name__ == "__main__":
    input_dir = "/scratch/bbqk/haozhang/files/physdreamer/PhysDreamer/data/process/mesh_raw"
    output_dir = "/scratch/bbqk/haozhang/files/physdreamer/PhysDreamer/data/process/mesh_renamed"
    copy_and_rename_meshes(input_dir, output_dir, prefix="textured_mesh_frame", zero_pad=4)