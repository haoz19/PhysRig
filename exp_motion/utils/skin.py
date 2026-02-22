import numpy as np
import plotly.graph_objects as go
import plotly.offline as py
import trimesh as tm
import json
import os
import argparse

# 解析输入参数
parser = argparse.ArgumentParser(description="Skinning weights processing script")
parser.add_argument("--mesh_path", required=True, help="Path to input mesh file")
parser.add_argument("--json_weights_path", required=True, help="Path to skinning weights JSON file")
parser.add_argument("--output_weights_path", required=True, help="Path to save skinning weights npy")
parser.add_argument("--output_masks_path", required=True, help="Path to save skinning masks npy")
parser.add_argument("--visualization_path", required=True, help="Path for saving visualization results")

args = parser.parse_args()

if not os.path.exists(args.json_weights_path) or os.stat(args.json_weights_path).st_size == 0:
    print(f"Error: json_weights_path is empty or does not exist: {args.json_weights_path}")
    exit(1)
    
def parse_json_weights(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    bone_names = list(data.keys())
    all_indices = set()

    # 提取所有顶点索引，并确保转换为整数
    for bone_name, weights in data.items():
        for vertex_id in weights.keys():
            try:
                all_indices.add(int(vertex_id))
            except ValueError:
                print(f"Warning: Invalid vertex index '{vertex_id}' in bone '{bone_name}'")

    num_vertices = len(all_indices)
    num_bones = len(bone_names)

    print(f"Total bones: {num_bones}, Total vertices: {num_vertices}")

    vertex_map = {v_idx: i for i, v_idx in enumerate(sorted(all_indices))}
    bone_map = {bone: i for i, bone in enumerate(bone_names)}

    # 初始化权重矩阵
    weights = np.zeros((num_vertices, num_bones), dtype=np.float32)
    
    # 填充权重矩阵
    for bone_name, vertices in data.items():
        bone_idx = bone_map[bone_name]
        for vertex_id, weight in vertices.items():
            global_idx = vertex_map[int(vertex_id)]
            weights[global_idx, bone_idx] = weight

    return weights, bone_names, vertex_map

def generate_masks(weights):
    labels = np.argmax(weights, axis=1)
    num_bones = weights.shape[1]
    masks = np.zeros((num_bones, weights.shape[0]), dtype=bool)
    for bone_idx in range(num_bones):
        masks[bone_idx] = (labels == bone_idx)
    return masks

# 解析 skinning weights
skinning_weights, bone_names, vertex_map = parse_json_weights(args.json_weights_path)
np.save(args.output_weights_path, skinning_weights)

# 生成掩码并保存
masks = generate_masks(skinning_weights)
np.save(args.output_masks_path, masks)

print(f"Skinned weights saved to: {args.output_weights_path}")
print(f"Skinned masks saved to: {args.output_masks_path}")

def visualize_skin_weights(mesh_path, weights, output_dir):
    labels = np.argmax(weights, axis=1)
    num_bones = weights.shape[1]

    mesh = tm.load(mesh_path, process=False)

    # 检查 mesh 顶点数量是否匹配 weights
    num_vertices = len(mesh.vertices)
    max_face_index = np.max(mesh.faces)

    if max_face_index >= num_vertices:
        raise ValueError(f"Mesh face index {max_face_index} exceeds vertex count {num_vertices}")

    if num_vertices != weights.shape[0]:
        print(f"Warning: Mesh has {num_vertices} vertices but weights have {weights.shape[0]}")
    
    # 生成随机颜色，每个骨骼分配一种颜色
    bone_colors = np.random.randint(50, 255, size=(num_bones, 3)) / 255.0
    vertex_colors = np.zeros((num_vertices, 3))  # 确保大小匹配

    # 仅填充已知顶点的颜色，确保不超出索引范围
    for idx, label in enumerate(labels):
        if idx < len(vertex_colors):
            vertex_colors[idx] = bone_colors[label]

    # 计算每个面片的平均颜色
    face_colors = np.mean(vertex_colors[mesh.faces], axis=1)

    # Mesh 顶点和三角面
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                facecolor=tuple(map(tuple, face_colors)),
                opacity=1.0
            )
        ],
        layout=dict(
            scene=dict(
                aspectmode='data', 
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False)
            ),
            margin=dict(r=0, l=0, b=0, t=0)
        )
    )

    output_html = os.path.join(output_dir, "visualized_skin_weights.html")
    py.plot(fig, filename=output_html)
    print(f"Visualization saved to: {output_html}")
    
    obj_output_path = os.path.join(output_dir, "visualized_skin_weights.obj")
    mesh_data = tm.Trimesh(vertices=vertices, faces=faces, vertex_colors=vertex_colors)
    mesh_data.export(obj_output_path)

    print(f"OBJ file saved to: {obj_output_path}")
    
visualize_skin_weights(args.mesh_path, skinning_weights, args.visualization_path)
