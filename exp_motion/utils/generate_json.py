import json
import trimesh

mesh_path = "/taiga/illinois/eng/ece/n-ahuja/haozhang/tjx/PHYSDREAMER/PhysDreamer/data/sphere/mesh/mesh_frame_0001.obj"
mesh = trimesh.load(mesh_path, process=False)
num_vertices = len(mesh.vertices)

# 构造全1权重
weights = {str(i): 1.0 for i in range(num_vertices)}
data = {"body": weights}

with open("skinning_weights.json", "w") as f:
    json.dump(data, f, indent=2)

print(f"已生成 skinning_weights.json，所有顶点都属于同一个 part。")