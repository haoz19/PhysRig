import numpy as np
import plotly.graph_objects as go
import trimesh
import random
import pdb

def visualize_mesh_and_cuboids(mesh_path, cuboid_centers, cuboid_sizes, output_html_path):
    # 加载完整的 mesh，包括面信息
    mesh = trimesh.load(mesh_path, process=False)
    mesh_vertices = np.array(mesh.vertices)
    mesh_faces = np.array(mesh.faces)
    
    
    
    # 创建 plotly Figure 并添加 mesh
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=mesh_vertices[:, 0],
                y=mesh_vertices[:, 1],
                z=mesh_vertices[:, 2],
                i=mesh_faces[:, 0],
                j=mesh_faces[:, 1],
                k=mesh_faces[:, 2],
                color='lightgrey',
                opacity=0.5,
                name='Mesh'
            ),
        ]
    )

    # 可视化每个 cuboid
    for idx, (center, size) in enumerate(zip(cuboid_centers, cuboid_sizes)):
        # 随机生成颜色
        color = f'rgb({random.randint(0, 255)}, {random.randint(0, 255)}, {random.randint(0, 255)})'

        # 根据中心点和尺寸计算 cuboid 的8个顶点
        half_size = np.array(size) / 2
        corners = np.array([
            [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
            [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
        ]) * half_size + center

        # 定义 cuboid 的 12 条边，每条边由顶点索引表示
        edges = [
            [0, 1], [1, 3], [3, 2], [2, 0],
            [4, 5], [5, 7], [7, 6], [6, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]

        # 绘制 cuboid 的每条边（仅显示一次在图例中）
        for edge_idx, edge in enumerate(edges):
            show_legend = (edge_idx == 0)  # 仅在第一次绘制时显示图例
            fig.add_trace(
                go.Scatter3d(
                    x=[corners[edge[0], 0], corners[edge[1], 0]],
                    y=[corners[edge[0], 1], corners[edge[1], 1]],
                    z=[corners[edge[0], 2], corners[edge[1], 2]],
                    mode='lines',
                    line=dict(color=color, width=3),
                    name=f'Cuboid_{idx}' if show_legend else None,
                    showlegend=show_legend
                )
            )

    # 配置布局，设置白色背景和黑色网格线
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False, backgroundcolor="white", gridcolor="black"),
            yaxis=dict(visible=False, backgroundcolor="white", gridcolor="black"),
            zaxis=dict(visible=False, backgroundcolor="white", gridcolor="black")
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=700,
        width=700,
        title="Mesh and Cuboid Visualization"
    )

    # 保存为 HTML 文件
    fig.write_html(output_html_path)
    print(f"可视化结果已保存到 {output_html_path}")
    