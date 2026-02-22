import numpy as np
import trimesh
from sklearn.decomposition import PCA
import os

def load_mesh(file_path):
    return trimesh.load(file_path)

def separate_loose_parts(mesh):
    return mesh.split(only_watertight=False)

def compute_base_apex(mesh):
    verts = np.array(mesh.vertices)
    if verts.shape[0] < 4:
        return None, None
    pca = PCA(n_components=3)
    pca.fit(verts)
    main_axis = pca.components_[0]

    projections = verts @ main_axis
    apex_idx = np.argmax(projections)
    base_idx = np.argsort(projections)[:3]
    base_center = verts[base_idx].mean(axis=0)
    apex = verts[apex_idx]
    return base_center, apex

def line_segment_distance(p1, p2, q1, q2):
    """返回两条线段最近点和距离"""
    u = p2 - p1
    v = q2 - q1
    w0 = p1 - q1
    a = np.dot(u, u)
    b = np.dot(u, v)
    c = np.dot(v, v)
    d = np.dot(u, w0)
    e = np.dot(v, w0)
    D = a*c - b*b
    sc, tc = 0, 0
    if D < 1e-8:  # 平行
        sc = 0.0
        tc = e / c if c > 0 else 0.0
    else:
        sc = (b*e - c*d)/D
        tc = (a*e - b*d)/D
    sc = np.clip(sc, 0, 1)
    tc = np.clip(tc, 0, 1)
    closest_p = p1 + sc * u
    closest_q = q1 + tc * v
    dist = np.linalg.norm(closest_p - closest_q)
    return closest_p, closest_q, dist

def adjust_axes_for_intersections(bases, apices):
    """调整椭球轴，使其只接触不穿透"""
    n = len(bases)
    centers = []
    directions = []
    lengths = []

    for i in range(n):
        base_i, apex_i = bases[i], apices[i]
        dir_i = apex_i - base_i
        length_i = np.linalg.norm(dir_i)
        center_i = (base_i + apex_i)/2
        # 检查和其他线段交点
        for j in range(n):
            if i == j:
                continue
            base_j, apex_j = bases[j], apices[j]
            _, _, dist = line_segment_distance(base_i, apex_i, base_j, apex_j)
            if dist < 1e-6:  # 相交或接触
                # 改变长半径，中心改为到交点的中点
                vec = apex_i - base_i
                t = 0.5  # 简单取中点
                apex_i = base_i + vec * t
                center_i = (base_i + apex_i)/2
                length_i = np.linalg.norm(apex_i - base_i)
        centers.append(center_i)
        directions.append((apex_i - base_i)/length_i)
        lengths.append(length_i/2)
    return centers, directions, lengths

def create_ellipsoid_mesh(center, long_radius, short_radius, direction, segments=16):
    u = np.linspace(0, 2*np.pi, segments)
    v = np.linspace(0, np.pi, segments)
    x = long_radius * np.outer(np.cos(u), np.sin(v))
    y = short_radius * np.outer(np.sin(u), np.sin(v))
    z = short_radius * np.outer(np.ones_like(u), np.cos(v))
    verts = np.stack([x, y, z], axis=-1).reshape(-1,3)

    def rotation_matrix_from_vectors(vec1, vec2):
        a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        if s < 1e-8:
            return np.eye(3)
        kmat = np.array([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]])
        R = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s**2))
        return R
    R = rotation_matrix_from_vectors(np.array([1,0,0]), direction)
    verts = (R @ verts.T).T + center

    faces = []
    for i in range(segments-1):
        for j in range(segments-1):
            idx0 = i*segments + j
            idx1 = i*segments + (j+1)
            idx2 = (i+1)*segments + j
            idx3 = (i+1)*segments + (j+1)
            faces.append([idx0, idx2, idx1])
            faces.append([idx2, idx3, idx1])
    return trimesh.Trimesh(vertices=verts, faces=np.array(faces))

def save_ellipsoid_cuboid_data(centers, directions, lengths, output_dir):
    """
    保存椭球的cuboid数据
    
    Args:
        centers: 椭球中心点列表
        directions: 椭球方向列表
        lengths: 椭球长轴半径列表
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 计算椭球尺寸
    ellipsoid_centers = []
    ellipsoid_sizes = []
    ellipsoid_types = []
    
    for center, direction, length in zip(centers, directions, lengths):
        # 椭球中心点
        ellipsoid_centers.append(center)
        
        # 椭球尺寸（长轴和短轴）
        long_radius = length
        short_radius = length * 0.3  # 短轴为长轴的30%
        
        # cuboid尺寸：[长轴, 短轴, 短轴]
        cuboid_size = np.array([long_radius, short_radius, short_radius])
        ellipsoid_sizes.append(cuboid_size)
        ellipsoid_types.append('ellipsoid')
    
    # 保存为numpy文件
    cuboid_data = {
        'centers': ellipsoid_centers,
        'sizes': ellipsoid_sizes,
        'types': ellipsoid_types,
        'directions': directions,
        'lengths': lengths
    }
    
    np.save(os.path.join(output_dir, "ellipsoid_cuboid_data.npy"), cuboid_data)
    
    print(f"Saved {len(ellipsoid_centers)} ellipsoid cuboid data points")
    print("Cuboid data:")
    
    
    return ellipsoid_centers, ellipsoid_sizes, ellipsoid_types

if __name__ == "__main__":
    input_path = "/taiga/illinois/eng/ece/n-ahuja/haozhang/tjx/PHYSDREAMER/PhysDreamer/data/dragon_tjx/skeleton_mesh/skeleton_frame_0000.obj"
    output_dir = "/taiga/illinois/eng/ece/n-ahuja/haozhang/tjx/PHYSDREAMER/PhysDreamer/data/dragon_tjx/skeleton_split"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "all_ellipsoids.obj")

    skeleton_mesh = load_mesh(input_path)
    tetra_parts = separate_loose_parts(skeleton_mesh)

    bases, apices = [], []
    for t in tetra_parts:
        base, apex = compute_base_apex(t)
        if base is None:
            continue
        bases.append(base)
        apices.append(apex)

    centers, directions, lengths = adjust_axes_for_intersections(bases, apices)

    all_ellipsoids = []
    for c, d, l in zip(centers, directions, lengths):
        # 短半径设为长半径的30%，让椭球更饱满
        short_radius = l * 0.3
        ellipsoid = create_ellipsoid_mesh(c, l, short_radius, d)
        all_ellipsoids.append(ellipsoid)

    if all_ellipsoids:
        combined = trimesh.util.concatenate(all_ellipsoids)
        combined.export(output_path)
        print(f"Exported {len(all_ellipsoids)} ellipsoids into {output_path}")
        
        # 保存椭球的cuboid数据
        save_ellipsoid_cuboid_data(centers, directions, lengths, output_dir)
    else:
        print("No valid ellipsoids generated")