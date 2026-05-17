import numpy as np
import trimesh
from scipy.stats import skew, kurtosis as scipy_kurtosis

def fix_mesh(mesh):
    mesh.process(validate=True)
    mesh.fill_holes()
    return mesh

def compute_principal_curvatures(mesh):
    """κ1 & κ2"""
    mesh = fix_mesh(mesh)
    vertices = np.array(mesh.vertices, dtype=np.float64)
    faces = np.array(mesh.faces, dtype=np.int64)
    N = len(vertices)
    vnormals = np.array(mesh.vertex_normals, dtype=np.float64)
    
    T_accum = np.zeros((N, 3))
    w_accum = np.zeros(N)
    
    for i in range(3):
        ia, ib = faces[:, i], faces[:, (i + 1) % 3]
        va, vb = vertices[ia], vertices[ib]
        na, nb = vnormals[ia], vnormals[ib]
        edge = vb - va
        edge_len = np.linalg.norm(edge, axis=1, keepdims=True)
        edge_dir = np.divide(edge, edge_len, out=np.zeros_like(edge), where=edge_len > 1e-12)
        dn = nb - na
        kappa = np.sum(dn * edge_dir, axis=1)
        t1 = edge_dir - np.sum(edge_dir * na, axis=1, keepdims=True) * na
        t1_norm = np.linalg.norm(t1, axis=1, keepdims=True)
        t1 = np.divide(t1, t1_norm, out=np.zeros_like(t1), where=t1_norm > 1e-12)
        t2 = np.cross(na, t1)
        e_coord, f_coord = np.sum(edge_dir * t1, axis=1), np.sum(edge_dir * t2, axis=1)
        w = 1.0
        np.add.at(T_accum[:, 0], ia, w * kappa * e_coord * e_coord)
        np.add.at(T_accum[:, 1], ia, w * kappa * e_coord * f_coord)
        np.add.at(T_accum[:, 2], ia, w * kappa * f_coord * f_coord)
        np.add.at(w_accum, ia, w)

    valid = w_accum > 0
    T00, T01, T11 = T_accum[valid, 0]/w_accum[valid], T_accum[valid, 1]/w_accum[valid], T_accum[valid, 2]/w_accum[valid]
    half_trace, half_diff = (T00 + T11) / 2.0, (T00 - T11) / 2.0
    discriminant = np.sqrt(np.maximum(half_diff**2 + T01**2, 0.0))
    k1, k2 = np.zeros(N), np.zeros(N)
    k1[valid], k2[valid] = half_trace + discriminant, half_trace - discriminant
    return k1, k2

def get_stats_features(values, n_bins=20):
    values = values[np.isfinite(values)]
    if len(values) == 0: return np.zeros(7 + n_bins)
    v_min, v_max = np.percentile(values, [1, 99])
    v_clipped = np.clip(values, v_min, v_max)
    stats = [np.mean(v_clipped), np.std(v_clipped), float(skew(v_clipped)), float(scipy_kurtosis(v_clipped)), np.min(v_clipped), np.max(v_clipped), np.median(v_clipped)]
    hist, _ = np.histogram(v_clipped, bins=n_bins, density=True)
    return np.concatenate([stats, hist])

def extract_all_descriptors(mesh):
    k1, k2 = compute_principal_curvatures(mesh)
    K, H = k1 * k2, (k1 + k2) / 2.0
    return {
        'shape_operator': np.concatenate([get_stats_features(k1), get_stats_features(k2)]),
        'gaussian': get_stats_features(K),
        'mean': get_stats_features(H)
    }
