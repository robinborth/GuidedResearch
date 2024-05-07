import numpy as np
import open3d as o3d


def load_pcd(path, color=None):
    if color is None:
        color = [255, 0, 0]
    points = np.load(path)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    color = [np.array(color, dtype=np.uint8)] * points.shape[0]
    pcd.colors = o3d.utility.Vector3dVector(np.stack(color))
    return pcd
