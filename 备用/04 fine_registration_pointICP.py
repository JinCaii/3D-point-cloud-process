import copy
from pathlib import Path

import numpy as np
import open3d as o3d

# Paths (Windows)
folder = Path(r"D:\captures_stop_and_go\forward")
preprocessed_folder = folder / "SOR"
initial_matrix_folder = folder / "initial_registration_matrices"
output_folder = folder / "ICP"
output_matrix_folder = output_folder / "transformation_matrices"
output_folder.mkdir(parents=True, exist_ok=True)
output_matrix_folder.mkdir(parents=True, exist_ok=True)

# All distances are in mm because the input PLY coordinates are in mm.
max_correspondence_distance = 5
max_iterations = 200


def evaluate(source, target, transform):
    return o3d.pipelines.registration.evaluate_registration(
        source, target, max_correspondence_distance, transform
    )


def show_alignment(point_clouds, title, use_capture_colors=False):
    displayed = [copy.deepcopy(pcd) for pcd in point_clouds]
    if use_capture_colors:
        colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0.706, 0]]
        for pcd, color in zip(displayed, colors):
            pcd.paint_uniform_color(color)
    o3d.visualization.draw_geometries(displayed, window_name=title)


# Load original (preprocessed) point clouds in their respective camera frames.
pcds = []
initial_global = []
for i in range(1, 5):
    ply_path = preprocessed_folder / f"capture_{i:03d}_sor.ply"
    matrix_path = initial_matrix_folder / f"capture_{i:03d}_initial.npy"
    if not ply_path.is_file():
        raise FileNotFoundError(ply_path)
    if not matrix_path.is_file():
        raise FileNotFoundError(matrix_path)
    pcd = o3d.io.read_point_cloud(str(ply_path))
    if pcd.is_empty():
        raise ValueError(f"Empty point cloud: {ply_path}")
    transform = np.load(matrix_path)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"Invalid transformation: {matrix_path}")
    pcds.append(pcd)
    initial_global.append(transform)
    print(f"Loaded capture {i:03d}: {len(pcd.points)} points")

# Global reference = capture 001. Each pair is registered in the original
# camera coordinates, then chained into the global reference frame.
refined_global = [np.eye(4, dtype=np.float64)]
for i in range(1, len(pcds)):
    source = pcds[i]
    target = pcds[i - 1]

    # Initial source -> previous capture transformation (mm).
    initial_pair = np.linalg.inv(initial_global[i - 1]) @ initial_global[i]
    before = evaluate(source, target, initial_pair)

    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        max_correspondence_distance,
        initial_pair,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations),
    )
    # The returned transformation maps original source -> original target.
    refined_pair = result.transformation
    refined_global_transform = refined_global[i - 1] @ refined_pair
    refined_global.append(refined_global_transform)

    print(f"\nCapture {i + 1:03d} -> Capture {i:03d}")
    print(f"Before ICP: fitness={before.fitness:.6f}, RMSE={before.inlier_rmse:.6f} mm")
    print(f"After  ICP: fitness={result.fitness:.6f}, RMSE={result.inlier_rmse:.6f} mm")
    print(f"Correspondences: {len(result.correspondence_set)}")
    print("Refined pair transformation (mm):\n", refined_pair)
    print("Refined global transformation (mm):\n", refined_global_transform)

    np.save(output_matrix_folder / f"capture_{i + 1:03d}_to_{i:03d}_icp.npy", refined_pair)

# Transform all original preprocessed point clouds to capture 001 frame.
aligned_pcds = []
for i, (pcd, transform) in enumerate(zip(pcds, refined_global), start=1):
    np.save(output_matrix_folder / f"capture_{i:03d}_icp_global.npy", transform)
    aligned = copy.deepcopy(pcd)
    aligned.transform(transform)
    aligned_pcds.append(aligned)
    path = output_folder / f"capture_{i:03d}_icp_aligned.ply"
    if not o3d.io.write_point_cloud(str(path), aligned):
        raise IOError(f"Failed to save {path}")
    print(f"Saved: {path}")

show_alignment(aligned_pcds, "ICP - Different Capture Colors", use_capture_colors=True)
show_alignment(aligned_pcds, "ICP - Original RGB Colors")
print("\nICP Fine Registration completed!")
