import copy
from pathlib import Path

import numpy as np
import open3d as o3d

# Paths (Windows)
folder = Path(r"D:\captures_stop_and_go\forward")

# Directly use Plane-based Correction results
plane_folder = folder / "plane_based_correction"

output_folder = folder / "ICP_after_plane_correction"
output_matrix_folder = output_folder / "transformation_matrices"

output_folder.mkdir(parents=True, exist_ok=True)
output_matrix_folder.mkdir(parents=True, exist_ok=True)

# All distances in mm
max_correspondence_distance = 5
max_iterations = 100


def evaluate(source, target, transform):
    return o3d.pipelines.registration.evaluate_registration(
        source,
        target,
        max_correspondence_distance,
        transform
    )


def show_alignment(point_clouds, title, use_capture_colors=False):
    displayed = [copy.deepcopy(pcd) for pcd in point_clouds]

    if use_capture_colors:
        colors = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0.706, 0]
        ]
        for pcd, color in zip(displayed, colors):
            pcd.paint_uniform_color(color)

    o3d.visualization.draw_geometries(
        displayed,
        window_name=title
    )


# --------------------------------------------------
# 1. Load Plane-based Correction results
# --------------------------------------------------

pcds = []
plane_global = []

for i in range(1, 5):

    ply_path = (
        plane_folder /
        f"capture_{i:03d}_plane_corrected.ply"
    )

    matrix_path = (
        plane_folder /
        f"capture_{i:03d}_final.npy"
    )

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
    plane_global.append(transform)

    print(f"Loaded capture {i:03d}: {len(pcd.points)} points")


# --------------------------------------------------
# 2. ICP Fine Registration
# --------------------------------------------------

# Capture 001 is the global reference.
# All loaded point clouds are already in the
# Plane-based Correction common coordinate frame.

refined_global = [np.eye(4, dtype=np.float64)]

for i in range(1, len(pcds)):

    source = pcds[i]
    target = pcds[i - 1]

    # Both clouds are already in the same coordinate frame.
    initial_pair = np.eye(4, dtype=np.float64)

    before = evaluate(source, target, initial_pair)

    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        max_correspondence_distance,
        initial_pair,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iterations
        ),
    )

    # ICP transformation: source -> previous capture
    refined_pair = result.transformation

    # Chain transformations into Capture 001 frame
    refined_global_transform = (
        refined_global[i - 1] @ refined_pair
    )

    refined_global.append(refined_global_transform)

    print(f"\nCapture {i + 1:03d} -> Capture {i:03d}")

    print(
        f"Before ICP: fitness={before.fitness:.6f}, "
        f"RMSE={before.inlier_rmse:.6f} mm"
    )

    print(
        f"After  ICP: fitness={result.fitness:.6f}, "
        f"RMSE={result.inlier_rmse:.6f} mm"
    )

    print(f"Correspondences: {len(result.correspondence_set)}")

    print("Refined pair transformation (mm):\n", refined_pair)

    print(
        "Refined global transformation (mm):\n",
        refined_global_transform
    )

    np.save(
        output_matrix_folder /
        f"capture_{i + 1:03d}_to_{i:03d}_icp.npy",
        refined_pair
    )


# --------------------------------------------------
# 3. Transform and save aligned point clouds
# --------------------------------------------------

aligned_pcds = []

for i, (pcd, transform) in enumerate(
    zip(pcds, refined_global),
    start=1
):

    # ICP correction relative to Plane-based Correction
    np.save(
        output_matrix_folder /
        f"capture_{i:03d}_icp_global.npy",
        transform
    )

    # Complete transformation:
    # Original camera frame -> final ICP frame
    T_total = transform @ plane_global[i - 1]

    np.save(
        output_matrix_folder /
        f"capture_{i:03d}_total.npy",
        T_total
    )

    aligned = copy.deepcopy(pcd)
    aligned.transform(transform)

    aligned_pcds.append(aligned)

    path = (
        output_folder /
        f"capture_{i:03d}_icp_aligned.ply"
    )

    if not o3d.io.write_point_cloud(str(path), aligned):
        raise IOError(f"Failed to save {path}")

    print(f"Saved: {path}")


# --------------------------------------------------
# 4. Visualization
# --------------------------------------------------

show_alignment(
    aligned_pcds,
    "ICP after Plane Correction - Different Capture Colors",
    use_capture_colors=True
)

show_alignment(
    aligned_pcds,
    "ICP after Plane Correction - Original RGB Colors"
)

print("\nICP Fine Registration completed!")