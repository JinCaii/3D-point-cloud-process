
import copy
from pathlib import Path

import numpy as np
import open3d as o3d


#Paths (Windows) --------------------------------------------------------------
folder = Path(r"D:\captures_stop_and_go\forward")
preprocessed_folder = folder / "SOR"
initial_matrix_folder = folder / "initial_registration_matrices"

output_folder = folder / "ICP_Point_to_Plane"
output_matrix_folder = output_folder / "transformation_matrices"

output_folder.mkdir(parents=True, exist_ok=True)
output_matrix_folder.mkdir(parents=True, exist_ok=True)


#ICP parameters ---------------------------------------------------------------
#All distances are in mm because the input PLY coordinates are in mm.

max_correspondence_distance = 5
max_iterations = 100

#Normal estimation parameters (mm)
normal_radius = 10.0
normal_max_nn = 30


# Evaluate registration --------------------------------------------------------
def evaluate(source, target, transform):
    return o3d.pipelines.registration.evaluate_registration(
        source,
        target,
        max_correspondence_distance,
        transform
    )


# Visualization ---------------------------------------------------------------
def show_alignment(point_clouds, title, use_capture_colors=False):

    displayed = [copy.deepcopy(pcd) for pcd in point_clouds]

    if use_capture_colors:
        colors = [
            [1, 0, 0],        # Capture 001: Red
            [0, 1, 0],        # Capture 002: Green
            [0, 0, 1],        # Capture 003: Blue
            [1, 0.706, 0]     # Capture 004: Orange
        ]

        for pcd, color in zip(displayed, colors):
            pcd.paint_uniform_color(color)

    o3d.visualization.draw_geometries(
        displayed,
        window_name=title
    )


# Load preprocessed point clouds and initial transformations ------------------
pcds = []
initial_global = []

for i in range(1, 5):

    ply_path = preprocessed_folder / f"capture_{i:03d}_sor.ply"
    matrix_path = initial_matrix_folder / f"capture_{i:03d}_initial.npy"

    if not ply_path.is_file():
        raise FileNotFoundError(ply_path)

    if not matrix_path.is_file():
        raise FileNotFoundError(matrix_path)

    # Load point cloud in original camera coordinates
    pcd = o3d.io.read_point_cloud(str(ply_path))

    if pcd.is_empty():
        raise ValueError(f"Empty point cloud: {ply_path}")

    # Load initial global transformation
    transform = np.load(matrix_path)

    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"Invalid transformation: {matrix_path}")

    pcds.append(pcd)
    initial_global.append(transform)

    print(f"Loaded capture {i:03d}: {len(pcd.points)} points")


# Estimate normals -------------------------------------------------------------
# Point-to-Plane ICP requires target point cloud normals.

print("\nEstimating normals...")

for i, pcd in enumerate(pcds, start=1):

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius,
            max_nn=normal_max_nn
        )
    )

    pcd.normalize_normals()

    print(f"Normals estimated for capture {i:03d}")


# Sequential Point-to-Plane ICP ------------------------------------------------
# Global reference = Capture 001
# Pairwise registration:
# 002 -> 001
# 003 -> 002
# 004 -> 003

refined_global = [np.eye(4, dtype=np.float64)]

for i in range(1, len(pcds)):

    source = pcds[i]
    target = pcds[i - 1]

    # Initial source -> previous capture transformation
    initial_pair = (
        np.linalg.inv(initial_global[i - 1])
        @ initial_global[i]
    )

    # Evaluate before ICP
    before = evaluate(source, target, initial_pair)

    # Point-to-Plane ICP
    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        max_correspondence_distance,
        initial_pair,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iterations
        )
    )

    # Refined pair transformation
    # Maps original source -> original target
    refined_pair = result.transformation

    # Refined global transformation
    refined_global_transform = (
        refined_global[i - 1] @ refined_pair
    )

    refined_global.append(refined_global_transform)

    # ICP correction relative to initial registration
    icp_correction = (
        refined_pair @ np.linalg.inv(initial_pair)
    )

    # Translation correction (mm)
    translation_correction = np.linalg.norm(
        icp_correction[:3, 3]
    )

    # Rotation correction (degrees)
    R = icp_correction[:3, :3]

    cos_angle = (np.trace(R) - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    rotation_correction = np.degrees(
        np.arccos(cos_angle)
    )

    # Print registration results
    print(f"\nCapture {i + 1:03d} -> Capture {i:03d}")

    print(
        f"Before ICP: fitness={before.fitness:.6f}, "
        f"RMSE={before.inlier_rmse:.6f} mm"
    )

    print(
        f"After  ICP: fitness={result.fitness:.6f}, "
        f"RMSE={result.inlier_rmse:.6f} mm"
    )

    print(
        f"Correspondences: {len(result.correspondence_set)}"
    )

    print(
        f"ICP translation correction: "
        f"{translation_correction:.3f} mm"
    )

    print(
        f"ICP rotation correction: "
        f"{rotation_correction:.4f} degrees"
    )

    print(
        "Refined pair transformation (mm):\n",
        refined_pair
    )

    print(
        "Refined global transformation (mm):\n",
        refined_global_transform
    )

    # Save refined pair transformation
    np.save(
        output_matrix_folder
        / f"capture_{i + 1:03d}_to_{i:03d}_icp.npy",
        refined_pair
    )


# Transform all point clouds into Capture 001 frame ---------------------------
aligned_pcds = []

for i, (pcd, transform) in enumerate(
    zip(pcds, refined_global),
    start=1
):

    # Save global transformation
    np.save(
        output_matrix_folder
        / f"capture_{i:03d}_icp_global.npy",
        transform
    )

    # Apply global transformation to original preprocessed point cloud
    aligned = copy.deepcopy(pcd)
    aligned.transform(transform)

    aligned_pcds.append(aligned)

    # Save aligned point cloud with original RGB colors
    path = output_folder / f"capture_{i:03d}_icp_aligned.ply"

    if not o3d.io.write_point_cloud(str(path), aligned):
        raise IOError(f"Failed to save {path}")

    print(f"Saved: {path}")


# Visualization ---------------------------------------------------------------
show_alignment(
    aligned_pcds,
    "Point-to-Plane ICP - Different Capture Colors",
    use_capture_colors=True
)

show_alignment(
    aligned_pcds,
    "Point-to-Plane ICP - Original RGB Colors"
)

print("\nPoint-to-Plane ICP Fine Registration completed!")
