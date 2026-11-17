import copy
import csv
from pathlib import Path

import numpy as np
import open3d as o3d

# Paths (Windows)
folder = Path(r"D:\captures_stop_and_go\forward")

# Directly use Plane-based Correction results
plane_folder = folder / "plane_based_correction"

output_folder = folder / "ICP_after_plane_correction"
output_matrix_folder = output_folder / "transformation_matrices"
metrics_folder = output_folder / "iteration_metrics"

output_folder.mkdir(parents=True, exist_ok=True)
output_matrix_folder.mkdir(parents=True, exist_ok=True)
metrics_folder.mkdir(parents=True, exist_ok=True)

# All distances in mm
max_correspondence_distance = 5
max_iterations = 200

# Tensor ICP callback records metrics for each iteration.
# Only the final transformation is saved, as in the original script.


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

    # Tensor ICP supports a per-iteration callback (Legacy ICP does not).
    source_t = o3d.t.geometry.PointCloud.from_legacy(source)
    target_t = o3d.t.geometry.PointCloud.from_legacy(target)
    history = [{
        "iteration": 0,
        "fitness": float(before.fitness),
        "rmse_mm": float(before.inlier_rmse),
    }]

    def on_iteration(info):
        iteration = int(info["iteration_index"].item()) + 1
        fitness = float(info["fitness"].item())
        rmse = float(info["inlier_rmse"].item())
        history.append({
            "iteration": iteration,
            "fitness": fitness,
            "rmse_mm": rmse,
        })

    result = o3d.t.pipelines.registration.icp(
        source_t,
        target_t,
        max_correspondence_distance,
        o3d.core.Tensor(initial_pair, dtype=o3d.core.Dtype.Float64),
        o3d.t.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.t.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-12,
            relative_rmse=1e-12,
            max_iteration=max_iterations,
        ),
        voxel_size=-1.0,
        callback_after_iteration=on_iteration,
    )

    # Final transformation only (not saved for individual iterations).
    refined_pair = result.transformation.cpu().numpy().copy()

    csv_path = metrics_folder / f"capture_{i + 1:03d}_to_{i:03d}_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["iteration", "fitness", "rmse_mm"])
        writer.writeheader()
        writer.writerows(history)

    # Exclude non-finite RMSE (e.g. if no correspondences exist).
    valid = [r for r in history if np.isfinite(r["rmse_mm"]) and np.isfinite(r["fitness"])]
    if not valid:
        raise RuntimeError(f"No valid ICP metrics for capture {i + 1:03d} -> {i:03d}")
    best_fitness = max(valid, key=lambda r: (r["fitness"], -r["rmse_mm"], -r["iteration"]))
    best_rmse = min(valid, key=lambda r: (r["rmse_mm"], -r["fitness"], r["iteration"]))

    print(f"\nCapture {i + 1:03d} -> Capture {i:03d}")
    print(f"Before ICP: fitness={before.fitness:.6f}, RMSE={before.inlier_rmse:.6f} mm")
    print(f"After  ICP: fitness={float(result.fitness):.6f}, RMSE={float(result.inlier_rmse):.6f} mm")
    print(f"BEST FITNESS: iteration={best_fitness['iteration']}, "
          f"fitness={best_fitness['fitness']:.6f}, RMSE={best_fitness['rmse_mm']:.6f} mm")
    print(f"BEST RMSE:    iteration={best_rmse['iteration']}, "
          f"fitness={best_rmse['fitness']:.6f}, RMSE={best_rmse['rmse_mm']:.6f} mm")
    print(f"Metrics saved: {csv_path}")

    # Chain transformations into Capture 001 frame
    refined_global_transform = (
        refined_global[i - 1] @ refined_pair
    )

    refined_global.append(refined_global_transform)

    print("Refined pair transformation (mm):\n", refined_pair)
    print("Refined global transformation (mm):\n", refined_global_transform)

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