import copy
import csv
from pathlib import Path

import numpy as np
import open3d as o3d

# Paths (Windows)
folder = Path(r"D:\captures_stop_and_go\forward")
plane_folder = folder / "plane_based_correction"
output_folder = folder / "ICP_Point_to_Plane_after_plane_correction"
output_matrix_folder = output_folder / "transformation_matrices"
history_folder = output_folder / "iteration_history"
for path in (output_folder, output_matrix_folder, history_folder):
    path.mkdir(parents=True, exist_ok=True)

# Units: mm
max_correspondence_distance = 5.0
max_iterations = 100
normal_radius = 10.0
normal_max_nn = 30



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


pcds = []
plane_global = []
for i in range(1, 5):
    ply_path = plane_folder / f"capture_{i:03d}_plane_corrected.ply"
    matrix_path = plane_folder / f"capture_{i:03d}_final.npy"
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
    if not np.allclose(transform[3], [0, 0, 0, 1]):
        raise ValueError(f"Invalid homogeneous matrix: {matrix_path}")
    pcds.append(pcd)
    plane_global.append(transform)
    print(f"Loaded capture {i:03d}: {len(pcd.points)} points")

print("\nEstimating normals...")
for i, pcd in enumerate(pcds, start=1):
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius, max_nn=normal_max_nn
        )
    )
    pcd.normalize_normals()
    print(f"Normals estimated for capture {i:03d}")

refined_global = [np.eye(4, dtype=np.float64)]
for i in range(1, len(pcds)):
    source, target = pcds[i], pcds[i - 1]
    initial_pair = np.eye(4, dtype=np.float64)
    before = evaluate(source, target, initial_pair)
    pair_name = f"capture_{i + 1:03d}_to_{i:03d}"

    # Convert Legacy point clouds to Tensor point clouds.
    source_t = o3d.t.geometry.PointCloud.from_legacy(source)
    target_t = o3d.t.geometry.PointCloud.from_legacy(target)
    history = []

    def callback_after_iteration(info):
        # Tensor ICP reports zero-based iteration_index.
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
        o3d.t.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.t.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-12,
            relative_rmse=1e-12,
            max_iteration=max_iterations,
        ),
        voxel_size=-1.0,
        callback_after_iteration=callback_after_iteration,
    )

    final_pair = np.asarray(result.transformation.cpu().numpy(), dtype=np.float64).copy()
    final_eval = evaluate(source, target, final_pair)

    # Evaluate iteration 0 too, using the same fixed evaluation threshold.
    # Keep callback metrics as reported by Open3D; also save legacy evaluation
    # to make initial and final results directly comparable.
    with (history_folder / f"{pair_name}_metrics.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["iteration", "fitness", "inlier_rmse_mm"])
        writer.writerow([0, before.fitness, before.inlier_rmse])
        for item in history:
            writer.writerow([item["iteration"], item["fitness"], item["rmse_mm"]])

    # Include iteration 0 in the metric comparison.
    candidates = [{
        "iteration": 0,
        "fitness": float(before.fitness),
        "rmse_mm": float(before.inlier_rmse),
    }] + history
    valid = [h for h in candidates if np.isfinite(h["rmse_mm"]) and h["fitness"] > 0]
    if not valid:
        raise RuntimeError(f"No valid registration metrics for {pair_name}")

    best_fitness = max(valid, key=lambda h: (h["fitness"], -h["rmse_mm"], -h["iteration"]))
    best_rmse = min(valid, key=lambda h: (h["rmse_mm"], -h["fitness"], h["iteration"]))

    # Preserve the original workflow: only the FINAL ICP transform is used.
    refined_pair = final_pair

    refined_global_transform = refined_global[i - 1] @ refined_pair
    refined_global.append(refined_global_transform)
    icp_correction = refined_pair @ np.linalg.inv(initial_pair)
    translation_correction = np.linalg.norm(icp_correction[:3, 3])
    rotation = icp_correction[:3, :3]
    cos_angle = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    rotation_correction = np.degrees(np.arccos(cos_angle))

    print(f"\nCapture {i + 1:03d} -> Capture {i:03d}")
    print(f"Before ICP: fitness={before.fitness:.6f}, RMSE={before.inlier_rmse:.6f} mm")
    print(f"Final  ICP: fitness={final_eval.fitness:.6f}, RMSE={final_eval.inlier_rmse:.6f} mm")
    print(f"Callback records: {len(history)}")
    print(f"BEST FITNESS: iteration={best_fitness['iteration']}, fitness={best_fitness['fitness']:.6f}, RMSE={best_fitness['rmse_mm']:.6f} mm")
    print(f"BEST RMSE:    iteration={best_rmse['iteration']}, fitness={best_rmse['fitness']:.6f}, RMSE={best_rmse['rmse_mm']:.6f} mm")
    print(f"ICP translation correction: {translation_correction:.3f} mm")
    print(f"ICP rotation correction: {rotation_correction:.4f} degrees")
    print("Final pair transformation (mm):\n", refined_pair)
    print("Refined global transformation (mm):\n", refined_global_transform)

    np.save(output_matrix_folder / f"{pair_name}_icp.npy", refined_pair)

aligned_pcds = []
for i, (pcd, transform) in enumerate(zip(pcds, refined_global), start=1):
    np.save(output_matrix_folder / f"capture_{i:03d}_icp_global.npy", transform)
    total = transform @ plane_global[i - 1]
    np.save(output_matrix_folder / f"capture_{i:03d}_total.npy", total)
    aligned = copy.deepcopy(pcd)
    aligned.transform(transform)
    aligned_pcds.append(aligned)
    path = output_folder / f"capture_{i:03d}_icp_aligned.ply"
    if not o3d.io.write_point_cloud(str(path), aligned):
        raise IOError(f"Failed to save {path}")
    print(f"Saved: {path}")

show_alignment(aligned_pcds, "Point-to-Plane ICP - Different Capture Colors", True)
show_alignment(aligned_pcds, "Point-to-Plane ICP - Original RGB Colors")
print("\nPoint-to-Plane ICP Fine Registration completed!")
