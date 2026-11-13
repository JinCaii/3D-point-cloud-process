import csv
from pathlib import Path

import numpy as np
import open3d as o3d

# Change this to the real Windows project directory.
folder = Path(r"D:\captures_stop_and_go\forward")
preprocessed_folder = folder / "SOR"
initial_matrix_folder = folder / "initial_registration_matrices"
output_path = folder / "registered_plane_results.csv"

# Units: mm (Zivid PLY and saved initial transformations).
distance_threshold = 3.0
ransac_n = 3
num_iterations = 2000

colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0.706, 0]]
plane_results = []
plane_clouds = []
models = []
inlier_points_list = []


def normal_angle_deg(n1, n2):
    cosine = np.clip(np.abs(np.dot(n1, n2)), 0.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


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

    T = np.load(matrix_path)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        raise ValueError(f"Invalid 4x4 transformation: {matrix_path}")

    # Transform the entire cloud to Capture 001 coordinates BEFORE fitting.
    pcd.transform(T)

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )
    if not inliers:
        raise ValueError(f"No RANSAC inliers for Capture {i:03d}")

    model = np.asarray(plane_model, dtype=np.float64)
    model /= np.linalg.norm(model[:3])
    # Orient normals consistently with the first fitted plane.
    if i == 1:
        if model[2] < 0:
            model = -model
    elif np.dot(model[:3], models[0][:3]) < 0:
        model = -model

    xyz = np.asarray(pcd.points)[inliers]
    own_signed_distances = xyz @ model[:3] + model[3]
    fitting_rmse = float(np.sqrt(np.mean(own_signed_distances ** 2)))

    models.append(model)
    inlier_points_list.append(xyz.copy())
    inlier_ratio = len(inliers) / len(pcd.points)

    plane_cloud = pcd.select_by_index(inliers)
    plane_cloud.paint_uniform_color(colors[i - 1])
    plane_clouds.append(plane_cloud)

    plane_results.append({
        "Capture": i,
        "Normal_X": float(model[0]),
        "Normal_Y": float(model[1]),
        "Normal_Z": float(model[2]),
        "Plane_d_mm": float(model[3]),
        "Number_of_Inliers": len(inliers),
        "Inlier_Ratio": float(inlier_ratio),
        "Plane_Fitting_RMSE_mm": fitting_rmse,
    })

# Evaluate all captures against the plane fitted to Capture 001.
reference = models[0]
for result, model, xyz in zip(plane_results, models, inlier_points_list):
    residuals = xyz @ reference[:3] + reference[3]
    result["Normal_Angle_to_Capture001_deg"] = normal_angle_deg(reference[:3], model[:3])
    result["Signed_Mean_Distance_to_Reference_mm"] = float(np.mean(residuals))
    result["Plane_to_Plane_RMSE_mm"] = float(np.sqrt(np.mean(residuals ** 2)))
    result["Plane_to_Plane_Median_Abs_mm"] = float(np.median(np.abs(residuals)))

    print(f"\nCapture {result['Capture']:03d} (Capture 001 coordinates)")
    print(f"Plane: {model[0]:.6f}x + {model[1]:.6f}y + {model[2]:.6f}z + {model[3]:.6f} = 0")
    print(f"Inliers: {result['Number_of_Inliers']} ({result['Inlier_Ratio']:.2%})")
    print(f"Fitting RMSE: {result['Plane_Fitting_RMSE_mm']:.4f} mm")
    print(f"Normal angle to Capture 001: {result['Normal_Angle_to_Capture001_deg']:.6f} deg")
    print(f"Signed mean distance to Capture 001 plane: {result['Signed_Mean_Distance_to_Reference_mm']:.4f} mm")
    print(f"Plane-to-plane RMSE: {result['Plane_to_Plane_RMSE_mm']:.4f} mm")

with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=list(plane_results[0].keys()))
    writer.writeheader()
    writer.writerows(plane_results)
print(f"\nResults saved to: {output_path}")

# Visualize only the RANSAC plane inliers, colored by capture.
o3d.visualization.draw_geometries(
    plane_clouds,
    window_name="Initial Registration - Four Fitted Board Planes",
    width=1280,
    height=720,
)
