"""Plane-based correction compatible with marker-based initial registration.
All point-cloud coordinates and correction translations are in millimeters.
"""
from pathlib import Path
import copy
import csv
import numpy as np
import open3d as o3d

# ----------------------------- Configuration -----------------------------
folder = Path(r"D:\captures_stop_and_go\forward")
sor_folder = folder / "SOR"
initial_folder = folder / "initial_registration_matrices"
output_folder = folder / "plane_based_correction"
output_folder.mkdir(parents=True, exist_ok=True)

marker_ids = [8, 7, 6, 5]  # Same capture ordering as initial registration
marker_x_positions = [0.0, -0.245, -0.545, -0.845]  # meters, informational
DISTANCE_THRESHOLD_MM = 3.0
RANSAC_N = 3
NUM_ITERATIONS = 2000
VISUALIZE = True
VISUALIZE_RANSAC_EACH_CAPTURE = True
SAVE_CORRECTED_PLY = True
COLORS = [[1, 0, 0], [0, 0.65, 0.93], [1, 0.71, 0], [0.6, 0.2, 0.8]]


def fit_plane(pcd):
    """Fit a plane, returning normalized coefficients, inlier indices, and XYZ."""
    model, indices = pcd.segment_plane(
        distance_threshold=DISTANCE_THRESHOLD_MM,
        ransac_n=RANSAC_N,
        num_iterations=NUM_ITERATIONS,
    )
    if len(indices) < RANSAC_N:
        raise RuntimeError("RANSAC could not identify a valid plane")
    model = np.asarray(model, dtype=np.float64)
    norm = np.linalg.norm(model[:3])
    if norm < 1e-12:
        raise RuntimeError("Invalid plane normal")
    model /= norm
    xyz = np.asarray(pcd.points)[indices]
    return model, indices, xyz


def orient_plane(model, reference_normal):
    """Resolve the sign ambiguity of a plane normal consistently."""
    result = model.copy()
    if np.dot(result[:3], reference_normal) < 0:
        result *= -1
    return result


def minimal_rotation(a, b):
    """Return the smallest 3D rotation mapping unit vector a to b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    cross = np.cross(a, b)
    s = np.linalg.norm(cross)
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        axis = np.eye(3)[np.argmin(np.abs(a))]
        axis = np.cross(a, axis)
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    K = np.array([[0, -cross[2], cross[1]],
                  [cross[2], 0, -cross[0]],
                  [-cross[1], cross[0], 0]])
    return np.eye(3) + K + K @ K * ((1.0 - c) / (s * s))


def plane_metrics(model, inlier_xyz, reference):
    """Evaluate only planar inliers against the reference plane."""
    n = model[:3]
    nr, dr = reference[:3], reference[3]
    angle = np.degrees(np.arccos(np.clip(abs(np.dot(n, nr)), 0, 1)))
    own = inlier_xyz @ n + model[3]
    relative = inlier_xyz @ nr + dr
    return {
        "normal_angle_deg": float(angle),
        "signed_mean_distance_mm": float(relative.mean()),
        "reference_plane_rmse_mm": float(np.sqrt(np.mean(relative**2))),
        "own_plane_rmse_mm": float(np.sqrt(np.mean(own**2))),
    }


# Visualization --------------------------------------------------------
def show_clouds(clouds, title, original_rgb=False):

    display = []

    for pcd, color in zip(clouds, COLORS):

        c = copy.deepcopy(pcd)

        # Only apply uniform color when original_rgb is False
        if not original_rgb:
            c.paint_uniform_color(color)

        display.append(c)

    o3d.visualization.draw_geometries(
        display,
        window_name=title,
        width=1280,
        height=720
    )


# --------------------- Load initial registration results -----------------
initial_clouds = []
initial_transforms = []
for capture in range(1, 5):
    ply_path = sor_folder / f"capture_{capture:03d}_sor.ply"
    matrix_path = initial_folder / f"capture_{capture:03d}_initial.npy"
    if not ply_path.is_file():
        raise FileNotFoundError(f"Missing preprocessed cloud: {ply_path}")
    if not matrix_path.is_file():
        raise FileNotFoundError(f"Missing initial transform: {matrix_path}")
    pcd = o3d.io.read_point_cloud(str(ply_path))
    if pcd.is_empty():
        raise ValueError(f"Empty point cloud: {ply_path}")
    T = np.load(matrix_path).astype(np.float64)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError(f"Invalid initial transform: {matrix_path}")
    if not np.allclose(T[3], [0, 0, 0, 1]):
        raise ValueError(f"Invalid homogeneous matrix last row: {matrix_path}")
    if capture == 1 and not np.allclose(T, np.eye(4), atol=1e-6):
        raise ValueError("Capture 001 must be the global reference (identity transform)")
    pcd.transform(T)
    initial_clouds.append(pcd)
    initial_transforms.append(T)
    print(f"Loaded Capture {capture:03d} (Marker {marker_ids[capture-1]}): {len(pcd.points)} points")

# --------------------- Fit the reference plane ---------------------------
reference_model, _, _ = fit_plane(initial_clouds[0])
# Choose one consistent normal orientation, without assuming camera Z direction.
if reference_model[2] < 0:
    reference_model *= -1
print("\nReference plane Capture 001 [a, b, c, d]:", reference_model)
print("Plane equation: ax + by + cz + d = 0; XYZ and d are in mm")

if VISUALIZE:
    show_clouds(initial_clouds, "Before Plane-Based Correction")

# --------------------- Correct captures 002-004 --------------------------
corrected_clouds = []
evaluation_rows = []
plane_rows = []
for i, initial_pcd in enumerate(initial_clouds):
    capture = i + 1
    before_model, _, before_xyz = fit_plane(initial_pcd)
    before_model = orient_plane(before_model, reference_model[:3])
    before = plane_metrics(before_model, before_xyz, reference_model)

    T_correction = np.eye(4, dtype=np.float64)
    if i > 0:
        R = minimal_rotation(before_model[:3], reference_model[:3])
        centroid = before_xyz.mean(axis=0)
        # Rotation about source inlier centroid; no arbitrary in-plane shift.
        # Place that centroid on the Capture 001 reference plane.
        normal_shift = -(np.dot(reference_model[:3], centroid) + reference_model[3])
        t = centroid - R @ centroid + normal_shift * reference_model[:3]
        T_correction[:3, :3] = R
        T_correction[:3, 3] = t

    corrected = copy.deepcopy(initial_pcd)
    corrected.transform(T_correction)
    corrected_clouds.append(corrected)

    # Independent RANSAC fit on each corrected cloud
    after_model, after_indices, after_xyz = fit_plane(corrected)
    after_model = orient_plane(after_model, reference_model[:3])
    after = plane_metrics(after_model, after_xyz, reference_model)
    ratio = len(after_indices) / len(corrected.points)

    T_final = T_correction @ initial_transforms[i]
    np.save(output_folder / f"capture_{capture:03d}_plane_correction.npy", T_correction)
    np.save(output_folder / f"capture_{capture:03d}_final.npy", T_final)
    if SAVE_CORRECTED_PLY:
        o3d.io.write_point_cloud(
            str(output_folder / f"capture_{capture:03d}_plane_corrected.ply"), corrected
        )

    evaluation_rows.append({
        "capture": capture, "marker_id": marker_ids[i],
        **{f"before_{k}": v for k, v in before.items()},
        **{f"after_{k}": v for k, v in after.items()},
    })
    plane_rows.append({
        "capture": capture,
        "marker_id": marker_ids[i],
        "a": float(after_model[0]), "b": float(after_model[1]),
        "c": float(after_model[2]), "d_mm": float(after_model[3]),
        "inlier_count": len(after_indices), "inlier_ratio": ratio,
        **after,
    })
    print(f"\n--- Capture {capture:03d}, Marker {marker_ids[i]} ---")
    print("Correction matrix (mm):\n", T_correction)
    print("Final matrix (mm):\n", T_final)
    print("Post-correction RANSAC plane [a,b,c,d]:", after_model)
    print("Post-correction unit normal:", after_model[:3])
    print(f"RANSAC inlier ratio: {ratio:.2%}")
    for metric in before:
        print(f"{metric}: {before[metric]:.5f} -> {after[metric]:.5f}")

    if VISUALIZE and VISUALIZE_RANSAC_EACH_CAPTURE:
        plane_inliers = corrected.select_by_index(after_indices)
        remaining = corrected.select_by_index(after_indices, invert=True)
        plane_inliers.paint_uniform_color([0, 0.8, 0.1])
        remaining.paint_uniform_color([0.9, 0.15, 0.15])
        o3d.visualization.draw_geometries(
            [plane_inliers, remaining],
            window_name=f"Capture {capture:03d}: RANSAC inliers (green) / others (red)"
        )

for filename, rows in [
    ("plane_correction_evaluation.csv", evaluation_rows),
    ("post_correction_ransac_planes.csv", plane_rows),
]:
    with (output_folder / filename).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

# Visualization after Plane-Based Correction --------------------------

if VISUALIZE:

    # 1. Corrected point clouds with different colors
    show_clouds(
        corrected_clouds,
        "After Plane-Based Correction - Colored",
        original_rgb=False
    )

    # 2. Corrected point clouds with original RGB colors
    show_clouds(
        corrected_clouds,
        "After Plane-Based Correction - Original RGB",
        original_rgb=True
    )

print("\nFinished. Outputs:", output_folder)