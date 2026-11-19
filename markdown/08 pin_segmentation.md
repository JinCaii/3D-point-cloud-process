"""Segment protruding pins from the fused point cloud and export centroids.

Requirements: numpy, open3d
Coordinates and thresholds: millimetres, matching 07 fusion.
"""
from pathlib import Path
import csv
import numpy as np
import open3d as o3d

ROOT = Path(r"D:\captures_stop_and_go\forward")
INPUT_PATH = ROOT / "Fusion_After_Pin_Refinement" / "fused_voxel_downsampled.ply"
OUTPUT_DIR = ROOT / "Pin_Segmentation_After_Fusion"

HEIGHT_THRESHOLD_MM = 6.0
PLANE_DISTANCE_THRESHOLD_MM = 2.0
PLANE_RANSAC_N = 3
PLANE_ITERATIONS = 1500
PLANE_FIT_VOXEL_MM = 3.0  # Fit on a lightweight copy; segment the actual input.

# Plane normal must point toward the protruding pins. Set to -1 if reversed.
# +1 chooses the normal with positive Z component; -1 chooses negative Z.
PIN_SIDE_Z_SIGN = -1    #RANSAC拟合的平面不保证方向，这里的-1是法向量负方向的意思

# Clustering separates different pins; tune to the point spacing / pin spacing.
DBSCAN_EPS_MM = 5.0
DBSCAN_MIN_POINTS = 12
MIN_CLUSTER_POINTS = 40

SHOW_HEIGHT_FILTERED = True
SHOW_CLUSTERED = True
SHOW_ORIGINAL_RGB_PINS = True
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 800


def save_cloud(path, cloud):
    if not o3d.io.write_point_cloud(str(path), cloud, write_ascii=False):
        raise IOError(f"Could not save: {path}")
    print(f"Saved: {path} ({len(cloud.points):,} points)")


def show(cloud, title, extras=None):
    geometries = [cloud] + (extras or [])
    o3d.visualization.draw_geometries(
        geometries, window_name=title,
        width=WINDOW_WIDTH, height=WINDOW_HEIGHT)


def fit_workpiece_plane(cloud):
    fit_cloud = (cloud.voxel_down_sample(PLANE_FIT_VOXEL_MM)
                 if PLANE_FIT_VOXEL_MM is not None else cloud)
    if len(fit_cloud.points) < 3:
        raise ValueError("Not enough points for plane fitting")
    model, inliers = fit_cloud.segment_plane(
        distance_threshold=PLANE_DISTANCE_THRESHOLD_MM,
        ransac_n=PLANE_RANSAC_N,
        num_iterations=PLANE_ITERATIONS)
    if len(inliers) < 3:
        raise RuntimeError("RANSAC did not find a plane")
    normal = np.asarray(model[:3], dtype=np.float64)
    norm = np.linalg.norm(normal)
    if norm < 1e-12:
        raise RuntimeError("Invalid plane normal")
    normal /= norm
    offset = float(model[3]) / norm
    if normal[2] * PIN_SIDE_Z_SIGN < 0:
        normal = -normal
        offset = -offset
    print(f"Plane: {normal[0]:.6f}x + {normal[1]:.6f}y + "
          f"{normal[2]:.6f}z + {offset:.6f} = 0")
    print(f"Plane inliers (fit copy): {len(inliers):,}/{len(fit_cloud.points):,}")
    return normal, offset


def main():
    if HEIGHT_THRESHOLD_MM <= 0 or DBSCAN_EPS_MM <= 0:
        raise ValueError("Height threshold and DBSCAN eps must be positive")
    if PIN_SIDE_Z_SIGN not in (-1, 1):
        raise ValueError("PIN_SIDE_Z_SIGN must be +1 or -1")
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"Run 07 fusion first: {INPUT_PATH}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cloud = o3d.io.read_point_cloud(str(INPUT_PATH))
    if cloud.is_empty():
        raise ValueError("Input cloud is empty")
    print(f"Input: {len(cloud.points):,} points; RGB={cloud.has_colors()}")

    normal, offset = fit_workpiece_plane(cloud)
    xyz = np.asarray(cloud.points)
    signed_height = xyz @ normal + offset  # normal is unit length; units = mm
    selected_idx = np.flatnonzero(signed_height > HEIGHT_THRESHOLD_MM)
    if len(selected_idx) == 0:
        raise RuntimeError("No points above 6 mm. Check PIN_SIDE_Z_SIGN / plane fit")
    candidates = cloud.select_by_index(selected_idx.tolist())
    print(f"Height > {HEIGHT_THRESHOLD_MM:g} mm: {len(selected_idx):,} points")
    save_cloud(OUTPUT_DIR / "pins_height_filtered_rgb.ply", candidates)
    if SHOW_HEIGHT_FILTERED:
        show(candidates, "Height filtered > 6 mm (original RGB)")

    labels = np.asarray(candidates.cluster_dbscan(
        eps=DBSCAN_EPS_MM, min_points=DBSCAN_MIN_POINTS,
        print_progress=True))
    valid_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
    kept_labels = valid_labels[counts >= MIN_CLUSTER_POINTS]
    if len(kept_labels) == 0:
        print("No clusters passed the size threshold; try adjusting DBSCAN parameters")
        return

    # Stable ordering by centroid X, then Y.
    candidate_xyz = np.asarray(candidates.points)
    groups = []
    for label in kept_labels:
        indices = np.flatnonzero(labels == label)
        centroid = candidate_xyz[indices].mean(axis=0)
        groups.append((centroid, indices))
    groups.sort(key=lambda item: (item[0][0], item[0][1]))

    cluster_view = o3d.geometry.PointCloud()
    rgb_pins = o3d.geometry.PointCloud()
    rows = []
    marker_geometries = []
    palette = np.asarray([
        [1.0, 1.0, 0.0],  #pin的颜色 黄色
        [0.0, 1.0, 1.0]  #青色
    ])
    for pin_id, (centroid, indices) in enumerate(groups, start=1):
        part = candidates.select_by_index(indices.tolist())
        rgb_pins += part
        colored = o3d.geometry.PointCloud(part)
        colored.paint_uniform_color(palette[(pin_id - 1) % len(palette)])
        cluster_view += colored
        save_cloud(OUTPUT_DIR / f"pin_{pin_id:03d}.ply", part)
        rows.append({
            "pin_id": pin_id,
            "centroid_x_mm": float(centroid[0]),
            "centroid_y_mm": float(centroid[1]),
            "centroid_z_mm": float(centroid[2]),
            "point_count": len(indices),
            "mean_height_above_plane_mm": float(np.mean(candidate_xyz[indices] @ normal + offset)),
        })
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=3.0)
        sphere.translate(centroid)
        sphere.paint_uniform_color([1.0, 0.0, 0.0]) #质心的颜色：红色
        marker_geometries.append(sphere)
        print(f"Pin {pin_id:03d}: centroid = ({centroid[0]:.3f}, "
              f"{centroid[1]:.3f}, {centroid[2]:.3f}) mm; points={len(indices):,}")

    csv_path = OUTPUT_DIR / "pin_centroids.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {csv_path}")
    save_cloud(OUTPUT_DIR / "pins_segmented_rgb.ply", rgb_pins)
    save_cloud(OUTPUT_DIR / "pins_segmented_by_cluster.ply", cluster_view)
    if SHOW_CLUSTERED:
        show(cluster_view, "Pin segmentation - colors by pin", marker_geometries)
    if SHOW_ORIGINAL_RGB_PINS:
        show(rgb_pins, "Segmented pins - original RGB", marker_geometries)
    print(f"Done: {len(rows)} pin clusters. Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
